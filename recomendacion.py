"""LLM abierto en el servidor: no usa API de pago ni envía la imagen."""
from functools import lru_cache
from threading import Lock
import re
import torch

LLM_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
# Fijar revisión hace reproducible la descarga del modelo público.
LLM_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
GENERACION_LOCK = Lock()
SISTEMA = '''Eres un editor de mensajes educativos en español.
Reformula el mensaje solicitado en una sola frase breve y completa, sin listas.
Conserva su significado. No añadas explicaciones, diagnósticos ni información nueva.
Devuelve solo el mensaje reformulado.'''

@lru_cache(maxsize=1)
def cargar_llm():
    # Se carga al pulsar el botón, no al abrir la app. Caché compartida entre sesiones.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, revision=LLM_REVISION)
    red = AutoModelForCausalLM.from_pretrained(
        LLM_ID, revision=LLM_REVISION, torch_dtype=torch.bfloat16,
        use_safetensors=True, trust_remote_code=False).to('cpu').eval()
    return tokenizer, red

def generar_recomendacion(resultado):
    # Solo se pasan campos derivados del modelo; no se envían imagen ni nombre de archivo.
    from modelo import NOMBRES
    accion = ('La confianza es baja: solicita revisión humana antes de interpretar el resultado.'
              if resultado['baja_confianza'] else
              'Aunque el puntaje sea alto, solicita revisión humana del resultado.')
    mensaje = (f"Contexto de una demostración académica: clase predicha {NOMBRES[resultado['clase']]}; "
               f"Confianza softmax: {resultado['confianza']:.1%}. "
               'No repitas esos datos en tu respuesta. '
               f'Mensaje que debes reformular en una frase: {accion} '
               'Consulta a un profesional y recuerda que esta aplicación tiene uso exclusivamente educativo.')
    # Serializar carga y generación evita duplicar memoria con varios usuarios.
    with GENERACION_LOCK:
        tokenizer, red = cargar_llm()
        texto = tokenizer.apply_chat_template([
            {'role':'system','content':SISTEMA}, {'role':'user','content':mensaje}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(texto, return_tensors='pt')
        with torch.inference_mode():
            salida = red.generate(**inputs, max_new_tokens=110, do_sample=False,
                                  repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
        respuesta = tokenizer.decode(salida[0, inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()
    if not respuesta:
        raise RuntimeError('El modelo de texto no devolvió una respuesta.')
    # Control conservador: un modelo pequeño puede desobedecer instrucciones.
    # Si interpreta el puntaje como riesgo clínico, no mostramos ese texto.
    if re.search(r'(probabilidad|posibilidad|riesgo|segur[oa]).{0,100}(tumor|enfermedad|c[aá]ncer|salud)|'
                 r'(tienes?|presentas?|padeces?|sufres?).{0,25}(tumor|c[aá]ncer)|'
                 r'(tratamiento|medicamento|cirug[ií]a|quimioterapia)', respuesta.lower()):
        raise ValueError('El texto generado interpretó el resultado clínicamente y se descartó.')
    return respuesta
