"""4.5: orientacion con Gemini API; clave exclusivamente en Secrets."""
import json
import os
import re
import requests
import streamlit as st

LLM_ID = 'Gemini (API de Google)'
MODELO_DEFAULT = 'gemini-3.5-flash-lite'

class ErrorRecomendacion(RuntimeError):
    """Mensaje seguro para el usuario: sin claves ni cuerpos HTTP."""

SISTEMA = '''Eres un orientador educativo de un proyecto universitario de clasificacion
 de resonancias cerebrales. Tu foco es orientar la revision humana de la prediccion,
 no evaluar la salud de una persona. Solo conoces clase y confianza; no viste la imagen.
 Responde en espanol claro, breve, respetuoso, con tildes correctas.
 Reglas obligatorias:
 - Describe la clase como prediccion del clasificador, nunca como hallazgo confirmado.
 - La confianza softmax no es probabilidad de enfermedad, exactitud individual ni gravedad.
 - Si confianza_baja=true, destaca incertidumbre y pide revision humana antes de interpretar.
 - Si confianza_baja=false, explica que una confianza alta tambien puede ser incorrecta.
 - Si clase=notumor, aclara que no descarta enfermedad y evita falsa tranquilidad.
 - Para clase tumoral, recomienda revision por un profesional de salud para evaluar
   la imagen y confirmar o descartar la prediccion; no asumas que la persona tiene tumor.
 - No determines urgencia, pronostico o gravedad por la categoria o el puntaje.
 - No prescribas tratamientos, examenes, medicamentos ni consejos personalizados.
 - No inventes sintomas, rasgos de imagen, fuentes, porcentajes o datos del paciente.
 - Nunca digas evaluacion favorable, estas sano, resultado benigno ni garantices seguridad.
 - Aclara: demostracion educativa sin validacion clinica, no es un diagnostico.
 Devuelve JSON con tres textos: significado (1-2 frases sobre alcance del resultado),
 revision (accion de revision humana acorde a clase y confianza) y limites (uso educativo
 y ausencia de diagnostico). Maximo 150 palabras en total. Usa solo el contexto dado;
 no agregues explicaciones medicas de los tipos de tumor.'''

def contexto_resultado(resultado):
    from modelo import NOMBRES, UMBRAL_CONFIANZA
    clase = resultado.get('clase')
    confianza = float(resultado.get('confianza', -1))
    if clase not in NOMBRES or not 0 <= confianza <= 1:
        raise ErrorRecomendacion('Resultado no válido. Clasifica de nuevo la imagen.')
    # Lista blanca: no envia imagen, nombre, ruta, metadatos ni datos personales.
    return {'clase':clase, 'nombre_clase':NOMBRES[clase],
        'confianza_softmax_porcentaje':round(confianza*100,2),
        'confianza_baja':confianza < UMBRAL_CONFIANZA,
        'umbral_educativo_porcentaje':UMBRAL_CONFIANZA*100,
        'contexto':'Proyecto académico, sin validación clínica; umbral orientativo no clínico.'}

def secreto(nombre, default=''):
    valor = os.environ.get(nombre)
    if valor:
        return valor
    try:
        return st.secrets.get(nombre, default)
    except FileNotFoundError:
        return default

def generar_recomendacion(resultado):
    contexto = contexto_resultado(resultado)
    key = str(secreto('GEMINI_API_KEY')).strip()
    if not key:
        raise ErrorRecomendacion('Falta GEMINI_API_KEY en Secrets de Streamlit. No la subas a GitHub.')
    model = str(secreto('GEMINI_MODEL', MODELO_DEFAULT)).strip()
    if not re.fullmatch(r'gemini-[a-zA-Z0-9.\-]+', model):
        raise ErrorRecomendacion('GEMINI_MODEL no tiene un nombre válido.')
    body = {'systemInstruction':{'parts':[{'text':SISTEMA}]},
        'contents':[{'role':'user','parts':[{'text':json.dumps(contexto,ensure_ascii=False)}]}],
        'generationConfig':{'temperature':0.2,'maxOutputTokens':2048,
            'responseMimeType':'application/json','responseSchema':{
                'type':'OBJECT','properties':{c:{'type':'STRING'} for c in
                    ['significado','revision','limites']},
                'required':['significado','revision','limites']}}}
    try:
        response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
            headers={'x-goog-api-key':key,'Content-Type':'application/json'},
            json=body, timeout=(10,55))
    except requests.RequestException:
        raise ErrorRecomendacion('No se pudo conectar con Gemini. Reintenta más tarde.') from None
    # Nunca mostrar response.text ni la clave en la interfaz o en logs.
    errores = {
        400:'Google rechazó la solicitud. Revisa la validez de la clave y la configuración del modelo.',
        401:'Google no aceptó la autenticación. Revisa GEMINI_API_KEY en Secrets.',
        403:'Google no autorizó la solicitud. Revisa permisos de la clave y acceso de tu cuenta a Gemini.',
        404:'El modelo no está disponible para esta cuenta. Revisa GEMINI_MODEL en Secrets.',
        429:'Gemini alcanzó una cuota o límite de solicitudes. Espera y revisa la cuota en AI Studio. La app no activará pagos.'}
    if response.status_code != 200:
        raise ErrorRecomendacion(errores.get(response.status_code,
            'Gemini no pudo responder en este momento. Reintenta más tarde.'))
    try:
        candidate = response.json()['candidates'][0]
        if candidate.get('finishReason') != 'STOP':
            raise ValueError('Respuesta incompleta o bloqueada')
        texto = ''.join(p.get('text','') for p in candidate['content']['parts'] if not p.get('thought'))
        respuesta = json.loads(texto)
        campos = ['significado','revision','limites']
        if any(not isinstance(respuesta.get(c),str) or not respuesta[c].strip() for c in campos):
            raise ValueError('Faltan secciones')
        conjunto = ' '.join(respuesta[c] for c in campos)
        if len(conjunto)>4000 or re.search(
            r'evaluaci[oó]n favorable|est[aá]s sano|resultado benigno|tienes un tumor',conjunto.lower()):
            raise ValueError('Contenido fuera del alcance')
    except (KeyError,IndexError,TypeError,ValueError):
        raise ErrorRecomendacion('La respuesta quedó incompleta o no pasó la revisión automática. No se muestra; puedes reintentar.') from None
    return ('Qué significa el resultado\n'+respuesta['significado'].strip()+
        '\n\nRevisión recomendada\n'+respuesta['revision'].strip()+
        '\n\nAlcance y límites\n'+respuesta['limites'].strip())
