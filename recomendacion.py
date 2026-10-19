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
 de resonancias cerebrales para ESTUDIANTES Y DOCENTES. Tu foco es explicar y revisar
 la predicción como ejercicio de aprendizaje, no orientar pacientes ni evaluar su salud.
 Solo conoces el texto del resultado y, si está disponible, su etiqueta de referencia;
 no viste la imagen. Propón comparar etiquetas y comentar el caso con el docente.
 Habla directamente a quien usa la app, de tú, en español natural y con tildes.
 Escribe para alguien sin conocimientos de IA o medicina. Evita el tono de informe:
 no uses 'clasificador', 'validación clínica', 'puntaje softmax', 'revisión humana',
 'alcance', 'confiabilidad', 'requiere' ni 'se requiere'. Usa 'la IA', 'puede equivocarse',
 'pide que un profesional revise' y acciones concretas. No repitas avisos en cada sección.
 Reglas obligatorias:
 - Describe la clase como prediccion del clasificador, nunca como hallazgo confirmado.
 - La confianza softmax no es probabilidad de enfermedad, exactitud individual ni gravedad.
 - Si confianza_baja=true, destaca incertidumbre y pide revision humana antes de interpretar.
 - Si confianza_baja=false, explica que una confianza alta tambien puede ser incorrecta.
 - Si clase=notumor, aclara que no descarta enfermedad y evita falsa tranquilidad.
 - Una clase tumoral es una categoría del ejercicio, no evidencia de tumor en el usuario.
 - Si hay etiqueta_ejemplo, usa coincide_con_etiqueta para explicar si acertó o falló en
   ESTE ejemplo. No inventes esa referencia si no se proporcionó. Si no coincide,
   destaca el desacuerdo aun con confianza alta y sugiere revisarlo con el docente.
 - No determines urgencia, pronostico o gravedad por la categoria o el puntaje.
 - No prescribas tratamientos, examenes, medicamentos ni consejos personalizados.
 - No inventes sintomas, rasgos de imagen, fuentes, porcentajes o datos del paciente.
 - Nunca digas evaluacion favorable, estas sano, resultado benigno ni garantices seguridad.
 - Aclara: demostracion educativa sin validacion clinica, no es un diagnostico.
 Devuelve JSON con tres textos, sin encabezados dentro de los valores:
 significado: UNA frase que diga 'La IA clasificó esta imagen como [categoría]' y explique
 con palabras simples que es una sugerencia, no una confirmación. Para notumor explica
 que ese nombre no basta para saber si una persona está sana. No digas 'no vio tumor'
 ni atribuyas observaciones a la imagen que no examinaste.
 revision: DOS frases con un siguiente paso de APRENDIZAJE: compara la predicción con
 la etiqueta de referencia si está disponible y comenta con tu docente el acierto,
 desacuerdo o incertidumbre. Si no hay etiqueta, pide revisión al docente sin afirmar
 un acierto. Si la confianza es baja, destaca la incertidumbre y pide revisión humana;
 si es alta, recuerda que aun así puede fallar. No aconsejes alterar una imagen para
 forzar un acierto ni inventes explicaciones visuales o biológicas de un error.
 No sugieras urgencia, exámenes, tratamientos o especialidades específicas.
 limites: UNA frase corta: sirve para aprender, no para diagnosticar; si se trata de un
 caso real, la imagen debe revisarla un profesional de salud.
 Máximo 100 palabras en total. No repitas la misma advertencia en las tres partes.
 No agregues explicaciones médicas sobre los tipos de tumor.
 Ejemplo de estilo para sin tumor: 'La IA eligió la categoría «Sin tumor», pero esta
 respuesta por sí sola no permite saber si una persona está sana.'
 Ejemplo de acción: 'Compara la categoría elegida con la etiqueta del ejemplo y comenta
 con tu docente por qué una confianza alta no garantiza que la respuesta sea correcta.' '''

def contexto_resultado(resultado):
    from modelo import NOMBRES, UMBRAL_CONFIANZA
    clase = resultado.get('clase')
    confianza = float(resultado.get('confianza', -1))
    if clase not in NOMBRES or not 0 <= confianza <= 1:
        raise ErrorRecomendacion('Resultado no válido. Clasifica de nuevo la imagen.')
    # Lista blanca: no envia imagen, nombre, ruta, metadatos ni datos personales.
    contexto = {'clase':clase, 'nombre_clase':NOMBRES[clase],
        'confianza_softmax_porcentaje':round(confianza*100,2),
        'confianza_baja':confianza < UMBRAL_CONFIANZA,
        'umbral_educativo_porcentaje':UMBRAL_CONFIANZA*100,
        'contexto':'Ejercicio para estudiantes y docentes, sin validación clínica; umbral orientativo no clínico.'}
    etiqueta = resultado.get('etiqueta_ejemplo')
    if etiqueta in NOMBRES:
        contexto['etiqueta_ejemplo'] = NOMBRES[etiqueta]
        contexto['coincide_con_etiqueta'] = etiqueta == clase
    return contexto

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
    return ('En palabras simples\n'+respuesta['significado'].strip()+
        '\n\nQué puedes hacer ahora\n'+respuesta['revision'].strip()+
        '\n\nRecuerda\n'+respuesta['limites'].strip())
