"""Punto 4.3. Ejecutar con: streamlit run app.py"""
from pathlib import Path
import hashlib
import json
import pandas as pd
import altair as alt
import streamlit as st
import torch
from modelo import (cargar_modelo, clasificar, leer_imagen, sha256_archivo,
                    NOMBRES, UMBRAL_CONFIANZA)
from recomendacion import generar_recomendacion, LLM_ID, ErrorRecomendacion

BASE = Path(__file__).resolve().parent
# Al cambiar el prompt, no conservar en pantalla una orientación de la versión anterior.
if st.session_state.get('version_orientacion') != 'aprendizaje-2':
    st.session_state.pop('recomendacion', None)
    st.session_state['version_orientacion'] = 'aprendizaje-2'
torch.set_num_threads(2)  # CPU: evita que varias sesiones saturen el servidor.
st.set_page_config(page_title='Neuro JVC · Resonancias cerebrales', page_icon='🧠', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp {background: #f5f9fc;color:#263746;}
.block-container {max-width:1180px;padding-top:1.3rem;}
h1,h2,h3,h4 {color:#164e63;}
[data-testid="stMetric"] {background:#ffffff;padding:18px;border-radius:14px;border:1px solid #dce7ed;}
.encabezado {padding:16px 22px;background:linear-gradient(110deg,#e7f3f6,#f2faf8);border:1px solid #d2e8e9;border-left:5px solid #2a9d8f;border-radius:16px;color:#263746;margin-bottom:12px;}
.encabezado h1 {color:#164e63;font-size:2rem;margin:0;}
.encabezado p {color:#4d6573;margin:8px 0 0;}
.etiqueta {font-size:11px;letter-spacing:1.4px;font-weight:700;color:#18746b;margin-bottom:3px;}
[data-testid="stSidebar"] {background:#edf4f7;border-right:1px solid #dce7ed;}
.encabezado {display:flex;justify-content:space-between;align-items:center;gap:24px;}
.encabezado h1 {font-size:2rem;letter-spacing:-0.6px;}
.marca-cerebro {width:68px;min-width:58px;color:#25877e;}
.equipo {font-size:0.78rem;color:#526b79;margin-top:5px;}
[data-testid="stMetricValue"] {white-space:normal;font-size:1.4rem;}
[data-testid="stMetric"] {padding:13px;}
.pasos {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:10px 0 24px;}
.paso {padding:16px 18px;border:1px solid #dce7ed;border-radius:14px;background:#fff;color:#4d6573;font-size:0.9rem;}
.paso strong {display:block;color:#164e63;margin-bottom:4px;}
.numero {display:inline-block;color:#18746b;font-weight:700;margin-right:5px;}
button[kind="primary"]:not(:disabled) {color:#ffffff!important;font-weight:700;}
[data-testid="stFileUploaderDropzone"] {border:1px dashed #8ab9bd;border-radius:14px;background:#fff;}
[data-testid="stFileUploaderDropzoneInstructions"] {display:none;}
[data-testid="stFileUploaderDropzone"] button {font-size:0!important;}
[data-testid="stFileUploaderDropzone"] button::after {content:"Seleccionar imagen";font-size:14px;}
@media(max-width:640px){.pasos {grid-template-columns:1fr;gap:8px;}.marca-cerebro {display:none;}.encabezado h1 {font-size:2.2rem;}.encabezado {padding:22px;}}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def recursos_modelo():
    return cargar_modelo(BASE / 'brain_tumor_efficientnet_b0.pt')

@st.cache_data
def metricas():
    return json.loads((BASE / 'metricas_test.json').read_text(encoding='utf-8'))

st.markdown('''<div class="encabezado"><div><div class="etiqueta">RESONANCIAS CEREBRALES · IA EDUCATIVA</div>
<h1>Neuro JVC</h1>
<p>Un espacio para aprender cómo clasifica la IA las resonancias cerebrales.</p>
<div class="equipo">Desarrollado por Javiera, Valentina y Cristian</div></div>
<svg class="marca-cerebro" viewBox="0 0 100 100" role="img" aria-label="Ícono de cerebro de Neuro JVC" fill="none" stroke="currentColor" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round">
<rect x="3" y="3" width="94" height="94" rx="26" fill="#e4f3ef" stroke="none"/>
<path d="M49 24c-8-12-25-5-25 7-12 3-15 19-7 27-7 14 5 27 17 24 5 8 15 4 15-5V24zm2 0c8-12 25-5 25 7 12 3 15 19 7 27 7 14-5 27-17 24-5 8-15 4-15-5V24z"/>
<path d="M24 31c10-3 14 5 12 12M17 58c6-8 14-6 18-1m-1 25c-4-7-2-14 5-17M76 31c-10-3-14 5-12 12m19 15c-6-8-14-6-18-1m1 25c4-7 2-14-5-17M36 43l6 5m22-5-6 5"/>
</svg></div>''', unsafe_allow_html=True)
st.caption('Para estudiantes y docentes · 1. Elige un ejemplo → 2. Compara la predicción → 3. Explícala con Gemini')
st.warning('Proyecto educativo: la IA puede equivocarse. Sus respuestas no son un diagnóstico.')

with st.sidebar:
    st.header('🧠 Neuro JVC')
    st.caption('Javiera · Valentina · Cristian')
    st.markdown('**Clasificación de resonancias cerebrales con inteligencia artificial.**')
    st.divider()
    st.markdown('**Dos herramientas, funciones distintas**')
    st.write('**Modelo:** EfficientNet-B0 con fine-tuning')
    st.caption('Clasifica la imagen en una de cuatro categorías.')
    st.write('**Orientación:** Gemini')
    st.caption('Explica el resultado por texto. No examina la imagen.')
    st.write('**Categorías:** glioma, meningioma, sin tumor y tumor hipofisario.')
    st.divider()
    st.caption('Para probarla, usa imágenes de la base del proyecto. Evita subir nombres u otros datos que identifiquen a una persona.')
    st.caption('La aplicación no añade tus imágenes a entrenamiento. El archivo se procesa en el servidor durante la sesión.')

analizar, desempeno, explicar = st.tabs(['Analizar imagen', 'Qué tan bien funciona', 'Sobre el proyecto'])

with analizar:
    izquierda, derecha = st.columns([1,1.15], gap='large')
    with izquierda:
        st.subheader('1. Elige una resonancia')
        modo = st.radio('¿Cómo quieres probar Neuro JVC?', ['Usar un ejemplo', 'Subir imagen'], horizontal=True, key='modo_neuro_v2')
        ejemplo = None
        if modo == 'Subir imagen':
            st.caption('Arrastra tu imagen al recuadro o pulsa «Seleccionar imagen».')
            archivo = st.file_uploader('Imagen de resonancia cerebral', type=['jpg','jpeg','png'])
            st.caption('JPG o PNG · máximo 10 MB. Una sola imagen, no un estudio médico completo.')
            datos = archivo.getvalue() if archivo else None
        else:
            ejemplos = json.loads((BASE / 'ejemplos.json').read_text(encoding='utf-8'))
            elegido = st.selectbox('Selecciona un ejemplo del proyecto',
                [e['nombre'] + ' · etiqueta del dataset' for e in ejemplos])
            ejemplo = next(e for e in ejemplos if e['nombre'] + ' · etiqueta del dataset' == elegido)
            datos = (BASE / ejemplo['archivo']).read_bytes()
            st.caption('Imagen de prueba con una etiqueta conocida. Úsala para comparar lo que responde la IA.')
        identificador = hashlib.sha256(datos).hexdigest() if datos else None
        # Invalidar todo al cambiar/quitar la imagen impide mostrar un resultado anterior.
        if st.session_state.get('imagen_id') != identificador:
            st.session_state['imagen_id'] = identificador
            st.session_state.pop('resultado', None)
            st.session_state.pop('recomendacion', None)
        imagen = None
        if datos:
            try:
                imagen = leer_imagen(datos)
                # Miniatura solo para la vista: el modelo sigue recibiendo la imagen original.
                vista = imagen.copy()
                vista.thumbnail((420, 280))
                st.image(vista, caption='Imagen seleccionada', width=vista.width)
                if ejemplo:
                    with st.expander('De dónde viene este ejemplo'):
                        st.caption(f"Etiqueta: {ejemplo['nombre']} · Archivo: {ejemplo['origen']}")
            except ValueError as exc:
                st.error(str(exc))
        if st.button('Clasificar imagen', type='primary', disabled=imagen is None, width='stretch'):
            st.session_state.pop('resultado', None)
            st.session_state.pop('recomendacion', None)
            try:
                with st.spinner('Analizando con el modelo entrenado…'):
                    st.session_state['resultado'] = clasificar(imagen, recursos_modelo())
            except Exception:
                st.error('No se pudo cargar o ejecutar el modelo. Revisa que el archivo .pt esté en el repositorio junto a app.py.')

    with derecha:
        st.subheader('2. Mira el resultado')
        resultado = st.session_state.get('resultado')
        if resultado is None:
            st.info('Pulsa «Clasificar imagen» para comparar la respuesta de la IA con la etiqueta del ejemplo. También puedes subir tu propia imagen.')
            if ejemplo:
                st.metric('Etiqueta del dataset', ejemplo['nombre'], help='Es la categoría que trae la base de datos. Se usa como referencia del ejercicio; no es una nueva evaluación médica.')
        else:
            if ejemplo:
                referencia, prediccion = st.columns(2)
                referencia.metric('Etiqueta del dataset', ejemplo['nombre'], help='Etiqueta original de esta imagen en la base de datos, independiente de la predicción.')
                prediccion.metric('Predicción de la IA', NOMBRES[resultado['clase']], help='Categoría que acaba de elegir el modelo a partir de los píxeles.')
            else:
                st.metric('Predicción de la IA', NOMBRES[resultado['clase']])
            st.metric('Confianza del modelo', f"{resultado['confianza']:.1%}", help='Cuánto favorece el modelo su opción elegida frente a las otras tres. No mide la probabilidad de enfermedad ni garantiza un acierto.')
            st.caption('Este porcentaje indica cuánto favorece el modelo esta categoría frente a las otras tres. No indica la probabilidad de tener un tumor ni garantiza que la respuesta sea correcta.')
            if ejemplo:
                if resultado['clase'] == ejemplo['clase']:
                    st.success('Coinciden en este ejemplo. Esto no demuestra que el modelo acierte con todas las imágenes.')
                else:
                    st.warning('No coinciden: es un error respecto de la etiqueta del dataset. Observa que una confianza alta no garantiza un acierto.')
            if resultado['baja_confianza']:
                st.warning('Confianza baja: menor al 80 %. Revisa este caso con tu docente antes de sacar conclusiones. Si fuera un caso real, necesita revisión profesional. El umbral es orientativo, no médico.')
            if resultado['clase'] == 'notumor':
                st.info('«Sin tumor» es la categoría elegida por la IA. Por sí sola, esta respuesta no permite saber si una persona está sana.')
            st.markdown('**Cómo se repartió la confianza entre las categorías**')
            tabla = pd.DataFrame([{'Categoría':NOMBRES[c], 'Confianza':p} for c,p in resultado['probabilidades'].items()])
            barras = alt.Chart(tabla).mark_bar(cornerRadiusEnd=5, color='#2a9d8f').encode(
                x=alt.X('Confianza:Q', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format='%'), title='Confianza del modelo (%)'),
                y=alt.Y('Categoría:N',sort=None,title=None),
                tooltip=['Categoría',alt.Tooltip('Confianza:Q', format='.2%')]).properties(height=200)
            st.altair_chart(barras, use_container_width=True)
            with st.expander('Ver porcentajes de las cuatro categorías'):
                st.dataframe(tabla.assign(Confianza=tabla.Confianza.map(lambda p:f'{p:.2%}')),
                             hide_index=True, width='stretch')

    st.divider()
    st.subheader('3. Entiende esta predicción')
    st.write('Pide a Gemini una explicación para el ejercicio: qué eligió la IA, cómo interpretar su confianza y qué revisar con tu docente.')
    st.caption('Gemini recibe solo la categoría, el porcentaje de confianza y el contexto del proyecto. No recibe tu imagen ni su nombre.')
    if st.button('Generar recomendación con IA', disabled=st.session_state.get('resultado') is None):
        st.session_state.pop('recomendacion', None)
        try:
            with st.spinner('Preparando una explicación para ti…'):
                contexto_educativo = dict(st.session_state['resultado'])
                if ejemplo:
                    contexto_educativo['etiqueta_ejemplo'] = ejemplo['clase']
                st.session_state['recomendacion'] = generar_recomendacion(contexto_educativo)
        except ErrorRecomendacion as exc:
            st.error(str(exc))
        except ValueError:
            st.warning('La respuesta del modelo de texto no superó la revisión automática y se descartó. No interpretes la confianza como riesgo de enfermedad. Solicita revisión humana.')
        except Exception:
            st.error('No se pudo generar la orientación. Reintenta más tarde; no se ha sustituido la respuesta por un texto simulado.')
    if st.session_state.get('recomendacion'):
        st.caption(f'Orientación generada por {LLM_ID}. También puede equivocarse.')
        # Los encabezados son nuestros; el texto generado se muestra sin HTML ni enlaces.
        titulos = {'En palabras simples', 'Qué puedes hacer ahora', 'Recuerda'}
        for bloque in st.session_state['recomendacion'].split('\n\n'):
            titulo, salto, contenido = bloque.partition('\n')
            if salto and titulo in titulos:
                with st.container(border=True):
                    st.markdown(f'#### {titulo}')
                    st.text(contenido)
            else:
                st.text(bloque)

with desempeno:
    st.subheader('Cómo funcionó el modelo con imágenes de prueba')
    m = metricas()
    st.write('Probamos el modelo con **1.600 imágenes** que no se usaron para ajustar sus pesos. Estos resultados resumen esa prueba; no dicen si la respuesta de tu imagen es correcta.')
    a,b,c = st.columns(3)
    a.metric('Aciertos en la prueba (accuracy)', f"{m['efficientnet']['accuracy']:.2%}", help='Porcentaje de imágenes de prueba cuya predicción coincidió con su etiqueta. No es la confianza de una imagen individual.')
    b.metric('F1 macro en test', f"{m['efficientnet']['f1_macro']:.2%}", help='Combina precisión y recall y promedia las cuatro categorías con igual peso. Un valor cercano a 100 % es mejor en esta prueba.')
    c.metric('Imágenes de prueba', '1.600')
    st.caption('El modelo acertó en 1.490 de las 1.600 imágenes de esa prueba. F1 macro resume precisión y detección dando el mismo peso a las cuatro categorías.')
    st.caption('Fuente: salidas guardadas de Certamen1_IA.ipynb. F1 y métricas por clase se transcriben con el redondeo mostrado en el notebook.')
    st.dataframe(pd.DataFrame(m['por_clase']).rename(columns={
        'clase':'Clase','precision':'Precisión','recall':'Recall','f1':'F1','n':'Imágenes'}),
        hide_index=True, width='stretch')
    st.image(str(BASE / 'matriz_confusion.png'), caption='Matriz de confusión de EfficientNet-B0 exportada del notebook: filas reales, columnas predichas.', width='stretch')
    st.markdown('**Comparación con la línea base**')
    st.dataframe(pd.DataFrame([
        {'Modelo':'CLIP zero-shot','Accuracy':m['clip']['accuracy'],'F1 macro':m['clip']['f1_macro']},
        {'Modelo':'EfficientNet-B0','Accuracy':m['efficientnet']['accuracy'],'F1 macro':m['efficientnet']['f1_macro']}]), hide_index=True, width='stretch')
    st.warning('Error especialmente costoso: 36 imágenes etiquetadas con tumor fueron clasificadas como «sin tumor» (36 de 1.200 imágenes tumorales; 3 %). El recall binario de tumor fue 97 %. Confundir tipos de tumor también importa.')
    st.caption('División: 4.760 entrenamiento / 840 validación / 1.600 test. Sin identificadores de paciente no se garantiza independencia entre cortes; las métricas pueden ser optimistas y no implican validación clínica.')
    with st.expander('Trazabilidad del modelo'):
        actual = sha256_archivo(BASE / 'brain_tumor_efficientnet_b0.pt')
        st.code(actual, language=None)
        if actual != m['modelo_sha256']:
            st.error('El archivo del modelo cambió. Estas métricas podrían no corresponder: actualiza la evaluación antes de presentar.')
        else:
            st.caption('Huella del archivo recibido junto al notebook, conservada para detectar reemplazos. No constituye una reevaluación independiente.')

with explicar:
    st.subheader('Neuro JVC: aprender a mirar una predicción con criterio')
    st.write('Somos Javiera, Valentina y Cristian. Desarrollamos este proyecto para explorar cómo la inteligencia artificial clasifica resonancias cerebrales y cómo explicar sus resultados de forma comprensible.')
    st.markdown('**Qué puedes hacer aquí:** probar una imagen, comparar las cuatro categorías, revisar los resultados de evaluación y pedir una explicación a Gemini.')
    st.markdown('**Qué no hace la app:** no localiza un tumor, no mide su tamaño, no establece gravedad y no determina un diagnóstico.')
    st.markdown('#### Las categorías del proyecto')
    st.dataframe(pd.DataFrame([{'Nombre en la app': nombre, 'Etiqueta original': clase}
        for clase, nombre in NOMBRES.items()]), hide_index=True, width='stretch')
    st.divider()
    st.subheader('Del entrenamiento a la aplicación')
    st.markdown('''
    1. **Preparación:** Training original se dividió 85/15 de forma estratificada, con semilla 42. Testing permaneció intacto.
    2. **Fine-tuning:** EfficientNet-B0 partió de pesos de ImageNet y ajustó todos sus parámetros para cuatro clases. La mejor época se eligió por F1 macro de validación.
    3. **Predicción:** usamos RGB, redimensionamiento, recorte central y normalización guardados en el `.pt`. El recorte puede omitir información en los bordes.
    4. **Orientación:** Gemini recibe por API clase, confianza y contexto educativo. Sugiere revisión humana, explica límites y no valida la predicción. La imagen no se envía a Google y la clave se guarda en Secrets.
    ''')
    st.markdown('**Límites:** solo cuatro categorías, imágenes 2D similares a las de entrenamiento, sin localización ni segmentación del tumor y sin validación clínica externa. La app no detecta automáticamente si subiste una fotografía u otra imagen ajena al problema: también podría asignarle una clase.')
    st.markdown('**Para el grupo:** `app.py` organiza la interfaz; `modelo.py` carga y ejecuta EfficientNet; `recomendacion.py` consulta el LLM; `metricas_test.json` conserva los resultados del notebook.')
    st.link_button('Ver código del proyecto', 'https://github.com/Crisgutierrezc/proyecto-tumores-cerebrales')

st.divider()
st.caption('Neuro JVC · Javiera, Valentina y Cristian · Proyecto académico con Brain Tumor MRI Dataset')

