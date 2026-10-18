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
from recomendacion import generar_recomendacion, LLM_ID

BASE = Path(__file__).resolve().parent
torch.set_num_threads(2)  # CPU: evita que varias sesiones saturen el servidor.
st.set_page_config(page_title='Clasificación de tumores cerebrales', page_icon='🧠', layout='wide')
st.markdown('''<style>
.stApp {background: #f4f7fb;}
.block-container {max-width:1180px;padding-top:2.3rem;}
h1,h2,h3 {color:#18354c;}
[data-testid="stMetric"] {background:white;padding:18px;border-radius:14px;border:1px solid #dce5ee;}
.encabezado {padding:25px 30px;background:linear-gradient(110deg,#18354c,#236a79);border-radius:20px;color:white;margin-bottom:20px;}
.encabezado h1 {color:white;font-size:2rem;margin:0;}
.encabezado p {color:#e0edf5;margin:8px 0 0;}
.etiqueta {font-size:12px;letter-spacing:2px;font-weight:700;color:#b8e5df;margin-bottom:10px;}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def recursos_modelo():
    return cargar_modelo(BASE / 'brain_tumor_efficientnet_b0.pt')

@st.cache_data
def metricas():
    return json.loads((BASE / 'metricas_test.json').read_text(encoding='utf-8'))

st.markdown('''<div class="encabezado"><div class="etiqueta">PROYECTO DE INTELIGENCIA ARTIFICIAL · 4.3</div>
<h1>Clasificación de tumores cerebrales</h1>
<p>Explora cómo un modelo entrenado clasifica imágenes de resonancia magnética cerebral.</p></div>''', unsafe_allow_html=True)
st.warning('Uso exclusivamente educativo. Esta app no confirma ni descarta un diagnóstico y no reemplaza la evaluación de un profesional de salud.')

with st.sidebar:
    st.header('Acerca del proyecto')
    st.write('**Modelo:** EfficientNet-B0 con fine-tuning')
    st.write('**Clases:** glioma, meningioma, sin tumor y tumor hipofisario.')
    st.divider()
    st.caption('Usa imágenes del dataset para la demostración. No subas información que identifique a pacientes.')
    st.caption('La aplicación no añade tus imágenes a entrenamiento. El archivo se procesa en el servidor durante la sesión.')

analizar, desempeno, explicar = st.tabs(['Analizar imagen', 'Desempeño en prueba', 'Cómo funciona'])

with analizar:
    izquierda, derecha = st.columns([1,1.15], gap='large')
    with izquierda:
        st.subheader('1. Sube una resonancia')
        archivo = st.file_uploader('Arrastra tu imagen o selecciona un archivo', type=['jpg','jpeg','png'])
        st.caption('JPG o PNG · máximo 10 MB · imagen 2D de resonancia cerebral. No admite DICOM ni volúmenes 3D.')
        datos = archivo.getvalue() if archivo else None
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
                st.image(imagen, caption='Imagen seleccionada', width='stretch')
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
        st.subheader('2. Revisa la clasificación')
        resultado = st.session_state.get('resultado')
        if resultado is None:
            st.info('Sube una imagen y pulsa «Clasificar imagen» para ver el resultado.')
        else:
            st.metric('Clase predicha', NOMBRES[resultado['clase']])
            st.metric('Confianza en esta imagen', f"{resultado['confianza']:.1%}")
            st.caption('Confianza = puntaje softmax de la clase ganadora. No es accuracy ni probabilidad clínica de tener un tumor.')
            if resultado['baja_confianza']:
                st.warning('Confianza baja: inferior al 80 %. Requiere revisión humana. Este umbral es orientativo y no está validado clínicamente.')
            else:
                st.info('Una confianza alta también puede acompañar una predicción incorrecta. Mantén la revisión humana.')
            if resultado['clase'] == 'notumor':
                st.warning('La etiqueta «Sin tumor» es una predicción del modelo; no permite descartar una enfermedad.')
            tabla = pd.DataFrame([{'Clase':NOMBRES[c], 'Probabilidad':p} for c,p in resultado['probabilidades'].items()])
            barras = alt.Chart(tabla).mark_bar(cornerRadiusEnd=5, color='#238b8e').encode(
                x=alt.X('Probabilidad:Q', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format='%'), title='Puntaje softmax'),
                y=alt.Y('Clase:N',sort=None,title=None),
                tooltip=['Clase',alt.Tooltip('Probabilidad:Q', format='.2%')]).properties(height=200)
            st.altair_chart(barras, use_container_width=True)
            st.dataframe(tabla.assign(Probabilidad=tabla.Probabilidad.map(lambda p:f'{p:.2%}')),
                         hide_index=True, width='stretch')

    st.divider()
    st.subheader('3. Genera una orientación con IA')
    st.write('Un modelo de lenguaje recibe la clase predicha y su confianza para redactar una orientación educativa. No recibe la imagen ni examina al paciente.')
    st.caption('Sin clave de API. La primera consulta descarga el modelo abierto (aprox. 1 GB) y puede tardar; las siguientes reutilizan el modelo en memoria.')
    if st.button('Generar recomendación con IA', disabled=st.session_state.get('resultado') is None):
        st.session_state.pop('recomendacion', None)
        try:
            with st.spinner('Preparando el modelo de texto y generando orientación…'):
                st.session_state['recomendacion'] = generar_recomendacion(st.session_state['resultado'])
        except ValueError:
            st.warning('La respuesta del modelo de texto no superó la revisión automática y se descartó. No interpretes la confianza como riesgo de enfermedad. Solicita revisión humana.')
        except Exception:
            st.error('No se pudo generar la recomendación. Puede faltar conexión o memoria del servidor. Puedes reintentar; no se ha sustituido la respuesta por un texto simulado.')
    if st.session_state.get('recomendacion'):
        st.caption(f'Texto generado por {LLM_ID}. Puede contener errores; debe revisarse.')
        # st.text evita que un LLM inserte enlaces o contenido HTML en la interfaz.
        st.text(st.session_state['recomendacion'])
        st.info('Orientación responsable de la app: ningún resultado confirma ni descarta enfermedad. Solicita revisión profesional y no tomes decisiones médicas con esta demostración.')

with desempeno:
    st.subheader('Resultados del conjunto de prueba')
    m = metricas()
    st.write('Estas métricas se obtuvieron en **1.600 imágenes de Testing**, reservadas para evaluación. Describen el rendimiento del conjunto; no la confianza de la imagen que subas.')
    a,b,c = st.columns(3)
    a.metric('Accuracy en test', f"{m['efficientnet']['accuracy']:.2%}")
    b.metric('F1 macro en test', f"{m['efficientnet']['f1_macro']:.2%}")
    c.metric('Imágenes de prueba', '1.600')
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
    st.subheader('Del entrenamiento a la aplicación')
    st.markdown('''
    1. **Preparación:** Training original se dividió 85/15 de forma estratificada, con semilla 42. Testing permaneció intacto.
    2. **Fine-tuning:** EfficientNet-B0 partió de pesos de ImageNet y ajustó todos sus parámetros para cuatro clases. La mejor época se eligió por F1 macro de validación.
    3. **Predicción:** usamos RGB, redimensionamiento, recorte central y normalización guardados en el `.pt`. El recorte puede omitir información en los bordes.
    4. **Orientación:** Qwen redacta texto a partir de clase y confianza. Es un modelo distinto al que clasifica la resonancia y no valida su predicción.
    ''')
    st.markdown('**Límites:** solo cuatro categorías, imágenes 2D similares a las de entrenamiento, sin localización ni segmentación del tumor y sin validación clínica externa. La app no detecta automáticamente si subiste una fotografía u otra imagen ajena al problema: también podría asignarle una clase.')
    st.markdown('**Para el grupo:** `app.py` organiza la interfaz; `modelo.py` carga y ejecuta EfficientNet; `recomendacion.py` consulta el LLM; `metricas_test.json` conserva los resultados del notebook.')
    st.link_button('Ver código del proyecto', 'https://github.com/Crisgutierrezc/proyecto-tumores-cerebrales')

st.caption('Proyecto académico · Brain Tumor MRI Dataset · Resultados para exploración y aprendizaje')
