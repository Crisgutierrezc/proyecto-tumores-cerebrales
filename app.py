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
if st.session_state.get('version_orientacion') != 'lenguaje-claro-1':
    st.session_state.pop('recomendacion', None)
    st.session_state['version_orientacion'] = 'lenguaje-claro-1'
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
st.warning('Esta herramienta es un proyecto educativo. Puede equivocarse y no reemplaza la evaluación de un profesional de salud.')

with st.sidebar:
    st.header('Acerca del proyecto')
    st.write('**Modelo:** EfficientNet-B0 con fine-tuning')
    st.write('**Categorías:** glioma, meningioma, sin tumor y tumor hipofisario.')
    st.divider()
    st.caption('Para probarla, usa imágenes de la base del proyecto. Evita subir nombres u otros datos que identifiquen a una persona.')
    st.caption('La aplicación no añade tus imágenes a entrenamiento. El archivo se procesa en el servidor durante la sesión.')

analizar, desempeno, explicar = st.tabs(['Analizar imagen', 'Qué tan bien funciona', 'Sobre el proyecto'])

with analizar:
    izquierda, derecha = st.columns([1,1.15], gap='large')
    with izquierda:
        st.subheader('1. Sube una resonancia')
        archivo = st.file_uploader('Arrastra tu imagen o selecciona un archivo', type=['jpg','jpeg','png'])
        st.caption('JPG o PNG · máximo 10 MB. Sube una sola imagen de resonancia cerebral, no un estudio médico completo.')
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
        st.subheader('2. Mira el resultado')
        resultado = st.session_state.get('resultado')
        if resultado is None:
            st.info('Sube una imagen y pulsa «Clasificar imagen» para ver el resultado.')
        else:
            st.metric('Categoría sugerida por la IA', NOMBRES[resultado['clase']])
            st.metric('Confianza del modelo', f"{resultado['confianza']:.1%}")
            st.caption('Este porcentaje indica cuánto favorece el modelo esta categoría frente a las otras tres. No indica la probabilidad de tener un tumor ni garantiza que la respuesta sea correcta.')
            if resultado['baja_confianza']:
                st.warning('El modelo tiene dudas: su confianza es menor al 80 %. No te quedes solo con esta respuesta; pide que un profesional revise la imagen. El 80 % es una referencia del proyecto, no un criterio médico.')
            if resultado['clase'] == 'notumor':
                st.info('«Sin tumor» es la categoría elegida por la IA. Por sí sola, esta respuesta no permite saber si una persona está sana.')
            st.markdown('**Cómo se repartió la confianza entre las categorías**')
            tabla = pd.DataFrame([{'Categoría':NOMBRES[c], 'Confianza':p} for c,p in resultado['probabilidades'].items()])
            barras = alt.Chart(tabla).mark_bar(cornerRadiusEnd=5, color='#238b8e').encode(
                x=alt.X('Confianza:Q', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format='%'), title='Confianza del modelo (%)'),
                y=alt.Y('Categoría:N',sort=None,title=None),
                tooltip=['Categoría',alt.Tooltip('Confianza:Q', format='.2%')]).properties(height=200)
            st.altair_chart(barras, use_container_width=True)
            with st.expander('Ver porcentajes de las cuatro categorías'):
                st.dataframe(tabla.assign(Confianza=tabla.Confianza.map(lambda p:f'{p:.2%}')),
                             hide_index=True, width='stretch')

    st.divider()
    st.subheader('3. Entiende el resultado y el siguiente paso')
    st.write('¿Qué hago con esta respuesta? Pide una explicación sencilla y una orientación para revisarla con un profesional de salud.')
    st.caption('Gemini recibe solo la categoría, el porcentaje de confianza y el contexto del proyecto. No recibe tu imagen ni su nombre.')
    if st.button('Generar recomendación con IA', disabled=st.session_state.get('resultado') is None):
        st.session_state.pop('recomendacion', None)
        try:
            with st.spinner('Preparando una explicación para ti…'):
                st.session_state['recomendacion'] = generar_recomendacion(st.session_state['resultado'])
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
                st.markdown(f'#### {titulo}')
                st.text(contenido)
            else:
                st.text(bloque)

with desempeno:
    st.subheader('Cómo funcionó el modelo con imágenes de prueba')
    m = metricas()
    st.write('Probamos el modelo con **1.600 imágenes** que no se usaron para ajustar sus pesos. Estos resultados resumen esa prueba; no dicen si la respuesta de tu imagen es correcta.')
    a,b,c = st.columns(3)
    a.metric('Aciertos en la prueba (accuracy)', f"{m['efficientnet']['accuracy']:.2%}")
    b.metric('F1 macro en test', f"{m['efficientnet']['f1_macro']:.2%}")
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

st.caption('Proyecto académico · Brain Tumor MRI Dataset · Resultados para exploración y aprendizaje')
