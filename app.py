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
# Carpeta de pesos; compatibilidad con despliegues anteriores durante la migración.
RUTA_MODELO = BASE / 'modelo' / 'brain_tumor_efficientnet_b0.pt'
if not RUTA_MODELO.exists():
    RUTA_MODELO = BASE / 'brain_tumor_efficientnet_b0.pt'
# Al cambiar el prompt, no conservar en pantalla una orientación de la versión anterior.
if st.session_state.get('version_orientacion') != 'aprendizaje-2':
    st.session_state.pop('recomendacion', None)
    st.session_state['version_orientacion'] = 'aprendizaje-2'
torch.set_num_threads(2)  # CPU: evita que varias sesiones saturen el servidor.
st.set_page_config(page_title='Neuro JVC · Resonancias cerebrales', page_icon='🧠', layout='wide', initial_sidebar_state='auto')
st.markdown('''<style>
.matriz-adaptable {max-width:780px;margin:18px auto 24px;}
.matriz-adaptable table {width:100%;table-layout:fixed;border-collapse:separate;border-spacing:3px;border:0;}
.matriz-adaptable th,.matriz-adaptable td {border:0!important;text-align:center;padding:18px 4px;border-radius:5px;overflow-wrap:anywhere;}
.matriz-adaptable th {color:#536888;font-size:12px;line-height:1.4;font-weight:600;background:transparent;}
.matriz-adaptable th:first-child {width:23%;text-align:left;}
.matriz-adaptable td {font-size:17px;font-weight:700;font-variant-numeric:tabular-nums;}
@media(max-width:640px){.matriz-adaptable table {border-spacing:2px;}.matriz-adaptable th {font-size:10px;}.matriz-adaptable td {font-size:13px;padding:20px 1px;}}
.resultado-destacado {border:1px solid #b6c9e8;border-left:4px solid #315fa8;border-radius:16px;padding:24px;background:linear-gradient(135deg,#fff,#edf3ff);margin:14px 0;}
.resultado-etiqueta {font-size:.67rem;letter-spacing:1px;font-weight:700;color:#536888;}
.resultado-destacado h3 {font-size:1.8rem;line-height:1.25;color:#12233f;margin:12px 0 20px;overflow-wrap:anywhere;}
.resultado-confianza {display:flex;align-items:center;gap:14px;border-top:1px solid #dce3f2;padding-top:16px;flex-wrap:wrap;}
.resultado-confianza strong {font-size:1.8rem;color:#24558b;}
.resultado-confianza span {font-size:.85rem;color:#536888;}
.comparacion-modelos {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:22px 0 16px;}
.modelo-comparado {background:#fff;border:1px solid #dce3f2;border-radius:18px;padding:26px;}
.modelo-principal {background:linear-gradient(145deg,#fff,#eef3ff);border-color:#a8bbdf;}
.modelo-insignia {display:inline-block;border-radius:6px;background:#edf0f7;color:#536888;padding:6px 9px;font-size:.65rem;letter-spacing:1px;font-weight:750;}
.modelo-principal .modelo-insignia {background:#dce7ff;color:#24558b;}
.modelo-comparado h3 {font-size:1.4rem;letter-spacing:-.4px;color:#12233f;margin:15px 0 8px;}
.modelo-descripcion {color:#536888;font-size:.9rem;line-height:1.7;min-height:5em;margin-bottom:20px;}
.modelo-cifras {display:grid;grid-template-columns:1fr 1fr;gap:16px;border-top:1px solid #dce3f2;padding-top:18px;}
.modelo-cifras span {display:block;font-size:.75rem;color:#536888;line-height:1.5;}
.modelo-cifras strong {display:block;font-size:clamp(1.3rem,2.8vw,2rem);color:#12233f;margin:7px 0;letter-spacing:-.6px;white-space:nowrap;}
.modelo-pie {color:#536888;font-size:.8rem;line-height:1.6;margin:16px 0 0;}
.conclusion-comparacion {padding:18px 20px;border-radius:12px;background:#e8effc;border-left:3px solid #315fa8;color:#24558b;line-height:1.7;font-size:.9rem;margin-bottom:24px;}
@media(max-width:760px){.comparacion-modelos {grid-template-columns:1fr;}.modelo-comparado {padding:20px;}.modelo-descripcion {min-height:0;}.modelo-cifras strong {font-size:1.8rem;}}
.indicadores-test {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:24px 0 16px;}
.indicador-test {background:#fff;border:1px solid #dce3f2;border-radius:18px;padding:24px;box-shadow:0 5px 18px rgba(18,35,63,.04);border-top:4px solid #315fa8;}
.indicador-test:nth-child(2) {border-top-color:#8278ce;}
.indicador-test:nth-child(3) {border-top-color:#61bcd6;}
.indicador-cabecera {display:flex;align-items:center;gap:10px;font-size:.65rem;font-weight:700;letter-spacing:.8px;line-height:1.5;color:#536888;}
.indicador-icono {display:flex;align-items:center;justify-content:center;flex-shrink:0;width:30px;height:30px;border-radius:9px;background:#eaf0fc;color:#315fa8;font-size:1.1rem;}
.indicador-test:nth-child(2) .indicador-icono {background:#efecfb;color:#6c61b5;}
.indicador-test:nth-child(3) .indicador-icono {background:#e4f6fc;color:#23748f;}
.indicador-valor {font-size:clamp(2rem,4vw,3rem);font-weight:750;color:#12233f;letter-spacing:-1.5px;line-height:1.2;margin:20px 0 5px;font-variant-numeric:tabular-nums;}
.indicador-valor span {font-size:.55em;color:#536888;letter-spacing:0;margin-left:4px;}
.indicador-nombre {font-size:.95rem;font-weight:650;color:#24558b;margin-bottom:12px;}
.indicador-test p {font-size:.85rem;color:#536888;line-height:1.65;margin:0;}
@media(max-width:760px){.indicadores-test {grid-template-columns:1fr;gap:12px;}.indicador-test {padding:20px;}.indicador-valor {font-size:2.5rem;margin-top:14px;}}
.introduccion {margin:18px 0 20px;max-width:920px;}
.introduccion h2 {font-size:1.35rem;margin:0 0 8px;letter-spacing:-0.3px;}
.introduccion p {font-size:1rem;line-height:1.7;color:#4e6180;margin:0;}
.resumen-proyecto {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:20px;}
.tarjeta-resumen {background:white;border:1px solid #dce3f2;border-top:3px solid #a9c7f2;border-radius:14px;padding:20px;}
.tarjeta-resumen:nth-child(2) {border-top-color:#a9a7ec;}
.tarjeta-resumen:nth-child(3) {border-top-color:#84d4ec;}
.sobretitulo {font-size:.7rem;letter-spacing:1.2px;color:#536888;font-weight:700;}
.tarjeta-resumen h3 {font-size:1.05rem;line-height:1.4;margin:9px 0;color:#12233f;}
.tarjeta-resumen p {font-size:.9rem;line-height:1.6;color:#4e6180;margin:0;}
.stApp {background: #f5f7fc;color:#12233f;}
.block-container {max-width:1180px;padding-top:4.8rem;padding-bottom:3rem;}
h1,h2,h3,h4 {color:#24558b;}
[data-testid="stMetric"] {background:#ffffff;padding:18px;border-radius:14px;border:1px solid #dce3f2;}
.encabezado {padding:16px 22px;background:linear-gradient(110deg,#0c1b2e,#343db7);border:1px solid #35458a;border-left:5px solid #bff0fc;border-radius:16px;color:#12233f;margin-bottom:12px;}
.encabezado > div {min-width:0;}
.encabezado h1 {color:#ffffff;font-size:2rem;margin:0;}
.encabezado p {color:#e7edff;margin:8px 0 0;}
.etiqueta {font-size:11px;letter-spacing:1.4px;font-weight:700;color:#bff0fc;margin-bottom:3px;}
[data-testid="stSidebar"] {background:#eaf0fc;border-right:1px solid #dce3f2;}
.encabezado {display:flex;justify-content:space-between;align-items:center;gap:24px;}
.encabezado h1 {font-size:2rem;letter-spacing:-0.6px;}
.marca-cerebro {width:68px;min-width:58px;color:#bff0fc;}
.equipo {font-size:0.78rem;color:#d4dcff;margin-top:5px;}
[data-testid="stMetricValue"] {white-space:normal;font-size:1.4rem;}
[data-testid="stMetricValue"] > div {white-space:normal!important;overflow:visible!important;text-overflow:clip!important;font-size:1.25rem;line-height:1.35;overflow-wrap:anywhere;}
[data-testid="stMetric"] {padding:13px;}
.pasos {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:10px 0 24px;}
.paso {padding:16px 18px;border:1px solid #dce3f2;border-radius:14px;background:#fff;color:#4e6180;font-size:0.9rem;}
.paso strong {display:block;color:#24558b;margin-bottom:4px;}
.numero {display:inline-block;color:#24558b;font-weight:700;margin-right:5px;}
button[kind="primary"]:not(:disabled) {color:#ffffff!important;font-weight:700;}
[data-testid="stFileUploaderDropzone"] {border:1px dashed #879ed0;border-radius:14px;background:#fff;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#dce3f2;}
[data-baseweb="tab-highlight"] {background-color:#315fa8;}
[data-testid="stFileUploaderDropzoneInstructions"] {display:none;}
[data-testid="stFileUploaderDropzone"] button {font-size:0!important;}
[data-testid="stFileUploaderDropzone"] button::after {content:"Seleccionar imagen";font-size:14px;}
.stTabs [role="tab"] {height:auto;min-height:44px;white-space:normal;}
@media(max-width:760px){.resumen-proyecto {grid-template-columns:1fr;gap:10px;}.tarjeta-resumen {padding:16px;}}
@media(max-width:640px){.block-container {padding-top:4.6rem;padding-left:1rem;padding-right:1rem;}.pasos {grid-template-columns:1fr;gap:8px;}.marca-cerebro {display:none;}.encabezado h1 {font-size:2.2rem;}.encabezado {padding:22px;}}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def recursos_modelo():
    return cargar_modelo(RUTA_MODELO)

@st.cache_data
def metricas():
    return json.loads((BASE / 'metricas_test.json').read_text(encoding='utf-8'))

st.markdown('''<div class="encabezado"><div><div class="etiqueta">RESONANCIAS CEREBRALES · IA EDUCATIVA</div>
<h1>Neuro JVC</h1>
<p>Explora una resonancia. Comprende la predicción. Aprende con IA.</p>
<div class="equipo">Desarrollado por Javiera, Valentina y Cristian</div></div>
<svg class="marca-cerebro" viewBox="0 0 100 100" role="img" aria-label="Ícono de cerebro de Neuro JVC" fill="none" stroke="currentColor" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round">
<rect x="3" y="3" width="94" height="94" rx="26" fill="#203e72" stroke="none"/>
<path d="M49 24c-8-12-25-5-25 7-12 3-15 19-7 27-7 14 5 27 17 24 5 8 15 4 15-5V24zm2 0c8-12 25-5 25 7 12 3 15 19 7 27 7 14-5 27-17 24-5 8-15 4-15-5V24z"/>
<path d="M24 31c10-3 14 5 12 12M17 58c6-8 14-6 18-1m-1 25c-4-7-2-14 5-17M76 31c-10-3-14 5-12 12m19 15c-6-8-14-6-18-1m1 25c4-7 2-14-5-17M36 43l6 5m22-5-6 5"/>
</svg></div>''', unsafe_allow_html=True)
st.caption('Clasifica imágenes en cuatro categorías: glioma, meningioma, sin tumor y tumor hipofisario. Carga una resonancia y explora la predicción; Gemini puede ayudarte a entenderla.')
st.info('Uso educativo para estudiantes y docentes. La predicción puede ser incorrecta y no confirma ni descarta un diagnóstico médico.')

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

analizar, desempeno, explicar = st.tabs(['Explorar una imagen', 'Resultados del modelo', 'Conoce el proyecto'])

with analizar:
    izquierda, derecha = st.columns([1,1.15], gap='large')
    with izquierda:
        st.subheader('1. Selecciona una imagen')
        st.caption('Arrastra tu imagen al recuadro o pulsa «Seleccionar imagen».')
        archivo = st.file_uploader('Imagen de resonancia cerebral', type=['jpg','jpeg','png'])
        st.caption('JPG o PNG · máximo 10 MB. Una sola imagen, no un estudio médico completo.')
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
                # Miniatura solo para la vista: el modelo sigue recibiendo la imagen original.
                vista = imagen.copy()
                vista.thumbnail((420, 280))
                st.image(vista, caption='Imagen seleccionada', width=vista.width)
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
        st.subheader('2. Revisa la predicción')
        resultado = st.session_state.get('resultado')
        if resultado is None:
            st.info('Carga una imagen y pulsa «Clasificar imagen». Aquí verás la categoría predicha y la confianza del modelo.')
        else:
            confianza_visible = f"{resultado['confianza'] * 100:.1f}".replace('.', ',')
            st.markdown(f"""<div class="resultado-destacado"><span class="resultado-etiqueta">PREDICCIÓN DEL MODELO</span><h3>{NOMBRES[resultado['clase']]}</h3><div class="resultado-confianza"><strong>{confianza_visible} %</strong><span>Confianza del modelo</span></div></div>""", unsafe_allow_html=True)
            st.caption('Este porcentaje indica cuánto favorece el modelo esta categoría frente a las otras tres. No indica la probabilidad de tener un tumor ni garantiza que la respuesta sea correcta.')
            if resultado['baja_confianza']:
                st.warning('Confianza baja: menor al 80 %. Revisa este caso con tu docente antes de sacar conclusiones. Si fuera un caso real, necesita revisión profesional. El umbral es orientativo, no médico.')
            if resultado['clase'] == 'notumor':
                st.info('«Sin tumor» es la categoría elegida por la IA. Por sí sola, esta respuesta no permite saber si una persona está sana.')
            st.markdown('**Cómo se repartió la confianza entre las categorías**')
            tabla = pd.DataFrame([{'Categoría':NOMBRES[c], 'Confianza':p} for c,p in resultado['probabilidades'].items()])
            barras = alt.Chart(tabla).mark_bar(cornerRadiusEnd=5, color='#315fa8').encode(
                x=alt.X('Confianza:Q', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format='%', tickCount=5), title='Confianza del modelo (%)'),
                y=alt.Y('Categoría:N',sort=None,title=None,axis=alt.Axis(labelLimit=160,labelFontSize=12)),
                tooltip=['Categoría',alt.Tooltip('Confianza:Q', format='.2%')]).properties(height=200)
            st.altair_chart(barras, use_container_width=True)
            with st.expander('Ver porcentajes de las cuatro categorías'):
                st.dataframe(tabla.assign(Confianza=tabla.Confianza.map(lambda p:f'{p:.2%}')),
                             hide_index=True, width='stretch')

    st.divider()
    st.subheader('3. Comprende el resultado con Gemini')
    st.write('Pide a Gemini una explicación para el ejercicio: qué eligió la IA, cómo interpretar su confianza y qué revisar con tu docente.')
    st.caption('Gemini recibe solo la categoría, el porcentaje de confianza y el contexto del proyecto. No recibe tu imagen ni su nombre.')
    if st.button('Generar recomendación con IA', disabled=st.session_state.get('resultado') is None):
        st.session_state.pop('recomendacion', None)
        try:
            with st.spinner('Preparando una explicación para ti…'):
                contexto_educativo = dict(st.session_state['resultado'])
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
    accuracy_texto = f"{m['efficientnet']['accuracy'] * 100:.2f}".replace('.', ',')
    f1_texto = f"{m['efficientnet']['f1_macro'] * 100:.2f}".replace('.', ',')
    st.markdown(f"""<div class="indicadores-test">
    <article class="indicador-test"><div class="indicador-cabecera"><span class="indicador-icono" aria-hidden="true">✓</span><span>ACIERTOS EN LA PRUEBA</span></div><div class="indicador-valor">{accuracy_texto}<span>%</span></div><div class="indicador-nombre">Accuracy</div><p>1.490 de 1.600 imágenes clasificadas correctamente.</p></article>
    <article class="indicador-test"><div class="indicador-cabecera"><span class="indicador-icono" aria-hidden="true">↔</span><span>EQUILIBRIO ENTRE CLASES</span></div><div class="indicador-valor">{f1_texto}<span>%</span></div><div class="indicador-nombre">F1 macro</div><p>Combina precisión y detección, dando igual peso a las cuatro categorías.</p></article>
    <article class="indicador-test"><div class="indicador-cabecera"><span class="indicador-icono" aria-hidden="true">▦</span><span>IMÁGENES EVALUADAS</span></div><div class="indicador-valor">1.600</div><div class="indicador-nombre">Conjunto de prueba</div><p>400 imágenes por categoría para evaluar el modelo.</p></article>
    </div>""", unsafe_allow_html=True)
    st.caption('Estas cifras describen el desempeño en el conjunto de prueba. No representan la confianza ni el riesgo de enfermedad de una imagen individual.')
    st.caption('Fuente: salidas guardadas de Certamen1_IA.ipynb. F1 y métricas por clase se transcriben con el redondeo mostrado en el notebook.')
    st.dataframe(pd.DataFrame(m['por_clase']).rename(columns={
        'clase':'Clase','precision':'Precisión','recall':'Recall','f1':'F1','n':'Imágenes'}),
        hide_index=True, width='stretch')
    st.subheader('¿Qué categorías acertó o confundió?')
    st.caption('Filas: categoría real. Columnas: predicción. Los aciertos están en la diagonal.')
    tipo_matriz = st.radio('Vista de la matriz', ['Cantidad de imágenes', 'Porcentaje por categoría'], horizontal=True, key='vista_matriz')
    archivo_matriz = 'matriz_cantidad.png' if tipo_matriz == 'Cantidad de imágenes' else 'matriz_porcentaje.png'
    # Mismos conteos del notebook; tabla de calor adaptable con texto legible en celular.
    conteos = [[315,41,35,9], [7,379,0,14], [0,0,400,0], [1,2,1,396]]
    categorias = ['Glioma', 'Meningioma', 'Sin tumor', 'Tumor hipofisario']
    cabecera = '<tr><th scope="col">Real ↓<br>Predicha →</th>' + ''.join('<th scope="col">' + nombre.replace('Tumor hipofisario','Tumor<br>hipofisario') + '</th>' for nombre in categorias) + '</tr>'
    filas = []
    for nombre, fila in zip(categorias, conteos):
        celdas = []
        for valor in fila:
            proporcion = valor / sum(fila)
            etiqueta = str(valor) if tipo_matriz == 'Cantidad de imágenes' else f'{proporcion * 100:.1f} %'.replace('.', ',')
            fondo = '#24558b' if proporcion >= 0.5 else ('#d5e5f7' if valor else '#f2f6fc')
            color = '#ffffff' if proporcion >= 0.5 else '#24558b'
            celdas.append(f'<td style="background:{fondo};color:{color}">{etiqueta}</td>')
        filas.append('<tr><th scope="row">' + nombre + '</th>' + ''.join(celdas) + '</tr>')
    st.markdown('<div class="matriz-adaptable"><table aria-label="Matriz de confusión de EfficientNet-B0"><thead>' + cabecera + '</thead><tbody>' + ''.join(filas) + '</tbody></table></div>', unsafe_allow_html=True)
    st.caption('Valores de la evaluación del notebook. La diagonal azul muestra los aciertos.')

    st.subheader('¿Qué aportó entrenar con estas imágenes?')
    st.write('Comparamos ambos enfoques en las mismas 1.600 imágenes de prueba. Un porcentaje mayor indica un mejor resultado en esta evaluación.')
    def porcentaje_comparacion(valor):
        return f'{valor * 100:.2f}'.replace('.', ',') + ' %'
    diferencia = (m['efficientnet']['accuracy'] - m['clip']['accuracy']) * 100
    st.markdown(f"""<div class="comparacion-modelos">
    <article class="modelo-comparado"><span class="modelo-insignia">LÍNEA BASE</span><h3>CLIP zero-shot</h3><p class="modelo-descripcion">Clasifica usando descripciones de texto, sin entrenarse con las imágenes del proyecto.</p><div class="modelo-cifras"><div><span>Aciertos · accuracy</span><strong>{porcentaje_comparacion(m['clip']['accuracy'])}</strong></div><div><span>F1 macro</span><strong>{porcentaje_comparacion(m['clip']['f1_macro'])}</strong></div></div><p class="modelo-pie">Nuestro punto de partida para comparar.</p></article>
    <article class="modelo-comparado modelo-principal"><span class="modelo-insignia">MODELO DE NEURO JVC</span><h3>EfficientNet-B0</h3><p class="modelo-descripcion">Parte de pesos preentrenados y se adapta a las cuatro categorías mediante fine-tuning.</p><div class="modelo-cifras"><div><span>Aciertos · accuracy</span><strong>{porcentaje_comparacion(m['efficientnet']['accuracy'])}</strong></div><div><span>F1 macro</span><strong>{porcentaje_comparacion(m['efficientnet']['f1_macro'])}</strong></div></div><p class="modelo-pie">Es el modelo que analiza las imágenes en esta aplicación.</p></article>
    </div><div class="conclusion-comparacion"><strong>¿Qué observamos?</strong> EfficientNet-B0 obtuvo {diferencia:.0f} puntos porcentuales más de accuracy que CLIP en esta prueba. Por eso lo elegimos para la aplicación.</div>""", unsafe_allow_html=True)
    st.caption('División: 4.760 entrenamiento / 840 validación / 1.600 test. Sin identificadores de paciente no se garantiza independencia entre cortes; las métricas pueden ser optimistas y no implican validación clínica.')
    # Comprobación interna: avisar solo si el archivo ya no corresponde al registrado.
    actual = sha256_archivo(RUTA_MODELO)
    if actual != m['modelo_sha256']:
        st.error('El archivo del modelo cambió. Estas métricas podrían no corresponder: actualiza la evaluación antes de presentar.')

with explicar:
    st.subheader('Neuro JVC: aprender a mirar una predicción con criterio')
    st.write('Somos Javiera, Valentina y Cristian. Desarrollamos este proyecto para explorar cómo la inteligencia artificial clasifica resonancias cerebrales y cómo explicar sus resultados de forma comprensible.')
    st.markdown('**Qué puedes hacer aquí:** cargar una imagen, comparar las cuatro categorías, revisar los resultados de evaluación y pedir una explicación a Gemini.')
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

