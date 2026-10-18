# Clasificación de tumores cerebrales en resonancias magnéticas

Aplicación educativa en Streamlit para el punto 4.3. Utiliza el checkpoint entrenado en el punto 4.2; no reentrena ni usa CLIP para la predicción de la app.

## Publicar paso a paso

1. Descomprime el ZIP en tu computador.
2. En el repositorio `Crisgutierrezc/proyecto-tumores-cerebrales`, abre **Code → Add file → Upload files**.
3. Sube **los archivos que están dentro de esta carpeta**, no el ZIP ni la carpeta contenedora. Deben quedar en la raíz: `app.py`, `modelo.py`, `recomendacion.py`, `requirements.txt`, `metricas_test.json`, `matriz_confusion.png`, `brain_tumor_efficientnet_b0.pt` y este README. Arrastra también la carpeta `.streamlit` con su archivo `config.toml`, que define el tema visual y limita cargas a 10 MB. Si no puedes subir la carpeta, crea en GitHub un archivo llamado `.streamlit/config.toml` y copia su contenido.
4. Confirma con **Commit changes**. El `.pt` pesa aproximadamente 16 MB y se puede subir desde la web de GitHub.
5. En Streamlit Community Cloud, elige **Create app → Deploy a public app from GitHub** (el texto puede variar).
6. Repositorio: `Crisgutierrezc/proyecto-tumores-cerebrales`; rama: la que contiene los archivos, normalmente `main`; archivo principal: `app.py`.
7. En configuración avanzada, selecciona **Python 3.12**. No requiere Secrets en esta versión.
8. Pulsa **Deploy** y espera a que instale las dependencias. Copia el enlace `.streamlit.app`.
9. Prueba una imagen del dataset y pulsa también **Generar recomendación con IA**. El punto del LLM solo se considera verificado al obtener una respuesta en el servidor publicado.

El LLM abierto Qwen2.5-0.5B-Instruct se descarga al primer uso sin cuenta ni API de pago. Su descarga es de aproximadamente 1 GB; necesita conexión, tiempo y memoria. La app lo carga en bfloat16 y limita a una generación simultánea. Si la instancia gratuita no tiene memoria suficiente, habrá que conectar una API de LLM o cambiar el alojamiento: el error se muestra explícitamente, sin simular una recomendación. No está garantizado el funcionamiento del LLM en cualquier cuota de Community Cloud hasta probar el despliegue real.

## Ejecutar en un computador

Con Python 3.12 y una terminal situada en esta carpeta:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Requisitos del profesor

| Requisito | Implementación |
|---|---|
| Carga de imagen | JPG/PNG, vista previa y botón para clasificar |
| Clase, confianza y todas las probabilidades | Nombre legible, confianza softmax, barras y tabla |
| Desempeño real de prueba | Pestaña propia, accuracy, F1 macro, por clase y matriz |
| Botón que envía el resultado a un LLM | Clase y confianza a Qwen; texto generado en el servidor |
| Uso responsable | Aviso permanente, alerta bajo 80 %, revisión humana |

El umbral 80 % es una regla de interfaz orientativa, no un umbral clínico validado. La confianza de una imagen no equivale a accuracy ni a probabilidad clínica.

## Resultados recibidos

Fuente: salidas guardadas de `Certamen1_IA.ipynb`, entregado con el archivo `.pt`.

- Training: 4.760 imágenes para entrenar y 840 para validar; Testing: 1.600 imágenes.
- EfficientNet-B0: accuracy 93,125 % (1.490/1.600); F1 macro mostrado en el notebook 0,9294.
- CLIP: accuracy 26,125 % (418/1.600); F1 macro mostrado 0,2107.
- 36 de 1.200 imágenes tumorales predichas como sin tumor; recall binario de tumor 97 %.

Las cifras por clase fueron transcritas desde las salidas con cuatro decimales. La matriz PNG proviene del notebook. No se ha repetido la evaluación completa sobre el dataset al construir la app. Se incluye una huella SHA256 del `.pt` recibido para detectar sustituciones posteriores, no para probar por sí sola la correspondencia clínica o experimental.

## Colaboración y presentación

Los colaboradores pueden editar archivos desde GitHub o trabajar en ramas y proponer pull requests. Las modificaciones a la rama desplegada actualizan la app. Mantener juntos el modelo, las clases, el preprocesamiento y sus métricas. No subir claves, datos personales ni imágenes de pacientes reales.

Para explicar el código: Streamlit vuelve a ejecutar `app.py` con cada interacción; `st.session_state` conserva el resultado por sesión y lo borra si cambia la imagen. `st.cache_resource` conserva el clasificador cargado. El LLM se carga una sola vez y no recibe píxeles. El modelo de clasificación siempre usa `eval()` y `torch.inference_mode()`.

## Limitaciones y fuentes

Uso académico, sin validación clínica. No hay control fiable de imágenes fuera de dominio ni garantía de división independiente por paciente. El LLM puede generar errores: su respuesta siempre se presenta como texto que requiere revisión. Un filtro conservador descarta algunas interpretaciones clínicas indebidas; no garantiza detectar todos los errores. Los avisos de revisión profesional permanecen visibles aunque falle la generación.

- Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Streamlit: https://docs.streamlit.io/deploy/streamlit-community-cloud
- timm: https://huggingface.co/docs/timm/quickstart
- Modelo de texto (Apache 2.0): https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Consultar las condiciones de los datos y pesos antes de cualquier uso distinto a la entrega académica. Las licencias de los modelos no se sustituyen por este README.
