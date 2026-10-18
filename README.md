# Clasificación de tumores cerebrales en resonancias magnéticas

Aplicación educativa en Streamlit para el punto 4.3. Utiliza el checkpoint entrenado en el punto 4.2; no reentrena ni usa CLIP para la predicción de la app.

## Publicar paso a paso

1. Descomprime el ZIP en tu computador.
2. En el repositorio `Crisgutierrezc/proyecto-tumores-cerebrales`, abre **Code → Add file → Upload files**.
3. Sube **los archivos que están dentro de esta carpeta**, no el ZIP ni la carpeta contenedora. Deben quedar en la raíz: `app.py`, `modelo.py`, `recomendacion.py`, `requirements.txt`, `metricas_test.json`, `matriz_confusion.png`, `brain_tumor_efficientnet_b0.pt` y este README. Arrastra también la carpeta `.streamlit` con su archivo `config.toml`, que define el tema visual y limita cargas a 10 MB. Si no puedes subir la carpeta, crea en GitHub un archivo llamado `.streamlit/config.toml` y copia su contenido.
4. Confirma con **Commit changes**. El `.pt` pesa aproximadamente 16 MB y se puede subir desde la web de GitHub.
5. En Streamlit Community Cloud, elige **Create app → Deploy a public app from GitHub** (el texto puede variar).
6. Repositorio: `Crisgutierrezc/proyecto-tumores-cerebrales`; rama: la que contiene los archivos, normalmente `main`; archivo principal: `app.py`.
7. En configuración avanzada, selecciona **Python 3.12**. Añade GEMINI_API_KEY en Secrets antes de probar la orientación.
8. Pulsa **Deploy** y espera a que instale las dependencias. Copia el enlace `.streamlit.app`.
9. Prueba una imagen del dataset y pulsa también **Generar recomendación con IA**. El punto del LLM solo se considera verificado al obtener una respuesta en el servidor publicado.

## Integración Gemini API (4.5)

La orientación usa Google Gemini por API. Se envían solo clase, confianza y contexto educativo; no se envía imagen, ruta, nombre ni datos personales. La respuesta se organiza en significado, revisión recomendada y límites. El prompt exige revisión humana para baja confianza (<80 %, umbral orientativo) y prohíbe diagnósticos, gravedad, tratamientos, falsa tranquilidad y afirmaciones inventadas. Es orientación de revisión de la predicción, no una evaluación de salud.

Guardar GEMINI_API_KEY exclusivamente en Secrets de Streamlit o variable de entorno. Opcional: GEMINI_MODEL permite configurar otro modelo compatible; valor por defecto gemini-3.5-flash-lite. Nunca subir una clave a GitHub. Para desarrollo local puede usarse .streamlit/secrets.toml, que debe estar excluido del repositorio.

La llamada usa la API REST oficial, timeout y errores seguros de clave/permisos/cuota. No activa facturación ni sustituye fallos por respuestas simuladas. La disponibilidad y cuota dependen de la cuenta de Google. Las respuestas incompletas no se presentan como orientación válida. El filtro automático no garantiza detectar todos los errores.

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
| Botón que envía el resultado a un LLM | Clase y confianza a Gemini por API; clave en Secrets |
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

Para explicar el código: Streamlit vuelve a ejecutar `app.py` con cada interacción; `st.session_state` conserva el resultado por sesión y lo borra si cambia la imagen. `st.cache_resource` conserva el clasificador cargado. El LLM se consulta por API y no recibe píxeles; la clave se lee de Secrets sin mostrarse. El modelo de clasificación siempre usa `eval()` y `torch.inference_mode()`.

## Limitaciones y fuentes

Uso académico, sin validación clínica. No hay control fiable de imágenes fuera de dominio ni garantía de división independiente por paciente. El LLM puede generar errores: su respuesta siempre se presenta como texto que requiere revisión. Un filtro conservador descarta algunas interpretaciones clínicas indebidas; no garantiza detectar todos los errores. Los avisos de revisión profesional permanecen visibles aunque falle la generación.

- Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Streamlit: https://docs.streamlit.io/deploy/streamlit-community-cloud
- timm: https://huggingface.co/docs/timm/quickstart
- Gemini API: https://ai.google.dev/api/generate-content

Consultar las condiciones de los datos y pesos antes de cualquier uso distinto a la entrega académica. Las licencias de los modelos no se sustituyen por este README.
