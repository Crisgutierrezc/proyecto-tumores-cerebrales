# Neuro JVC · Clasificación de resonancias cerebrales

Proyecto académico de **Javiera García, Cristian Gutiérrez y Valentina Jiménez** para estudiantes y docentes. Permite cargar una resonancia, explorar la clasificación de EfficientNet-B0 y obtener una explicación educativa de Gemini.

- **App:** https://proyecto-tumores-cerebrales.streamlit.app/
- **Repositorio:** https://github.com/Crisgutierrezc/proyecto-tumores-cerebrales
- **Notebook ejecutado:** [notebooks/Certamen1_IA_Final.ipynb](notebooks/Certamen1_IA_Final.ipynb)

Prototipo educativo, sin validación clínica. No confirma ni descarta un diagnóstico y no debe utilizarse para tomar decisiones médicas.

## Datos: origen y licencia

**Brain Tumor MRI Dataset**, publicado en Kaggle por **Masoud Nickparvar**, reúne imágenes de Figshare, SARTAJ y Br35H. Se obtuvo mediante `kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")`.

- [Fuente y atribución: Masoud Nickparvar, Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
- Licencia indicada en Kaggle: **Attribution 4.0 International (CC BY 4.0)**, consultada el 19 de octubre de 2026. [Licencia](https://creativecommons.org/licenses/by/4.0/).
- Figshare: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
- SARTAJ: https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri
- Br35H: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

El conteo registrado en las salidas del notebook del grupo es **7.200 imágenes**, con **1.800 por clase**. No hay desbalance por cantidad entre clases. Corresponde a la versión descargada y puede diferir de descripciones de otras versiones.

| Categoría | Etiqueta | Entrenamiento | Validación | Prueba | Total |
|---|---|---:|---:|---:|---:|
| Glioma | `glioma` | 1.190 | 210 | 400 | 1.800 |
| Meningioma | `meningioma` | 1.190 | 210 | 400 | 1.800 |
| Sin tumor | `notumor` | 1.190 | 210 | 400 | 1.800 |
| Tumor hipofisario | `pituitary` | 1.190 | 210 | 400 | 1.800 |
| **Total** | | **4.760** | **840** | **1.600** | **7.200** |

Se dividió únicamente `Training` original (5.600 imágenes) en **85 % entrenamiento y 15 % validación**, estratificando por clase con `random_state=42`. `Testing` permaneció intacto. La época se eligió mediante F1 macro de validación; test se reservó para la evaluación final.

**Limitaciones:** sin identificadores de paciente no se garantiza independencia entre cortes. Rutas distintas no prueban ausencia de duplicados ni de imágenes del mismo paciente. Las fuentes, etiquetas y equipos pueden introducir sesgos. El equilibrio entre clases no garantiza representatividad poblacional ni desempeño clínico.

## Modelo, comparación y métricas

**EfficientNet-B0 de timm** parte de pesos de ImageNet, con cuatro salidas y fine-tuning de todos sus parámetros. **CLIP zero-shot** clasifica por similitud con descripciones de texto sin entrenarse con las imágenes del proyecto. Ambos se evaluaron con las mismas imágenes y métricas de prueba.

| Modelo | Accuracy en test | F1 macro |
|---|---:|---:|
| CLIP zero-shot | 26,125 % | 0,2107 |
| EfficientNet-B0 con fine-tuning | 93,125 % | 0,9294 |

EfficientNet-B0 acertó en **1.490 de 1.600 imágenes**, superando a CLIP en 67 puntos porcentuales de accuracy; por eso se eligió para la app. El notebook incluye precisión, recall, F1 por clase y matrices de confusión. Los resultados provienen de sus salidas guardadas; F1 y métricas por clase conservan su redondeo. No se repitió el entrenamiento ni la evaluación completa al preparar la app.

**Error más costoso (4.2):** 36 de las 1.200 imágenes etiquetadas con algún tumor se clasificaron como «sin tumor» (3 %). El recall binario de tumor fue 97 %: agrupa los tipos tumorales y no exige acertar el subtipo. Un falso negativo podría dar falsa tranquilidad si se tratara como diagnóstico. Confundir tipos también importa. La interpretación completa está en el notebook.

El `.pt` exportado contiene pesos, arquitectura, orden de clases y preprocesamiento. La app conserva la transformación de evaluación y usa `eval()` y `torch.inference_mode()`. El notebook comprueba la recarga del archivo exportado.

## Uso de la app

1. Carga una imagen JPG o PNG de hasta 10 MB.
2. Pulsa **Clasificar imagen**: se muestran clase, confianza y porcentajes de las cuatro categorías.
3. Pulsa **Generar recomendación con IA** para obtener una explicación de Gemini.
4. Consulta **Resultados del modelo** para ver métricas, matriz y comparación, o **Conoce el proyecto** para revisar metodología y límites.

La interfaz usa los azules de la presentación, un ícono propio y una barra lateral informativa. Las imágenes públicas de la carpeta `ejemplos/` permiten probar la app manualmente. Cambiar o retirar la imagen borra el resultado y la explicación anteriores.

La confianza softmax no es probabilidad de enfermedad ni accuracy individual. El umbral de baja confianza del 80 % es orientativo y educativo. La app avisa y pide revisión cuando la confianza es baja. No tiene un detector fiable de imágenes ajenas al problema y también podría asignarles una clase.

## Gemini API y protección de la clave

`recomendacion.py` consulta la API REST de Google Gemini. Envía únicamente texto con clase, confianza y contexto educativo: **no envía la imagen, su nombre ni su ruta**. Gemini no examina la resonancia ni verifica la clasificación.

El prompt pide lenguaje sencillo, limita el tema, prohíbe diagnósticos y tratamientos, explica la incertidumbre y solicita revisión cuando la confianza es baja. «Sin tumor» no descarta enfermedad. Solicita un JSON con `significado`, `revision` y `limites`, que se presenta en tres apartados. Las comprobaciones de estructura y contenido son básicas y no detectan todos los errores del LLM. Un fallo de API no se sustituye por una respuesta inventada.

**`GEMINI_API_KEY` se guarda en Secrets de Streamlit Cloud**, o en una variable de entorno. Localmente se puede usar `.streamlit/secrets.toml`, excluido por `.gitignore`. Nunca subir claves a GitHub. `GEMINI_MODEL` permite configurar el modelo; su valor predeterminado está en `recomendacion.py`.

La imagen se procesa en el servidor de Streamlit durante la sesión y no se añade a un entrenamiento. Utilizar imágenes públicas de prueba sin nombres ni datos personales; evitar imágenes de pacientes reales.

## Archivos y ejecución

| Archivo o carpeta | Función |
|---|---|
| `app.py` | Interfaz, carga y estado de sesión |
| `modelo.py` | Validación de imágenes, carga del modelo e inferencia |
| `recomendacion.py` | Prompt, API de Gemini y comprobaciones |
| `notebooks/Certamen1_IA_Final.ipynb` | Notebook del grupo con resultados visibles |
| `modelo/brain_tumor_efficientnet_b0.pt`, `modelo/clases.json` | Pesos y metadatos, incluidas las clases |
| `metricas_test.json` | Resultados del notebook |
| `ejemplos/ejemplo_*.jpg`, `ejemplos/ejemplos.json` | Cuatro imágenes públicas y su procedencia |
| `requirements.txt` | Dependencias de la app |
| `.streamlit/config.toml` | Tema y límite de carga |

Los pesos y las clases están en `modelo/`, y las imágenes de muestra en `ejemplos/`. La app carga los pesos desde `modelo/`. No se incluye el dataset completo. Los ejemplos conservan la atribución y condiciones de sus fuentes; no fueron elegidos por el acierto del modelo.

Para la app, usar **Python 3.12**:

```sh
python -m venv .venv
# Windows: .venv/Scripts/activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Configurar la clave antes de probar Gemini. Para entrenar, abrir el notebook en Colab con GPU; sus dependencias se instalan en sus celdas.

En Streamlit Cloud seleccionar repositorio, rama `main`, Python 3.12 y preferentemente `app.py`. El despliegue existente utiliza `modelo.py`, que conserva una entrada compatible para ejecutar `app.py`. Guardar la clave en Secrets. Antes de entregar, probar clasificación y Gemini desde otro computador o celular.

## Uso de inteligencia artificial

Durante el desarrollo utilizamos **ChatGPT, Codex y Claude** como herramientas de apoyo para escribir y comentar código, adaptar el notebook, diseñar la interfaz, integrar la API y revisar la documentación.

**Google Gemini** genera las explicaciones educativas que muestra la app a partir del resultado del modelo. No analiza la imagen ni calcula las métricas.

Como equipo, revisamos el código y asumimos la responsabilidad de comprender y explicar su funcionamiento, las decisiones tomadas y sus limitaciones. Las métricas presentadas corresponden a los resultados guardados tras ejecutar el notebook entregado.

## Fuentes técnicas

- timm: https://huggingface.co/docs/timm/quickstart
- Gemini: https://ai.google.dev/api/generate-content
- Streamlit: https://docs.streamlit.io/
