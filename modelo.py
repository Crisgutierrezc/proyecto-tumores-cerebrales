"""Carga e inferencia: mismo orden de clases y preprocesamiento que en 4.2."""
from pathlib import Path
from io import BytesIO
import hashlib
import warnings
import torch
import timm
from timm.data import create_transform
from PIL import Image, UnidentifiedImageError

NOMBRES = {'glioma': 'Glioma', 'meningioma': 'Meningioma',
           'notumor': 'Sin tumor', 'pituitary': 'Tumor hipofisario'}
UMBRAL_CONFIANZA = 0.80  # Regla orientativa educativa, no calibrada clínicamente.
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000

def sha256_archivo(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()

def cargar_modelo(ruta):
    # weights_only evita deserializar objetos Python arbitrarios del checkpoint.
    ckpt = torch.load(ruta, map_location='cpu', weights_only=True)
    clases = ckpt['classes']
    if ckpt['architecture'] != 'efficientnet_b0' or set(clases) != set(NOMBRES):
        raise ValueError('El archivo no corresponde al modelo del proyecto.')
    if ckpt['class_to_idx'] != {c:i for i,c in enumerate(clases)}:
        raise ValueError('El orden de las clases es inconsistente.')
    # pretrained=False: los pesos entrenados vienen del archivo .pt.
    red = timm.create_model(ckpt['architecture'], pretrained=False, num_classes=len(clases))
    red.load_state_dict(ckpt['state_dict'], strict=True)
    red.eval()
    transformacion = create_transform(**ckpt['data_config'], is_training=False)
    return red, transformacion, clases

def leer_imagen(datos):
    """Valida el contenido, no solamente el nombre o extensión del archivo."""
    if not datos or len(datos) > MAX_BYTES:
        raise ValueError('Sube una imagen de hasta 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(datos)) as original:
                if original.format not in {'JPEG', 'PNG'}:
                    raise ValueError('Usa una imagen JPG o PNG.')
                if original.width * original.height > MAX_PIXELS:
                    raise ValueError('La imagen supera 16 millones de píxeles.')
                if min(original.size) < 32:
                    raise ValueError('La imagen es demasiado pequeña (mínimo 32 × 32).')
                # No aplicar rotaciones EXIF: replicamos la lectura usada al entrenar.
                return original.convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('No se pudo leer la imagen. Revisa que sea un JPG o PNG válido.') from exc

def clasificar(imagen, recursos):
    red, transformacion, clases = recursos
    tensor = transformacion(imagen).unsqueeze(0)
    with torch.inference_mode():
        scores = red(tensor).softmax(dim=1)[0].tolist()
    ganador = max(range(len(scores)), key=scores.__getitem__)
    return {'clase': clases[ganador], 'confianza': scores[ganador],
            'probabilidades': dict(zip(clases, scores)),
            'baja_confianza': scores[ganador] < UMBRAL_CONFIANZA}


# Compatibilidad con el archivo de inicio seleccionado en Streamlit Cloud.
# Las importaciones desde app.py siguen definiendo solo las funciones.
if __name__ == '__main__':
    import runpy
    import sys
    carpeta = str(Path(__file__).resolve().parent)
    if carpeta not in sys.path:
        sys.path.insert(0, carpeta)
    runpy.run_path(str(Path(__file__).with_name('app.py')), run_name='__main__')
