"""Extracción de características clásicas genérica (perezosa: solo lo pedido).

Todas las funciones aceptan imágenes en gris o BGR y convierten internamente.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import to_gray


def hist_gray(img: np.ndarray, bins: int = 256, mask: np.ndarray | None = None) -> np.ndarray:
    """Histograma de intensidades (normalizado)."""
    return cv2.calcHist([to_gray(img)], [0], mask, [bins], [0, 256]).ravel()


def hist_color(img: np.ndarray, bins: int = 32, channel: int = -1) -> np.ndarray:
    """Histograma de color. channel=-1 concatena los 3 canales BGR; 0/1/2 un canal."""
    chans = range(3) if channel == -1 else [channel]
    hists = [cv2.calcHist([img], [c], None, [bins], [0, 256]).ravel() for c in chans]
    return np.concatenate(hists)


def orb_keypoints(img: np.ndarray, n_features: int = 500):
    """Detecta y describe con ORB. Devuelve (keypoints, descriptores)."""
    orb = cv2.ORB_create(nfeatures=n_features)
    return orb.detectAndCompute(to_gray(img), None)


def hog_features(img: np.ndarray, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """HOG vía skimage si está disponible; si no, error claro."""
    try:
        from skimage.feature import hog
    except ImportError as e:
        raise ImportError("scikit-image no instalado: pip install scikit-image") from e
    from .preprocessing import resize
    g = resize(to_gray(img), size)
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)


def basic_stats(img: np.ndarray) -> dict:
    """Estadísticas de intensidad básicas (útiles para exploración)."""
    g = to_gray(img).astype(np.float32)
    return {
        "shape": img.shape,
        "mean": float(g.mean()),
        "std": float(g.std()),
        "min": float(g.min()),
        "max": float(g.max()),
        "median": float(np.median(g)),
    }
