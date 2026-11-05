"""Preprocesamiento genérico: conversiones de color, resize, tipos.

Solo utilidades reutilizables; ningún pipeline específico hasta leer el enunciado.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR/RGB -> escala de grises. Si ya es 2D, la devuelve tal cual."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convierte a float32 en [0,1] (para redes)."""
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convierte a uint8 recortando a [0,255]."""
    return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img


def resize(img: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a (width, height)."""
    return cv2.resize(img, size, interpolation=interpolation)


def resize_max_side(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Escala preservando aspecto para que el lado mayor sea max_side."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, method: str = "gaussian", ksize: int = 5) -> np.ndarray:
    """Denoising genérico. method: 'gaussian' | 'median' | 'bilateral' | 'nlmeans'.

    NOTA (regla AGENTS.md #2): usar solo si se ha observado ruido real.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "median":
        return cv2.medianBlur(img, ksize | 1)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    if method == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(img) if img.ndim == 3 else cv2.fastNlMeansDenoising(img)
    raise ValueError(f"Método desconocido: {method}")


def hist_equalize(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE (adaptativo). En gris; para color, aplica al canal L de LAB.

    NOTA (regla AGENTS.md #2): usar solo con bajo contraste observado.
    """
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Normaliza cualquier array a [0,1] (min-max)."""
    amin, amax = arr.min(), arr.max()
    return (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=np.float32)
