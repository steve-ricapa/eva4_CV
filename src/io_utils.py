"""Utilidades de entrada/salida y medición de tiempos."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from config import DATA_RAW

IMREAD_FLAGS = {
    "color": cv2.IMREAD_COLOR,
    "gray": cv2.IMREAD_GRAYSCALE,
    "unchanged": cv2.IMREAD_UNCHANGED,
}


def load_image(path: str | Path, mode: str = "color") -> np.ndarray:
    """Carga una imagen con OpenCV. mode: 'color' (BGR), 'gray' o 'unchanged'."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe: {path}")
    img = cv2.imread(str(path), IMREAD_FLAGS[mode])
    if img is None:
        raise ValueError(f"OpenCV no pudo leer: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> Path:
    """Guarda una imagen creando el directorio padre si hace falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise IOError(f"No se pudo escribir: {path}")
    return path


def list_images(folder: str | Path = None, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> list[Path]:
    """Lista imágenes de una carpeta (por defecto data/raw)."""
    folder = Path(folder) if folder else DATA_RAW
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convierte BGR (OpenCV) a RGB (matplotlib). Si es gris, la devuelve tal cual."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


# --- Timing ---


@contextmanager
def timed(label: str = "", verbose: bool = True) -> Iterator[dict]:
    """Contexto que mide tiempo de ejecución.

    with timed("mi operacion") as t:
        ...
    print(t["elapsed"])
    """
    info: dict = {"elapsed": None}
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed"] = time.perf_counter() - t0
        if verbose and label:
            print(f"[{label}] {info['elapsed'] * 1000:.1f} ms")


def benchmark(fn: Callable, *args, repeats: int = 5, warmup: int = 1, **kwargs) -> dict:
    """Ejecuta fn varias veces y devuelve tiempos (ms): min/mean/max y resultado de la última llamada."""
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "min_ms": min(times),
        "mean_ms": float(np.mean(times)),
        "max_ms": max(times),
        "repeats": repeats,
        "result": result,
    }
