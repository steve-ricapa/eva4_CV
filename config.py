"""Configuración central: rutas, dispositivo y reproducibilidad."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# --- Rutas base del proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FIG = OUTPUT_DIR / "figures"
OUTPUT_RESULTS = OUTPUT_DIR / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Crear carpetas de salida si no existen (idempotente)
for _d in (DATA_RAW, DATA_PROCESSED, OUTPUT_FIG, OUTPUT_RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibilidad ---
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --- Dispositivo (torch) ---


def get_device() -> str:
    """Devuelve 'cuda' si hay GPU utilizable, si no 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def device_info() -> dict:
    """Info de hardware: dispositivo, nombre de GPU y memoria VRAM libre/total (MB)."""
    info = {"device": get_device(), "gpu_name": None, "vram_free_mb": None, "vram_total_mb": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["vram_free_mb"] = round(free / 1e6, 1)
            info["vram_total_mb"] = round(total / 1e6, 1)
    except ImportError:
        pass
    return info


DEVICE = get_device()

set_seed()
