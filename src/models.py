"""Helpers de modelos PyTorch: dispositivo, transferencia e inferencia ligera.

No se instancia nada al importar (lazy) ni se descargan pesos hasta llamar explícitamente.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

import config


def get_device() -> torch.device:
    return torch.device(config.DEVICE)


# --- Transformaciones estándar para modelos preentrenados de torchvision ---


def preprocess_for_torch(img_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """Imagen BGR (H,W,C uint8) -> batch tensor normalizado ImageNet en DEVICE."""
    import cv2

    from .preprocessing import resize

    img = resize(img_bgr, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    t = torch.from_numpy(rgb.transpose(2, 0, 1))[None]  # (1,3,H,W)
    return t.to(get_device())


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Inferencia sin gradiente; devuelve softmax (probabilidades) como numpy."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return torch.softmax(logits, dim=1).cpu().numpy()


def vram_usage() -> dict:
    """VRAM asignada/reservada en MB (solo CUDA)."""
    if not torch.cuda.is_available():
        return {"allocated_mb": None, "reserved_mb": None}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
    }
