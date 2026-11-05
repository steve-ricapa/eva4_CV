"""Métricas y registro de experimentos: un DataFrame de resultados comparables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_RESULTS

_RESULTS: list[dict] = []


def log_result(name: str, **metrics) -> None:
    """Registra una fila de resultados: log_result('baseline', accuracy=0.72, time_ms=41.2)."""
    _RESULTS.append({"name": name, **metrics})


def results_df() -> pd.DataFrame:
    return pd.DataFrame(_RESULTS)


def save_results(path: str | Path | None = None, name: str = "results.csv") -> Path:
    path = Path(path) if path else OUTPUT_RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df().to_csv(path, index=False)
    return path


def clear_results() -> None:
    _RESULTS.clear()


# --- Métricas comunes (vectorizadas, sin sklearn para no obligar a instalar nada extra) ---


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def iou(mask_a, mask_b) -> float:
    """Intersección sobre unión entre dos máscaras binarias (0/1 o bool)."""
    a, b = np.asarray(mask_a).astype(bool), np.asarray(mask_b).astype(bool)
    union = a | b
    return float((a & b).sum() / union.sum()) if union.any() else 1.0


def dice(mask_a, mask_b) -> float:
    """Coeficiente Dice (F1 de píxeles) entre dos máscaras binarias."""
    a, b = np.asarray(mask_a).astype(bool), np.asarray(mask_b).astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else 1.0


def confusion(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Matriz de confusión como DataFrame etiquetado."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[list(labels).index(t), list(labels).index(p)] += 1
    return pd.DataFrame(mat, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def psnr(a, b, data_range: float = 255.0) -> float:
    err = mse(a, b)
    return float("inf") if err == 0 else 10.0 * np.log10(data_range**2 / err)
