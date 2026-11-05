"""Visualización: comparaciones, histogramas, grids. Guarda figures en outputs/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_FIG
from .io_utils import bgr_to_rgb

DEFAULT_CMAP = "gray"


def _prep(img: np.ndarray, cmap: str | None) -> np.ndarray:
    return img if (cmap or img.ndim == 2) is None else bgr_to_rgb(img)


def show_images(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    cmap: str | None = None,
    cols: int | None = None,
    figsize: tuple = (12, 4),
    save: str | None = None,
) -> plt.Figure:
    """Muestra n imágenes en una fila (o grid). cmap='gray' aplica a todas.

    Asume BGR si es 3 canales y cmap es None (convierte a RGB).
    """
    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = images[i]
            img = bgr_to_rgb(img) if (img.ndim == 3 and cmap is None) else img
            ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
    plt.tight_layout()
    if save:
        save_figure(fig, save)
    plt.show()
    return fig


def show_hist(
    img: np.ndarray,
    bins: int = 256,
    title: str = "Histograma",
    save: str | None = None,
) -> plt.Figure:
    """Histograma de una imagen en gris (o promedio de canales si es color)."""
    g = img if img.ndim == 2 else img.mean(axis=2)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(g.ravel(), bins=bins, color="steelblue", edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    if save:
        save_figure(fig, save)
    plt.show()
    return fig


def plot_comparison(x, ys: list, labels: list[str], xlabel: str = "", ylabel: str = "",
                    title: str = "", save: str | None = None) -> plt.Figure:
    """Curvas comparativas (ej. métrica vs. iteración/umbral para baseline y variantes)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for y, lab in zip(ys, labels):
        ax.plot(x, y, marker="o", label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        save_figure(fig, save)
    plt.show()
    return fig


def plot_confusion_matrix(cm, labels: list[str] | None = None, save: str | None = None) -> plt.Figure:
    """Heatmap de matriz de confusión (acepta DataFrame de evaluation.confusion)."""
    arr = cm.values if hasattr(cm, "values") else np.asarray(cm)
    labels = labels or (list(cm.columns) if hasattr(cm, "columns") else [str(i) for i in range(len(arr))])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, arr[i, j], ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black", fontsize=9)
    fig.colorbar(im)
    ax.set_title("Matriz de confusión")
    plt.tight_layout()
    if save:
        save_figure(fig, save)
    plt.show()
    return fig


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Guarda la figura en outputs/figures/ con nombre dado (añade .png si falta)."""
    if not name.endswith(".png"):
        name += ".png"
    path = OUTPUT_FIG / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {path}")
    return path
