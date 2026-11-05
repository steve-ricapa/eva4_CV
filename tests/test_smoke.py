"""Smoke test: importaciones, rutas, dispositivo y utilidades básicas."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    import config
    import src.io_utils as io
    import src.preprocessing as pp
    import src.features as ft
    import src.models as md
    import src.evaluation as ev
    import src.visualization as viz
    assert all([config.DEVICE, io, pp, ft, md, ev, viz])


def test_paths_exist():
    import config
    for d in (config.DATA_RAW, config.DATA_PROCESSED, config.OUTPUT_FIG, config.OUTPUT_RESULTS):
        assert d.is_dir(), f"Falta carpeta: {d}"
    assert (config.PROJECT_ROOT / "notebooks" / "examen.ipynb").is_file()


def test_device_info():
    import config
    info = config.device_info()
    assert info["device"] in ("cuda", "cpu")
    if info["device"] == "cuda":
        assert info["gpu_name"] and info["vram_total_mb"] > 0
        print(f"GPU: {info['gpu_name']} | VRAM libre {info['vram_free_mb']} / {info['vram_total_mb']} MB")


def test_load_and_convert():
    import cv2
    from src.io_utils import load_image, save_image, bgr_to_rgb
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[..., 2] = 255  # rojo en BGR
    tmp = Path(__file__).parent / "_tmp.png"
    save_image(img, tmp)
    loaded = load_image(tmp)
    tmp.unlink()
    assert loaded.shape == (16, 16, 3)
    assert bgr_to_rgb(loaded)[0, 0, 0] == 255  # rojo en RGB


def test_preprocessing():
    from src.preprocessing import to_gray, to_float01, to_uint8, normalize_minmax
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    g = to_gray(img)
    assert g.shape == (8, 8)
    f = to_float01(img)
    assert f.max() <= 1.0 and f.dtype == np.float32
    assert to_uint8(f * 255).dtype == np.uint8
    assert normalize_minmax(np.array([0.0, 5.0, 10.0])).max() == pytest.approx(1.0)


def test_timing():
    from src.io_utils import timed, benchmark
    with timed("noop", verbose=False) as t:
        sum(range(10))
    assert t["elapsed"] >= 0
    res = benchmark(lambda x: x * 2, 3, repeats=3)
    assert res["result"] == 6 and res["repeats"] == 3


def test_metrics():
    from src.evaluation import accuracy, iou, dice, mse, psnr, log_result, results_df, clear_results
    assert accuracy([1, 0, 1], [1, 0, 0]) == pytest.approx(2 / 3)
    m = np.zeros((4, 4), bool)
    m[:2] = True
    assert iou(m, m) == 1.0 and dice(m, m) == 1.0
    assert mse([1, 2], [1, 2]) == 0.0
    assert psnr([1, 2], [1, 2]) == float("inf")
    log_result("smoke", accuracy=1.0)
    assert len(results_df()) == 1
    clear_results()


def test_visualization_save(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src import visualization as viz
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    fig = viz.show_images([np.zeros((8, 8), dtype=np.uint8)], titles=["t"], cmap="gray", save="smoke_test")
    assert fig is not None
    saved = viz.OUTPUT_FIG / "smoke_test.png"
    assert saved.is_file()
    saved.unlink()
    plt.close("all")
