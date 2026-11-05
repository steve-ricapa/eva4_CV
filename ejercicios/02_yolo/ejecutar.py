"""Ejecuta el notebook con el intérprete local sin registrar kernels globales."""
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
local = root / 'outputs' / 'jupyter'
kernel = local / 'kernels' / 'yolo-local'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'YOLO local', 'language': 'python',
}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(local)
os.environ['JUPYTER_CONFIG_DIR'] = str(local / 'config')
os.environ['JUPYTER_DATA_DIR'] = str(local / 'data')
os.environ['JUPYTER_RUNTIME_DIR'] = str(local / 'runtime')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from nbclient import NotebookClient
import nbformat

path = root / 'yolo.ipynb'
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
NotebookClient(nb, timeout=300, kernel_name='yolo-local', resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(nb, path)
print('Notebook ejecutado correctamente:', path)
