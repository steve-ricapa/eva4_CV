# Ejercicio 1: Encoders

Inferencia ImageNet con ResNet18 preentrenada e inspección de mapas de activación de `layer3`.

## Archivos
- `encoders.ipynb`: notebook, resultados y visualizaciones integradas.
- `data/dog.jpg`: imagen de entrada; su URL original aparece en el notebook.
- `models/resnet18-f37072fd.pth`: pesos locales.

## Requisitos
Python, PyTorch, torchvision, Pillow, Matplotlib, Jupyter Notebook e ipykernel.
Funciona con CPU o CUDA; la GPU es opcional.

## Ejecución
Prepara el entorno común siguiendo el README principal. Desde la raíz del proyecto en Windows:
```powershell
.\.venv\Scripts\python -m jupyter notebook 01_encoders/encoders.ipynb
```
Selecciona el kernel **Python (examen-cv)** y ejecuta todas las celdas en orden.
Si ese kernel no está registrado, selecciona uno con las dependencias indicadas.
La imagen y los pesos se incluyen en `data/` y `models/`; consérvalos para ejecutar sin descargar.
Las rutas se resuelven desde la raíz del proyecto o la carpeta del ejercicio.

## Resultados esperados
- Clase: **Samoyed**, probabilidad aproximada **88,46 %**.
- Entrada: `(1, 3, 224, 224)`; `layer3`: `(1, 256, 14, 14)`; salida: `(1, 1000)`.
- La imagen original y ocho feature maps aparecen en las salidas del notebook.
