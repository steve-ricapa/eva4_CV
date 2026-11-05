# Ejercicio 3: Residual U-Net

Red ligera de segmentación binaria sin entrenar: devuelve logits, no una máscara fiable.

## Arquitectura

```mermaid
flowchart LR
    X["Entrada: 3 canales"] --> E1["enc1: bloque residual 3→16"]
    E1 --> P1["MaxPool 2"] --> E2["enc2: bloque residual 16→32"]
    E2 --> P2["MaxPool 2"] --> B["Bottleneck: bloque residual 32→64"]
    B --> U2["Interpolación bilineal"] --> C2["Concatenar"]
    E2 -- "skip U-Net" --> C2
    C2 --> D2["dec2: bloque residual 96→32"] --> U1["Interpolación bilineal"] --> C1["Concatenar"]
    E1 -- "skip U-Net" --> C1
    C1 --> D1["dec1: bloque residual 48→16"] --> H["Conv 1×1: 16→1"] --> Y["Salida: logits"]
```

Cada bloque residual suma una ruta identidad o proyección 1×1 a su rama de dos convoluciones 3×3. Las conexiones U-Net concatenan encoder y decoder. Los canales mostrados corresponden a los valores por defecto de [model.py](model.py).

## Ejecución

Prepara el entorno común siguiendo el README principal. Desde la raíz del proyecto en Windows:
```powershell
.\.venv\Scripts\python -m jupyter notebook 03_unet/unet.ipynb
```
Selecciona el kernel **Python (examen-cv)** y ejecuta todas las celdas. El notebook comprueba dimensiones y gradientes de dos parámetros seleccionados con tensores sintéticos; no descarga datos ni pesos. Mantén [model.py](model.py) junto a [unet.ipynb](unet.ipynb).
