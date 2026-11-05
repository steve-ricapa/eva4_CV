"""Añade solo la subsección final; mantiene las celdas anteriores intactas."""
from pathlib import Path
import shutil
import nbformat as nf

root = Path(__file__).resolve().parent
path = root/'yolo.ipynb'
backup = root/'outputs/yolo_antes_refinamiento.ipynb'
if not backup.exists():
    shutil.copy2(path, backup)
nb = nf.read(path,4)
marker = '### Refinamiento final para escenas concurridas'
for i,c in enumerate(nb.cells):
    if c.source.startswith(marker):
        nb.cells = nb.cells[:i]; break
nb.cells.extend([
    nf.v4.new_markdown_cell(marker + '''

Baseline: imagen completa, 1280, conf=0.15, IoU NMS=0.50, solo personas. Ensayos sin entrenamiento ni descargas. Los tiempos son medianas de tres repeticiones tras calentamiento, sin dibujar, y varían con la carga CPU.

**Slicing refinado:** tiles 768, paso 576 (25% de solapamiento), padding gris solo en límites exteriores, traslado de coordenadas y NMS global 0.50 (distinto paso del NMS local 0.50). Para los fragmentos observados en el ensayo anterior: excluir centros en la mitad exterior del overlap (96 px) y cajas que tocan cortes internos (tolerancia 1 px). No penalizar bordes físicos de la foto. Guardias de geometría: dimensión ≤2 px o aspecto ancho/alto <0.05. Son hipótesis conservadoras de fragmentación, no reglas validadas contra anotaciones. Se registra cuántas cajas descarta cada etapa y la ablación sin filtros. Riesgo: eliminar personas válidas altas u ocluidas. No se usa un tamaño mínimo grande para evitar perder personas lejanas.

Tabla completa y contadores de ablación: `outputs/refinamiento.csv`. Cada variante tiene un JPG con IDs y confianza para revisión visual; coordenadas completas en NPZ.'''),
    nf.v4.new_code_cell('''from refinamiento import ejecutar_refinamiento
filas_finales, cajas_finales = ejecutar_refinamiento(model, img, ROOT, device)
from IPython.display import display, Markdown
tabla_final = ['| Método | imgsz/tile | conf | Cajas person | Tiempo ms | Descartes geom./centro/trunc. |',
               '|---|---:|---:|---:|---:|---|']
for f in filas_finales:
    tabla_final.append(f"| {f['metodo']} | {f['imgsz_tile']} | {f['conf']:.2f} | {f['personas']} | {f['tiempo_ms']:.1f} | {f['desc_geometria']}/{f['desc_centro']}/{f['desc_truncadas']} |")
display(Markdown(chr(10).join(tabla_final)))'''),
    nf.v4.new_markdown_cell('''**Selección final provisional: imagen completa, `imgsz=1536, conf=0.15, iou=0.50, classes=[0]`.** Produce 92 cajas frente a 62 en 1280/0.15. Inspección por zonas: aumenta cobertura de personas visibles de la zona media; el fondo lejano sigue muy incompleto. No introduce cortes artificiales de tiles ni depende de reglas de filtrado específicas de esta foto. No es una optimización validada de precisión: siguen existiendo cajas ambiguas y posibles duplicados de diferente extensión (por ejemplo, en la persona de traje alrededor de x≈1340, y≈300). Se mantiene conf=0.15 para evitar las 15 cajas adicionales con score<0.15 de la variante 1536/0.10.

**Ablación del slicing (tiles 768):** a conf=0.12, NMS sin filtros deja 161 cajas; con filtros, 128. De los 210 candidatos antes de fusión, el filtro de centros descarta 76 y el de truncamiento 6; los 128 restantes sobreviven al NMS global. La guarda de geometría descarta cero en los tres ensayos: no se le atribuye mejora. La reducción de cajas no prueba eliminación de falsos positivos: también puede perder personas válidas, especialmente si su cabeza y torso producen centros diferentes según el tile. Persisten fragmentos cabeza/cuerpo y duplicados de distinto tamaño, incluso con NMS. Slicing recupera más candidatos del fondo, pero no es nuestra opción principal por su complejidad, coste y ambigüedad de fusión.

**Entrega:** mantener las comparaciones obligatorias y presentar 1536/0.15 como recomendación complementaria razonada, no como un conteo real ni como mayor precisión demostrada. No hay ground truth para calcular precisión o recall. Los tiempos son mediciones locales con variabilidad, no benchmarks generalizables. Todas las variantes conservan sus cajas, conteos y visualizaciones en `outputs/`.'''),
])
nf.write(nb,path)
