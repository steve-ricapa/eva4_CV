"""Añade el análisis sin regenerar ni alterar las celdas obligatorias."""
from pathlib import Path
import shutil
import nbformat as nf

root = Path(__file__).resolve().parent
path = root/'yolo.ipynb'
backup = root/'outputs'/'yolo_original.ipynb'
if not backup.exists():
    shutil.copy2(path, backup)
nb = nf.read(path,4)
marker = '## Análisis complementario: recuperación de objetos pequeños en escenas concurridas'
for i,c in enumerate(nb.cells):
    if c.source.startswith(marker):
        nb.cells = nb.cells[:i]; break
nb.cells.extend([
    nf.v4.new_markdown_cell(marker + '''

**Diagnóstico.** El baseline obligatorio usa `imgsz=640`: 1880×1254 pasa a aproximadamente 640×427 antes del padding al múltiplo del stride. El tensor real se registra mediante un hook. `max_det=300` no limita las 16 detecciones (14 personas). `Results.plot()` dibuja en coordenadas originales; se convierte BGR→RGB al mostrar. Oclusión, reducción de detalle y capacidad de YOLO11n son causas plausibles de omisiones; NMS no es la única causa.

**Referencia:** la búsqueda pública por `shutterstock 86973980 crowd people count` encontró la fotografía en [una publicación de salud](https://management-krankenhaus.de/de/news/die-gesundheit-der-bevoelkerung-im-europaeischen-vergleich), pero no un conteo verificable. No hay ground truth disponible para este análisis. No damos un conteo manual sin marcar exhaustivamente personas: cabezas parcialmente ocultas y recortes de borde hacen ambigua esa estimación. No se reportan precision, recall, F1 ni mAP.

**Protocolo:** solo personas (`classes=[0]`), YOLO11n local, sin entrenamiento. Resoluciones a conf=0.25; después barrido de confianza en 1280 para conservar detalle. Tiempos: mediana de tres ejecuciones tras calentamiento, extremo a extremo sin dibujo. Slicing: tiles 640, paso 512, coordenadas trasladadas, NMS global 0.50 independiente del IoU local de YOLO. Los pares con IoU>0.70 son candidatos a duplicación, NO falsos positivos certificados.'''),
    nf.v4.new_code_cell('''import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from complementario import ejecutar
registros_extra, cajas_extra = ejecutar(model, img, ROOT, device)
from IPython.display import display, Markdown
tabla = ['| Método | imgsz | conf | Personas | Conf. mín/media/máx | Tiempo (ms) |',
         '|---|---:|---:|---:|---|---:|']
for f in registros_extra:
    tabla.append(f"| {f['metodo']} | {f['imgsz']} | {f['conf']:.2f} | {f['personas']} | {f['conf_min']:.3f}/{f['conf_media']:.3f}/{f['conf_max']:.3f} | {f['tiempo_ms']:.1f} |")
display(Markdown(chr(10).join(tabla)))
print((ROOT / 'outputs' / 'diagnostico.json').read_text())'''),
    nf.v4.new_markdown_cell('''### Interpretación

Las figuras comparan detecciones de **personas**, no todas las clases. Aumentar resolución evita perder tantos píxeles por persona; slicing presenta cada región a mayor escala efectiva, pero corta cuerpos en sus bordes y puede generar duplicados con cajas distintas que sobreviven al NMS global. Reducir confianza admite tanto personas reales como respuestas inciertas. Solapamiento entre personas distintas no implica duplicación. Sin anotaciones, más cajas no demuestra mejor precisión.

**Inspección visual organizada:** primer plano (y≥800), zona media (400≤y<800) y fondo (y<400). Los IDs y scores en los JPG individuales permiten revisar cajas. En slicing se aprecian cajas de cuerpo parciales que terminan en límites de tile, por ejemplo alrededor de la persona con jersey claro en x≈160, y≈650. En conf=0.05 aparecen cajas de cabeza/cuerpo de una misma persona y candidatos ambiguos. No se identificó ni anotó exhaustivamente cada falso positivo: no hay una cantidad verificada. Cero pares con IoU>0.70 no descarta duplicados de distinto tamaño. La columna de centros en el fondo describe localización de cajas, no cuenta personas reales.

**Recomendación provisional:** imagen completa, `imgsz=1280, conf=0.15, iou=0.50, classes=[0]`: 62 cajas de persona frente a las 14 del baseline. Al revisar las 16 cajas adicionales respecto a conf=0.25 se observan candidatos sobre personas visibles de la zona media, aunque siguen siendo predicciones, no aciertos certificados. Es un compromiso conservador por simplicidad y ausencia de fragmentación artificial por tiles. Slicing a conf=0.25 ofrece mayor cobertura exploratoria (114 cajas), pero exige revisión de duplicados; no es un conteo de personas distintas. Conf=0.05 (130 cajas) no se recomienda como resultado principal solo por producir más detecciones. La multitud lejana sigue muy incompleta.

**Limitaciones del experimento:** una imagen, sin anotaciones, CPU y tres repeticiones; los tiempos tienen variabilidad y no permiten atribuir pequeñas diferencias al umbral. No había YOLO11s local y no se descargaron pesos adicionales. El experimento obligatorio se conserva intacto; este complemento no sustituye sus comparaciones de conf/NMS.''')
])
nf.write(nb,path)
