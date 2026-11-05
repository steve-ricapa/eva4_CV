"""Experimentos locales, sin entrenamiento ni descargas. Coordenadas siempre xyxy originales."""
from pathlib import Path
from time import perf_counter
import csv
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.ops import nms, box_iou


def ejecutar(model, img, root, device):
    out = Path(root) / 'outputs'
    records, predictions = [], {}
    shapes = []

    def sync():
        if device != 'cpu':
            torch.cuda.synchronize()

    def predict(source, size, conf):
        # AutoBackend usa una copia fusionada; registrar el tensor tras letterbox.
        hook = model.predictor.model.model.model[0].register_forward_pre_hook(
            lambda m, args: shapes.append(list(args[0].shape)))
        try:
            return model.predict(source=source, imgsz=size, conf=conf, iou=0.50,
                                 classes=[0], max_det=300, device=device, verbose=False,
                                 save=False, seed=42)[0]
        finally:
            hook.remove()

    def registrar(name, size, conf, boxes, times, raw=None):
        assert boxes.shape[1] == 6
        assert np.all(boxes[:, 5] == 0), 'Solo clase person'
        assert np.all(boxes[:, 0] >= 0) and np.all(boxes[:, 2] <= img.shape[1]+1)
        assert np.all(boxes[:, 1] >= 0) and np.all(boxes[:, 3] <= img.shape[0]+1)
        scores = boxes[:, 4]
        overlaps = box_iou(torch.tensor(boxes[:, :4]), torch.tensor(boxes[:, :4])).numpy()
        pairs = int(np.triu(overlaps > 0.70, 1).sum())
        observations = {
            'full_640_025': 'Baseline; muchas omisiones visibles en fondo y zona media',
            'full_960_025': 'Mayor cobertura; persisten omisiones lejanas',
            'full_1280_025': 'Mayor cobertura conservando umbral; fondo aun poco detectado',
            'full_1280_005': 'Recupera candidatos lejanos; cajas parciales y ambiguas, posible duplicacion cabeza/cuerpo',
            'full_1280_010': 'Mas candidatos; requiere revision de cajas parciales',
            'full_1280_015': 'Compromiso conservador recomendado; no valida precision ni conteo real',
            'slicing_640_025': 'Mayor cobertura; fragmentos en bordes de tiles y posibles duplicados',
            'slicing_640_015': 'Recupera mas fondo; mas fragmentos y candidatos inciertos',
        }
        records.append(dict(metodo=name, imgsz=size, conf=conf, personas=len(boxes),
                            conf_min=float(scores.min()) if len(scores) else None,
                            conf_max=float(scores.max()) if len(scores) else None,
                            conf_media=float(scores.mean()) if len(scores) else None,
                            tiempo_ms=float(np.median(times)), pares_iou_mayor_070=pairs,
                            antes_nms_global=raw,
                            centros_en_fondo_y_menor_400=int(((boxes[:, 1]+boxes[:, 3])/2 < 400).sum()),
                            observaciones=observations[name]))
        predictions[name] = boxes

    for size in (640, 960, 1280):
        predict(img, size, 0.25)  # calentamiento, no incluido
        times = []
        for _ in range(3):
            sync(); start = perf_counter()
            r = predict(img, size, 0.25)
            sync(); times.append((perf_counter()-start)*1000)
        registrar(f'full_{size}_025', size, 0.25, r.boxes.data.cpu().numpy(), times)

    # Barrido de confianza a 1280 para conservar detalle.
    for conf in (0.05, 0.10, 0.15):
        predict(img, 1280, conf)
        times = []
        for _ in range(3):
            sync(); start = perf_counter(); r = predict(img, 1280, conf)
            sync(); times.append((perf_counter()-start)*1000)
        registrar(f'full_1280_{int(conf*100):03d}', 1280, conf, r.boxes.data.cpu().numpy(), times)

    # Tiles 640x640 con paso 512; el ajuste al borde puede aumentar el solapamiento.
    # iou_local=0.50 dentro de cada llamada YOLO; iou_global=0.50 SOLO para fusionar tiles.
    h, w = img.shape[:2]
    def starts(length):
        return sorted(set(list(range(0, max(length-640, 0)+1, 512)) + [max(length-640, 0)]))

    def sliced(conf):
        boxes = []
        for y in starts(h):
            for x in starts(w):
                b = predict(img[y:y+640, x:x+640], 640, conf).boxes.data.cpu().numpy().copy()
                b[:, [0, 2]] += x
                b[:, [1, 3]] += y
                boxes.append(b)
        b = np.concatenate(boxes)
        keep = nms(torch.tensor(b[:, :4]), torch.tensor(b[:, 4]), 0.50).numpy()
        return b[keep], len(b)

    for conf in (0.25, 0.15):
        sliced(conf)
        times = []
        for _ in range(3):
            sync(); start = perf_counter(); b, raw = sliced(conf)
            sync(); times.append((perf_counter()-start)*1000)
        registrar(f'slicing_640_{int(conf*100):03d}', 640, conf, b, times, raw)
    assert shapes, 'El hook no registro el tensor real de entrada'
    with (out/'complementario.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    np.savez(out/'predicciones_complementarias.npz', **predictions)
    (out/'diagnostico.json').write_text(json.dumps({
        'tensor_shapes': sorted(set(tuple(s) for s in shapes)),
        'original_hw': [h, w], 'max_det': 300, 'iou_local': 0.5,
        'iou_global_slicing': 0.5, 'tiles': len(starts(h))*len(starts(w)),
        'tiempo': 'mediana de 3 repeticiones tras calentamiento, extremo a extremo; sin dibujo',
        'device': str(device), 'rect': model.predictor.args.rect,
        'stride': int(model.predictor.model.stride),
        'letterbox_full': [{'imgsz': s, 'scale': s/max(h,w),
                            'resize_wh': [round(w*s/max(h,w)), round(h*s/max(h,w))],
                            'tensor_hw': [int(np.ceil(round(h*s/max(h,w))/32)*32), s]}
                           for s in (640,960,1280)]}, indent=2), encoding='utf-8')

    def draw(boxes):
        canvas = img.copy()
        for idx, b in enumerate(boxes):
            x1,y1,x2,y2 = np.rint(b[:4]).astype(int)
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,230,0),2)
            cv2.putText(canvas,f'{idx}:{b[4]:.2f}',(x1,max(y1,12)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        return cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)

    groups = [(['full_640_025','full_960_025','full_1280_025'],'resoluciones'),
              (['full_1280_005','full_1280_010','full_1280_015','full_1280_025'],'confianza_alta_resolucion'),
              (['full_640_025','full_1280_025','slicing_640_025','slicing_640_015'],'original_vs_optimizada')]
    for names, filename in groups:
        fig, axes = plt.subplots(1,len(names),figsize=(7*len(names),6))
        for ax, name in zip(axes,names):
            ax.imshow(draw(predictions[name])); ax.set_title(f'{name}: {len(predictions[name])} personas'); ax.axis('off')
        fig.tight_layout(); fig.savefig(out/f'{filename}.png',dpi=150); plt.show(); plt.close(fig)
    for name, b in predictions.items():
        cv2.imwrite(str(out/f'{name}.jpg'), cv2.cvtColor(draw(b),cv2.COLOR_RGB2BGR))
    return records, predictions
