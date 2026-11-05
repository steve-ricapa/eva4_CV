"""Refinamiento reproducible: no descarga pesos ni cambia el experimento obligatorio."""
from pathlib import Path
from time import perf_counter
import csv
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.ops import nms


def ejecutar_refinamiento(model, img, root, device):
    out = Path(root) / 'outputs'
    h, w = img.shape[:2]
    rows, predictions = [], {}

    def predict(source, size, conf):
        return model.predict(source=source, imgsz=size, conf=conf, iou=0.50,
                             classes=[0], max_det=300, device=device, save=False,
                             verbose=False, seed=42)[0].boxes.data.cpu().numpy().copy()

    def sync():
        if device != 'cpu':
            torch.cuda.synchronize()

    def medir(fn):
        fn()  # calentamiento excluido
        times = []
        for _ in range(3):
            sync(); start = perf_counter(); result = fn(); sync()
            times.append((perf_counter() - start) * 1000)
        return result, times

    def guardar(name, size, conf, b, times, stats, observaciones):
        assert b.ndim == 2 and b.shape[1] == 6
        assert np.all(b[:, 5] == 0)
        assert np.all(b[:, 0] >= 0) and np.all(b[:, 2] <= w)
        assert np.all(b[:, 1] >= 0) and np.all(b[:, 3] <= h)
        predictions[name] = b
        rows.append(dict(metodo=name, imgsz_tile=size, conf=conf, personas=len(b),
                         tiempo_ms=round(float(np.median(times)), 2),
                         tiempo_min_ms=round(min(times), 2), tiempo_max_ms=round(max(times), 2),
                         conf_media=float(b[:, 4].mean()) if len(b) else None,
                         cajas_conf_menor_015=int((b[:, 4] < .15).sum()),
                         **stats, observaciones=observaciones))

    for size, conf in ((1280, .15), (1280, .10), (1280, .12), (1536, .15), (1536, .10)):
        b, times = medir(lambda: predict(img, size, conf))
        notes = {
            (1280, .15): 'Referencia; numerosas omisiones en fondo, cajas mayormente sobre personas visibles',
            (1280, .10): '22 cajas adicionales frente a referencia; algunas parciales/ambiguas',
            (1280, .12): '13 cajas adicionales frente a referencia; compromiso intermedio sin mas resolucion',
            (1536, .15): 'Recomendada tras inspeccion; mayor cobertura zona media, persisten posibles duplicados',
            (1536, .10): '15 cajas debiles adicionales frente a 1536/0.15; algunas parciales y ambiguas',
        }
        guardar(f'full_{size}_{int(conf*100):03d}', size, conf, b, times,
                dict(candidatos=len(b), desc_geometria=0, desc_centro=0, desc_truncadas=0,
                     tras_nms_sin_filtros=len(b), antes_nms=len(b)),
                notes[(size,conf)])

    tile, overlap = 768, .25
    step = int(tile * (1-overlap))  # 576: solapamiento fijo de 192 píxeles
    margin = (tile-step)/2  # 96: mitad del solapamiento, no un margen arbitrario

    def starts(length):
        # El último tile se completa con gris; no se desplaza, preservando overlap=25%.
        return list(range(0, max(length-tile, 0)+step, step))

    def global_nms(b):
        if not len(b):
            return b
        return b[nms(torch.from_numpy(b[:, :4].copy()), torch.from_numpy(b[:, 4].copy()), .50).numpy()]

    def slicing(conf):
        raw, filtered = [], []
        stats = dict(candidatos=0, desc_geometria=0, desc_centro=0, desc_truncadas=0)
        for y in starts(h):
            for x in starts(w):
                crop = img[y:min(y+tile,h), x:min(x+tile,w)]
                ch, cw = crop.shape[:2]
                padded = cv2.copyMakeBorder(crop, 0, tile-ch, 0, tile-cw,
                                           cv2.BORDER_CONSTANT, value=(114,114,114))
                b = predict(padded, tile, conf)
                cx, cy = (b[:, 0]+b[:, 2])/2, (b[:, 1]+b[:, 3])/2
                # Excluir predicciones cuyo centro cae en padding, antes de cualquier filtro.
                b = b[(cx < cw) & (cy < ch)]
                b[:, [0,2]] = b[:, [0,2]].clip(0,cw)
                b[:, [1,3]] = b[:, [1,3]].clip(0,ch)
                stats['candidatos'] += len(b)
                translated = b.copy(); translated[:, [0,2]] += x; translated[:, [1,3]] += y
                raw.append(translated)

                bw, bh = b[:,2]-b[:,0], b[:,3]-b[:,1]
                # Guardia mínima: <=2 px no permite inspección corporal; NO usar un
                # mínimo grande, pues eliminaría personas pequeñas. Aspecto 1:20 es
                # una hipótesis conservadora de fragmento, no un criterio de verdad.
                geometry = (bw > 2) & (bh > 2) & (bw / np.maximum(bh,1) >= .05)
                stats['desc_geometria'] += int((~geometry).sum())
                cx, cy = (b[:,0]+b[:,2])/2, (b[:,1]+b[:,3])/2
                # Centro en zona interior: mitad de overlap a cada lado interno;
                # los bordes físicos de la fotografía no se penalizan.
                center = ((cx >= margin) | (x == 0)) & ((cy >= margin) | (y == 0))
                center &= ((cx < tile-margin) | (x+tile >= w))
                center &= ((cy < tile-margin) | (y+tile >= h))
                stats['desc_centro'] += int((geometry & ~center).sum())
                # Cajas que tocan un corte interno (1 px de tolerancia de redondeo)
                # son sospechosas de truncamiento. Puede perder cuerpos altos: se
                # documenta y compara con NMS sin estos filtros, no se afirma mejora.
                trunc = (((b[:,0] <= 1) & (x > 0)) | ((b[:,1] <= 1) & (y > 0)) |
                         ((b[:,2] >= cw-1) & (x+tile < w)) |
                         ((b[:,3] >= ch-1) & (y+tile < h)))
                stats['desc_truncadas'] += int((geometry & center & trunc).sum())
                filtered.append(translated[geometry & center & ~trunc])
        raw, b = np.concatenate(raw), np.concatenate(filtered)
        stats['tras_nms_sin_filtros'] = len(global_nms(raw))
        stats['antes_nms'] = len(b)
        return global_nms(b), stats

    for conf in (.10, .12, .15):
        (b, stats), times = medir(lambda: slicing(conf))
        guardar(f'tiles_768_{int(conf*100):03d}', tile, conf, b, times, stats,
                'Mejora cobertura del fondo respecto a full; persisten fragmentos/posibles duplicados; filtros pueden omitir personas validas')

    with (out/'refinamiento.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez(out/'predicciones_refinamiento.npz', **predictions)
    (out/'refinamiento_protocolo.json').write_text(json.dumps(dict(
        tile=tile, overlap=overlap, step=step, margen_centro=margin,
        tiles_xy=[(x,y) for y in starts(h) for x in starts(w)],
        iou_local=.50, iou_global=.50, tolerancia_borde_px=1,
        min_dimension_exclusiva_px=2, min_aspecto_ancho_alto=.05,
        max_det_por_llamada=300, repeticiones=3, device=str(device)),indent=2),encoding='utf-8')

    def draw(b):
        canvas = img.copy()
        for idx, box in enumerate(b):
            x1,y1,x2,y2 = np.rint(box[:4]).astype(int)
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,230,0),2)
            cv2.putText(canvas,f'{idx}:{box[4]:.2f}',(x1,max(12,y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    for name,b in predictions.items():
        cv2.imwrite(str(out/f'refinado_{name}.jpg'),cv2.cvtColor(draw(b),cv2.COLOR_RGB2BGR))
    # Galería de diagnóstico; elección final documentada después de inspección, no argmax(conteo).
    names = ['full_1280_015','full_1280_012','full_1536_015','full_1536_010','tiles_768_012']
    fig, axes = plt.subplots(1,len(names),figsize=(25,5))
    for ax,name in zip(axes,names):
        ax.imshow(draw(predictions[name])); ax.axis('off')
        ax.set_title(f'{name}\nn={len(predictions[name])}',fontsize=9)
    fig.tight_layout(); fig.savefig(out/'refinamiento_comparacion.png',dpi=150)
    plt.show(); plt.close(fig)
    # Selección por inspección visual documentada, NO por máximo número de cajas.
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    for ax,name in zip(axes,['full_1280_015','full_1536_015']):
        ax.imshow(draw(predictions[name])); ax.axis('off')
        ax.set_title(f'{name}: {len(predictions[name])} cajas de persona')
    fig.tight_layout(); fig.savefig(out/'refinamiento_seleccion_final.png',dpi=150)
    plt.show(); plt.close(fig)
    return rows, predictions
