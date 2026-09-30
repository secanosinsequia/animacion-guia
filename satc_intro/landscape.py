"""El territorio al alba: capas de acuarela y gouache con perspectiva atmosférica.

Referencia visual: volcán nevado sobre un lago del sur de Chile (tipo Osorno / Llanquihue),
resuelto como lámina de guía de campo: capas de montaña que se disuelven en neblina (papel),
un sol rojo que asoma tras la ladera del volcán y un bosque nativo en primer plano.
"""
import cv2
import numpy as np

from .geometry import ridge, volcano_profile, close_down
from .noise import fbm, smoothstep
from .watercolor import wash


def plate_mask(W, H, rng, box):
    """Viñeta de la lámina: la pintura termina en bordes irregulares sobre el papel."""
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n1 = (fbm((H, W), 120, rng, octaves=5) - 0.5)
    n2 = (fbm((H, W), 30, rng, octaves=3) - 0.5)
    ragged = 70 * n1 + 18 * n2
    left = smoothstep(x0 - 40, x0 + 90, xx + ragged)
    right = smoothstep(x1 + 40, x1 - 90, xx - ragged)
    bottom = smoothstep(y1 + 30, y1 - 110, yy - ragged * 1.3)
    top = smoothstep(y0 - 30, y0 + 60, yy + ragged)
    return (left * right * bottom * top).astype(np.float32)


def disc(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def ellipse(cx, cy, rx, ry, n=64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def fill_mask(W, H, poly):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


def build(W, H, paper_height, seed=11, lay=None):
    """Construye las capas del territorio (aguadas de tinta + gouache blanco).

    `lay` es el diccionario de maquetación (ver scene.layout). Devuelve (capas, geometría).
    """
    rng = np.random.default_rng(seed)
    lay = lay or {}
    sx, sy = W / 1920.0, H / 1080.0

    def P(x, y):
        return (x * sx, y * sy)

    box = lay.get("plate_box", (150 * sx, 470 * sy, 1770 * sx, 950 * sy))
    V = plate_mask(W, H, rng, box)
    shape = (H, W)
    layers = []
    geo = {"mask": V}

    hy = lay.get("horizon", 690 * sy)             # línea de cerros lejanos
    tx, ty = lay.get("tower_base", P(1590, 640))  # cima del cerro de la torre
    my = lay.get("meadow_top", 772 * sy)          # borde superior del potrero

    cord = ridge(box[0] - 30 * sx, box[2] + 30 * sx, hy - 18 * sy, 20 * sy, rng, n=9, rough=0.52,
                 peaks=[(box[0] + 0.12 * (box[2] - box[0]), hy - 44 * sy),
                        (box[0] + 0.35 * (box[2] - box[0]), hy - 26 * sy),
                        (box[0] + 0.55 * (box[2] - box[0]), hy - 40 * sy)])
    # Cerro de la torre: loma redondeada con la cima en (tx, ty)
    hill = []
    hw = lay.get("hill_w") or 330 * sx
    hh = lay.get("hill_h") or 118 * sy
    for u in np.linspace(-1, 1, 60):
        x = tx + u * hw
        y = ty + (1 - np.cos(u * np.pi / 2) ** 1.6) * hh * (1.0 if u < 0 else 0.85)
        hill.append((x, y))
    hill = np.array(hill)
    hill[:, 1] += rng.normal(0, 0.6, len(hill))
    forest = ridge(box[0] - 20 * sx, tx - 250 * sx, my - 24 * sy, 8 * sy, rng, n=8, rough=0.75,
                   peaks=[(box[0] + 120 * sx, my - 44 * sy), (box[0] + 420 * sx, my - 30 * sy),
                          (box[0] + 760 * sx, my - 36 * sy)])
    meadow_line = ridge(box[0] - 40 * sx, box[2] + 40 * sx, my, 9 * sy, rng, n=9, rough=0.62)
    geo.update(cordillera=cord, hill=hill, forest=forest, meadow=meadow_line, tower_base=(tx, ty),
               horizon=hy, meadow_top=my, box=box)

    ridge_var = lambda pts: np.where(pts[:, 1] > pts[:, 1].min() + 0.9 * (pts[:, 1].max() - pts[:, 1].min()),
                                     0.55, 0.10)

    # 1. Resplandor del alba detrás del cerro de la torre (gouache muy suelto)
    layers.append(wash(ellipse(tx - 120 * sx, hy - 10 * sy, 760 * sx, 190 * sy), "#f6ead0", shape,
                       paper_height, rng, mode="gouache", opacity=0.55, softness=1.0, flat=0.0,
                       edge_dark=0.0, soft_blur=40 * sx, layers=22, base_var=0.3, layer_var=0.6,
                       granulation=0.25, texture=0.45, grade=(hy, hy - 300 * sy, 0.0, 1.2), mask=V))
    # 2. Cordillera lejana: aguada de tinta fría, que se disuelve en neblina
    layers.append(wash(close_down(cord, hy + 70 * sy), "#6b7575", shape, paper_height, rng, strength=0.42,
                       layers=26, max_seg=22, var_fn=ridge_var, base_var=0.2, layer_var=0.35,
                       edge_dark=0.6, grade=(hy - 40 * sy, hy + 60 * sy, 0.05, 0.9), mask=V))
    # 3. Neblina de gouache
    layers.append(wash(ellipse((box[0] + box[2]) / 2, hy + 28 * sy, 0.55 * (box[2] - box[0]), 30 * sy),
                       "#f3e6c6", shape, paper_height, rng, mode="gouache", opacity=0.42, soft_blur=20 * sx,
                       softness=1.0, flat=0.0, edge_dark=0.0, layers=14, texture=0.5, mask=V))
    # 4. Cerro de la torre: aguada sepia cálida con borde nítido arriba
    layers.append(wash(close_down(hill, my + 14 * sy), "#77695a", shape, paper_height, rng, strength=0.58,
                       layers=26, max_seg=16, var_fn=ridge_var, base_var=0.15, layer_var=0.3,
                       edge_dark=0.75, grade=(ty, ty + 170 * sy, 0.06, 0.9), mask=V))
    # 5. Potrero: aguada muy clara y cálida
    layers.append(wash(close_down(meadow_line, box[3] + 20 * sy), "#8b7657", shape, paper_height, rng,
                       strength=0.26, layers=22, max_seg=30, var_fn=ridge_var, base_var=0.2, layer_var=0.4,
                       softness=0.95, flat=0.15, edge_dark=0.1, grade=(my, box[3] + 20 * sy, 0.5, 1.0),
                       mask=V, pooling=0.45, soft_blur=3 * sx))
    # 6. Cercos vivos: 3 filas de árboles (atrás más chicos y claros), tamaños ±40 %, algunos álamos
    trees = []
    xend = tx - hw * 0.9
    for row, (dy, scale, tone0, n_gap) in enumerate([(-16, 0.6, 0.16, 2.2), (-6, 0.8, 0.26, 1.5), (4, 1.0, 0.42, 1.3)]):
        x = box[0] + rng.uniform(30, 90) * sx
        while x < xend:
            poplar = rng.random() < (0.08 if row < 2 else 0.2)
            k = scale * rng.uniform(0.6, 1.4)
            if poplar:
                rx, ry = rng.uniform(7, 10) * sx * k, rng.uniform(40, 60) * sy * k
            else:
                rx, ry = rng.uniform(15, 30) * sx * k, rng.uniform(14, 26) * sy * k
            base_y = my + dy * sy + rng.normal(0, 2) * sy
            cy = base_y - ry * 0.95
            a = np.linspace(0, 2 * np.pi, 22, endpoint=False)
            pts = np.stack([x + rx * np.cos(a), cy + ry * np.sin(a)], 1)
            pts[:, 1] = np.minimum(pts[:, 1], base_y)
            tone = tone0 * rng.uniform(0.75, 1.3)
            layers.append(wash(pts, "#4a4636", shape, paper_height, rng, strength=tone, layers=10, base_depth=3,
                               base_var=0.32 if not poplar else 0.12, layer_depth=2, layer_var=0.45,
                               edge_dark=0.3, softness=0.8, flat=0.3, granulation=0.75, texture=0.55,
                               dry=0.8, grade=(cy - ry, base_y + 4 * sy, 0.45, 1.0), mask=V, margin=12))
            if row == 2:
                trees.append((x, base_y, rx, ry, poplar))
            gap = rng.uniform(-0.5, n_gap) if rng.random() < 0.85 else rng.uniform(2.0, 4.5)
            x += rx * (1.1 + gap)
    geo["trees"] = trees
    return layers, geo


def ink_strokes(geo, W, H, seed=21, avoid=None):
    """Plumilla del paisaje (fija): crestas con líneas perdidas y encontradas, copas del bosque,
    matas de pasto en el potrero. `avoid`: función (x, y) -> True si hay que dejar libre."""
    from .ink import Stroke
    rng = np.random.default_rng(seed)
    sx = W / 1920.0
    u = sx
    out = []
    avoid = avoid or (lambda x, y: False)

    def partial(line, keep=0.7, w=1.4, seglen=(60, 180)):
        line = np.asarray(line)
        i = 0
        n = len(line)
        while i < n - 2:
            L = int(rng.uniform(*seglen) * u / max(1e-6, np.mean(np.linalg.norm(np.diff(line, axis=0), axis=1))))
            L = max(3, L)
            seg = line[i:i + L]
            if rng.random() < keep and len(seg) > 2 and not avoid(*seg[len(seg) // 2]):
                out.append(Stroke(seg, w * u * rng.uniform(0.8, 1.2), rng, pool=0.25, smooth=False))
            i += L + int(rng.uniform(2, 12))

    from .geometry import resample
    partial(resample(geo["cordillera"], 3), keep=0.55, w=1.0, seglen=(40, 140))
    hill = resample(geo["hill"], 3)
    bx0_, bx1_ = geo["box"][0], geo["box"][2]
    hill_c = hill[(hill[:, 0] > bx0_) & (hill[:, 0] < bx1_ - 10 * u)]
    partial(hill_c, keep=0.9, w=1.8, seglen=(120, 260))
    # Hilera de árboles: contorno festoneado parcial y tronco
    for (x, base_y, rx, ry, poplar) in geo.get("trees", []):
        if avoid(x, base_y - ry):
            continue
        cy = base_y - ry * 0.95
        if poplar:
            a = np.linspace(np.pi * 0.62, np.pi * 1.38, 9)
            arc = np.stack([x + rx * np.cos(a), cy + ry * np.sin(a) * 0.98], 1)
            out.append(Stroke(arc, 1.0 * u, rng, pool=0.2, smooth=True, taper=(0.3, 0.3)))
        else:
            a0 = rng.uniform(np.pi * 0.95, np.pi * 1.15)
            a = np.linspace(a0, a0 + np.pi * rng.uniform(0.6, 0.95), 9)
            r = np.array([rx, ry]) * (1 + 0.06 * np.sin(np.arange(9) * 2.3))[:, None]
            arc = np.stack([x + r[:, 0] * np.cos(a), cy + r[:, 1] * np.sin(a)], 1)
            out.append(Stroke(arc, 1.1 * u, rng, pool=0.2, smooth=True, taper=(0.25, 0.35)))
        out.append(Stroke([(x + rng.normal(0, 1), base_y - ry * 0.25), (x, base_y + 2 * u)], 1.0 * u, rng,
                          pool=0.1, smooth=False))
    # Árboles sueltos en la ladera del cerro (lejos: pequeños)
    tx, ty = geo["tower_base"]
    for k in range(7):
        px = tx + rng.uniform(-300, -120) * u if k < 4 else tx + rng.uniform(110, 280) * u
        py = np.interp(px, hill[:, 0], hill[:, 1]) + rng.uniform(4, 30) * u
        if avoid(px, py):
            continue
        r = rng.uniform(3.5, 6) * u
        a = np.linspace(np.pi * 1.0, np.pi * 2.0, 6)
        out.append(Stroke(np.stack([px + r * np.cos(a), py + r * np.sin(a)], 1), 1.0 * u, rng, pool=0.2,
                          smooth=False))
        out.append(Stroke([(px, py), (px, py + 5 * u)], 0.8 * u, rng, pool=0.1, smooth=False))

    # Pasto: matas de distinto tamaño, briznas sueltas, espigas; más denso y grande adelante
    box = geo["box"]
    my = geo["meadow_top"]
    for k in range(300):
        yy = my + (box[3] - my) * rng.random() ** 0.75
        xx = rng.uniform(box[0] + 40 * u, box[2] - 40 * u)
        if avoid(xx, yy):
            continue
        depth = (yy - my) / max(1.0, box[3] - my)
        kind = rng.random()
        hgt = (3 + 11 * depth) * u * rng.uniform(0.5, 1.6)
        if kind < 0.25:          # brizna suelta
            n = 1
        elif kind < 0.85:        # mata
            n = int(rng.integers(2, 6))
        else:                    # mata con espiga
            n = int(rng.integers(3, 7))
        dirn = rng.normal(0.2, 0.45)
        for j in range(n):
            lean = (dirn + rng.normal(0, 0.3)) * hgt
            hh = hgt * rng.uniform(0.6, 1.15)
            bx = xx + rng.normal(0, 2.0 + 2.5 * depth) * u
            out.append(Stroke([(bx, yy), (bx + lean * 0.35, yy - hh * 0.55), (bx + lean, yy - hh)],
                              (0.55 + 0.7 * depth) * u * rng.uniform(0.8, 1.25), rng, taper=(0.02, 0.85), pool=0.08))
        if kind >= 0.85:
            sx_, sy_ = xx + dirn * hgt * 1.1, yy - hgt * 1.25
            out.append(Stroke([(xx, yy), (sx_, sy_)], 0.6 * u, rng, taper=(0.05, 0.3), pool=0.05))
            for j in range(4):
                out.append(Stroke([(sx_ + rng.normal(0, 1.2) * u, sy_ + j * 1.8 * u),
                                   (sx_ + rng.normal(0, 1.2) * u + 1.5 * u, sy_ + j * 1.8 * u - 1.2 * u)],
                                  0.9 * u, rng, taper=(0.1, 0.2), pool=0.3))
    return out
