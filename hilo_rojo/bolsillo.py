"""El revés de la arpillera: el bolsillo con la carta (para el llamado a leer la guía, en la web).

Las arpilleras de los talleres de la Vicaría de la Solidaridad llevaban, cosido al revés, un bolsillo
con la carta de quien las hizo. Aquí, al dar vuelta la tela, el estarcido del saco se lee al derecho
(HARINERA · LA ESPERANZA) y en el bolsillo va la carta: la guía.

Uso:
    python -m hilo_rojo.bolsillo web/hilvan/bolsillo.webp
"""
import sys

import cv2
import numpy as np

from satc_intro.color import lin, linear_to_srgb
from satc_intro.noise import fbm, smooth_noise
from .figures import hershey_strokes
from .pieces import Piece
from .territory import stencil_mask
from .textile import burlap, fray_mask
from .thread import Sprite, Yarn, blanket_stitch, composite, knot, running_stitch


def _paper(shape, rng):
    """Hoja de carta: papel roneo crema, fibras, una doblez."""
    h, w = shape
    base = lin("#efe6cf")
    tone = 1 + 0.05 * (fbm(shape, 90, rng, octaves=4) - 0.5) + 0.02 * (rng.random(shape).astype(np.float32) - 0.5)
    rgb = base[None, None, :] * tone[..., None]
    # doblez horizontal (la carta estuvo doblada en tres)
    yy = np.arange(h, dtype=np.float32)[:, None]
    for fy in (0.36,):
        d = (yy - h * fy) / 3.0
        crease = np.exp(-d * d) * 0.10
        rgb *= (1 - crease + 0.06 * np.exp(-((yy - h * fy - 5) / 4) ** 2))[..., None]
    return np.clip(rgb, 0, 1).astype(np.float32)


def _ink_strokes(canvas, strokes, col, width, ss=3):
    """Tinta de pluma: trazos finos con leve corrido."""
    H, W = canvas.shape[:2]
    m = np.zeros((H * ss, W * ss), np.float32)
    for st in strokes:
        pts = np.round(np.asarray(st) * ss * 8).astype(np.int32)
        cv2.polylines(m, [pts], False, 1.0, max(1, int(width * ss)), cv2.LINE_AA, shift=3)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
    m = np.clip(cv2.GaussianBlur(m, (0, 0), 0.5) * 1.1, 0, 1)
    c = lin(col)
    canvas *= (1 - 0.88 * m)[..., None]
    canvas += c[None, None, :] * 0.88 * m[..., None]


def render(W=1400, H=1000, seed=77):
    rng = np.random.default_rng(seed)
    u = min(W, H) / 1080
    col, hgt, _ = burlap(W, H, seed=seed)
    # el estarcido del saco, al derecho (tinta azul desteñida, gastada)
    ink = np.fliplr(stencil_mask(W, H, u * 1.25, y0=0.07))
    tinta = lin("#34476a")
    col = col * (1 - 0.62 * ink[..., None]) + tinta[None, None, :] * 0.62 * ink[..., None]
    canvas = col.copy()
    # el revés de las costuras del frente: nudos y cabos sueltos
    for _ in range(14):
        p = np.array([rng.uniform(0.08, 0.92) * W, rng.uniform(0.08, 0.95) * H])
        c = ["#3a2a1c", "#8e160f", "#c98f2c", "#2a2019"][rng.integers(0, 4)]
        composite(canvas, knot(p, rng.uniform(3.5, 5.5) * u, c, rng), 0.45)
        tail = [p, p + rng.normal(0, 14, 2) * u, p + rng.normal(0, 26, 2) * u]
        Yarn(np.array(tail), 2.0 * u, c, rng, fuzz=0, kind="floss").draw(canvas, 1e9, shadow=0.4)
    # la carta (la parte de abajo queda dentro del bolsillo)
    pw, ph = 0.34 * W, 0.46 * H
    cx, cy = 0.52 * W, 0.47 * H
    ang = np.deg2rad(-5.0)
    lw, lh = int(pw), int(ph)
    paper = _paper((lh, lw), rng)
    lines = [("Querida comunidad:", 36), ("esto es lo que aprendimos", 27), ("para no llegar tarde.", 27)]
    y = 0.16 * lh
    for txt, hh in lines:
        strokes, _ = hershey_strokes(txt, "scripts", hh * u * 0.95, 0.09 * lw, y + hh * u, anchor="left", ref="H")
        _ink_strokes(paper, strokes, "#253046", 1.5 * u)
        y += hh * u * 1.7
    # renglones de una carta larga (garabato leve)
    for k in range(3):
        yl = y + k * 30 * u
        xs = np.linspace(0.09 * lw, (0.85 - 0.25 * (k == 2)) * lw, 90)
        ys = yl + 2.2 * u * np.sin(xs / (7 * u) + k) * (0.6 + 0.4 * rng.random(90))
        _ink_strokes(paper, [np.stack([xs, ys], 1)], "#253046", 1.1 * u)
    # rotar la carta y ponerla en el lienzo
    pad = 40
    big = cv2.copyMakeBorder(paper, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    a0 = np.zeros((lh + 2 * pad, lw + 2 * pad), np.float32)
    a0[pad:pad + lh, pad:pad + lw] = 1
    M = cv2.getRotationMatrix2D(((lw + 2 * pad) / 2, (lh + 2 * pad) / 2), float(np.rad2deg(ang)), 1.0)
    prgb = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), flags=cv2.INTER_CUBIC)
    pa = cv2.warpAffine(a0, M, (big.shape[1], big.shape[0]), flags=cv2.INTER_LINEAR)
    x0, y0 = int(cx - big.shape[1] / 2), int(cy - big.shape[0] / 2)
    sh = cv2.GaussianBlur(pa, (0, 0), 6 * u)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, 8 * u], [0, 1, 9 * u]]), (sh.shape[1], sh.shape[0]))
    composite(canvas, Sprite(x0, y0, (prgb * pa[..., None]).astype(np.float32), pa.astype(np.float32), sh), 0.45)
    # el bolsillo: tocuyo cosido con festón de lana roja (abierto arriba)
    bx0, bx1 = 0.27 * W, 0.77 * W
    by0, by1 = 0.52 * H, 0.90 * H
    poly = np.array([(bx0, by0 + 6 * u), (bx1, by0 - 4 * u), (bx1 + 4 * u, by1), (bx0 - 3 * u, by1 + 5 * u)])
    pocket = Piece(poly, "#e6dcc2", rng, kind="plain", stitch=None, fabric_scale=u * 1.3, puff=1.4, margin=14)
    pocket.draw(canvas, 1.0, force=True)
    # dobladillo de la boca del bolsillo
    hem = np.array([(bx0 + 6 * u, by0 + 20 * u), (bx1 - 6 * u, by0 + 10 * u)])
    for _, sp in running_stitch(hem, 11 * u, 7 * u, 2.2 * u, "#8a7b62", rng, jitter=0.4):
        composite(canvas, sp, 0.4)
    side = [poly[0] + (0, 4 * u), poly[3], poly[2], poly[1] + (0, 4 * u)]
    for _, sp in blanket_stitch(np.array(side), 16 * u, 18 * u, 4.4 * u, "#c3241c", rng, inward=1):
        composite(canvas, sp, 0.5)
    # borde del saco deshilachado y sombra sobre la pared
    edge = [(0.02 * W, 0.03 * H), (0.985 * W, 0.02 * H), (0.975 * W, 0.975 * H), (0.015 * W, 0.97 * H)]
    m = fray_mask(edge, (H, W), rng, fray=2.4)
    m = cv2.GaussianBlur(m, (0, 0), 0.6)
    wall = lin("#e2d7c3")
    sh = cv2.GaussianBlur(m, (0, 0), 14 * u)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, 16 * u], [0, 1, 18 * u]]), (W, H))
    out = wall[None, None, :] * (1 - 0.38 * sh * (1 - m))[..., None] * np.ones((H, W, 1), np.float32)
    out = out * (1 - m[..., None]) + canvas * m[..., None]
    # luz rasante de la sala
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out *= (1 + 0.06 * (-(xx / W - 0.5) * 0.8 - (yy / H - 0.5) * 0.6))[..., None]
    return np.clip(out, 0, 1), dict(letter=(cx / W, cy / H), pocket=(bx0 / W, by0 / H, bx1 / W, by1 / H))


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "bolsillo.webp"
    img, info = render()
    srgb = (linear_to_srgb(img) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 82])
    print(out_path, info)
