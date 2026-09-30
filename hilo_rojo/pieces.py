"""Retazos (aplicaciones de tela) cosidos sobre la arpillera, con relieve, sombra y puntada de borde.

Cada retazo se pre-renderiza como sprite: tela con su tejido, borde cortado a tijera con hilos
sueltos, «acolchado» de los bordes (la tela se abomba un poco), sombra de su espesor y la puntada
corrida que lo fija. En la animación se «posa» como en stop-motion: un cuadro levantado (más grande,
sombra lejana y difusa) y luego apoyado.
"""
import cv2
import numpy as np

from satc_intro.color import lin
from .textile import fabric, fray_mask, shade, transmission
from .thread import Sprite, composite, running_stitch

UFPS = 12.0   # cuadros únicos por segundo (stop-motion «en dos» a 24 fps)


def inset_poly(poly, d):
    """Encoge un polígono hacia su centroide (aprox.) para la puntada de borde."""
    poly = np.asarray(poly, np.float64)
    c = poly.mean(axis=0)
    v = poly - c
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return c + v * np.clip((n - d) / n, 0.2, 1.0)


class Piece:
    def __init__(self, poly, color, rng, kind="plain", color2=None, t_place=0.0, stitch=("#5a3b28", 8, 6, 1.9),
                 fabric_scale=1.0, angle=0.0, felt=None, puff=1.0, shadow=0.5, fray=1.0, inset=5.5, margin=10):
        poly = np.asarray(poly, np.float64)
        self.poly = poly
        self.t_place = t_place
        self.shadow_k = shadow
        x0 = int(np.floor(poly[:, 0].min() - margin))
        y0 = int(np.floor(poly[:, 1].min() - margin))
        x1 = int(np.ceil(poly[:, 0].max() + margin))
        y1 = int(np.ceil(poly[:, 1].max() + margin))
        shape = (y1 - y0, x1 - x0)
        felt = (kind == "felt") if felt is None else felt
        fab, fh = fabric(shape, rng, color, kind=kind, color2=color2, scale=fabric_scale, angle=angle)
        m = fray_mask(poly, shape, rng, offset=(x0, y0), fray=fray, felt=felt)
        # acolchado: los bordes de la tela se redondean (luz arriba-izquierda, sombra abajo-derecha)
        body = cv2.GaussianBlur(m, (0, 0), 2.4)
        pf = shade(body * 3.2 * puff, strength=1.0, ambient=0.72)
        pf = pf / max(1e-6, float(np.percentile(pf[m > 0.9], 60))) if (m > 0.9).any() else pf
        rgb = fab * np.clip(pf, 0.55, 1.25)[..., None]
        sh = cv2.GaussianBlur(m, (0, 0), 3.0)
        sh = cv2.warpAffine(sh, np.float32([[1, 0, 4.2], [0, 1, 3.7]]), (shape[1], shape[0]))
        sh = np.clip(sh - m * 0.0, 0, 1)
        rgb_p = rgb * m[..., None]
        a = m.copy()
        # puntada corrida que fija el retazo
        if stitch:
            scol, slen, sgap, sw = stitch
            loc = Sprite(0, 0, rgb_p, a, None)
            for _, sp in running_stitch(inset_poly(poly, inset), slen, sgap, sw, scol, rng, closed=True):
                sp.x0 -= x0
                sp.y0 -= y0
                _composite_premult(rgb_p, a, sp)
        tr = transmission(fab, fh, kind)
        if stitch:   # las puntadas del borde no dejan pasar luz
            tr = tr * (1 - np.clip((a - m) * 0, 0, 1))[..., None]
        self.sprite = Sprite(x0, y0, rgb_p.astype(np.float32), a.astype(np.float32), sh.astype(np.float32),
                             tr=tr.astype(np.float32))
        self.center = poly.mean(axis=0)

    def bake(self, canvas, T, T0=None, C=None):
        composite(canvas, self.sprite, shadow=self.shadow_k)
        from .thread import composite_T, composite_T_piece
        if T0 is not None and C is not None:
            composite_T_piece(T, T0, C, self.sprite)
        else:
            composite_T(T, self.sprite)

    def draw(self, canvas, t, lift_center=None, force=False):
        if t < self.t_place and not force:
            return
        k = (t - self.t_place) * UFPS
        if k < 1 and not force:
            # levantado: un poco más grande y girado, sombra lejana y difusa
            c = self.center if lift_center is None else lift_center
            sp = _lifted(self.sprite, c, 1.025, np.deg2rad(0.8))
            composite(canvas, sp, shadow=self.shadow_k * 0.8)
            return
        composite(canvas, self.sprite, shadow=self.shadow_k)


def _composite_premult(rgb, a, sp, shadow=0.5):
    """Compone un sprite sobre una capa premultiplicada local (para hornear puntadas en el retazo)."""
    H, W = a.shape
    h, w = sp.a.shape
    x0, y0 = sp.x0, sp.y0
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    reg_rgb = rgb[cy0:cy1, cx0:cx1]
    reg_a = a[cy0:cy1, cx0:cx1]
    if sp.sh is not None:
        reg_rgb *= (1 - shadow * sp.sh[sl])[..., None]
    sa = sp.a[sl]
    reg_rgb *= (1 - sa)[..., None]
    reg_rgb += sp.rgb[sl]
    reg_a *= (1 - sa)
    reg_a += sa


def _lifted(sp, c, scale, rot):
    h, w = sp.a.shape
    pad = int(max(h, w) * (scale - 1) * 0.6) + 12
    cx, cy = c[0] - sp.x0 + pad, c[1] - sp.y0 + pad
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(np.rad2deg(rot)), float(scale))
    big = (w + 2 * pad, h + 2 * pad)

    def warp(img):
        src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        return cv2.warpAffine(src, M, big, flags=cv2.INTER_LINEAR, borderValue=0)

    rgb = warp(sp.rgb)
    a = warp(sp.a)
    sh = cv2.GaussianBlur(a, (0, 0), 7)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, 9], [0, 1, 12]]), big)
    return Sprite(sp.x0 - pad, sp.y0 - pad, rgb, a, sh * 0.8)


def ellipse_poly(cx, cy, rx, ry, n=40, wobble=0.0, rng=None, rot=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + (rng.normal(0, wobble, n) if (rng is not None and wobble) else 0)
    x = rx * np.cos(a) * r
    y = ry * np.sin(a) * r
    c, s = np.cos(rot), np.sin(rot)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)
