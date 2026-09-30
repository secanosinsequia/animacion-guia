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


def roughen(poly, rng, amp=1.8, wl=40.0, notch=0.02, step=4.0):
    """Contorno cortado a tijera: ondulación de 1–3 px (longitud de onda 20–60 px) y alguna muesca."""
    from satc_intro.geometry import resample
    poly = np.asarray(poly, np.float64)
    closed = np.vstack([poly, poly[:1]])
    pts = resample(closed, step)[:-1]
    n = len(pts)
    if n < 6:
        return poly
    # normal hacia afuera (aprox.) de cada punto
    tng = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tng /= np.linalg.norm(tng, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([tng[:, 1], -tng[:, 0]], 1)
    c = pts.mean(axis=0)
    if np.mean(np.sum((pts - c) * nrm, axis=1)) < 0:
        nrm = -nrm
    arc = np.arange(n) * step
    off = np.zeros(n)
    for _ in range(3):
        w_ = rng.uniform(0.5, 1.5) * wl
        off += rng.normal(0, amp / 1.7) * np.sin(2 * np.pi * arc / w_ + rng.uniform(0, 2 * np.pi))
    fine = rng.normal(0, amp * 0.35, n)
    fine = np.convolve(np.concatenate([fine[-3:], fine, fine[:3]]), np.ones(7) / 7, mode="same")[3:-3]
    off += fine
    # muescas: la tijera se devolvió
    for i in np.nonzero(rng.random(n) < notch)[0]:
        k = np.arange(-2, 3)
        off[(i + k) % n] -= amp * 1.1 * np.array([0.3, 0.8, 1.0, 0.8, 0.3])
    return pts + nrm * off[:, None]


def inset_poly(poly, d):
    """Encoge un polígono hacia su centroide (aprox.) para la puntada de borde."""
    poly = np.asarray(poly, np.float64)
    c = poly.mean(axis=0)
    v = poly - c
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return c + v * np.clip((n - d) / n, 0.2, 1.0)


class Piece:
    def __init__(self, poly, color, rng, kind="plain", color2=None, t_place=0.0, stitch=("#5a3b28", 8, 6, 1.9),
                 fabric_scale=1.0, angle=0.0, felt=None, puff=1.0, shadow=0.5, fray=1.0, inset=5.5, margin=10,
                 rough=1.0, wear=1.0, crisp=None):
        poly = np.asarray(poly, np.float64)
        self.kind = kind
        self.crisp = crisp
        if rough > 0:
            size = float(min(np.ptp(poly[:, 0]), np.ptp(poly[:, 1])))
            k = rough * float(np.clip(size / (140 * max(0.6, fabric_scale)), 0.2, 1.0))    # retazos chicos: tijera fina
            poly = roughen(poly, rng, amp=1.0 * k * max(0.6, fabric_scale), wl=48 * max(0.6, fabric_scale) * max(0.4, k))
        self.poly = poly
        self.t_place = t_place
        self.shadow_k = shadow * rng.uniform(0.72, 1.3)          # cada retazo tiene su grosor
        x0 = int(np.floor(poly[:, 0].min() - margin))
        y0 = int(np.floor(poly[:, 1].min() - margin))
        x1 = int(np.ceil(poly[:, 0].max() + margin))
        y1 = int(np.ceil(poly[:, 1].max() + margin))
        shape = (y1 - y0, x1 - x0)
        felt = (kind == "felt") if felt is None else felt
        fab, fh = fabric(shape, rng, color, kind=kind, color2=color2, scale=fabric_scale, angle=angle)
        if wear > 0:
            fab = _wear(fab, rng, wear, fabric_scale)
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
        self.seam = None
        if stitch:
            scol, slen, sgap, sw = stitch
            self.seam = (inset_poly(poly, inset), sw)       # por el revés la puntada corrida es continua
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
            # la tela difunde la luz: la trama del saco apenas se adivina a través de ella (solo se ve bajo la
            # tela delgada, como el cielo)
            crisp = self.crisp if self.crisp is not None else \
                dict(felt=0.0, cord=0.02, flannel=0.02, satin=0.08).get(self.kind, 0.04)
            composite_T_piece(T, T0, C, self.sprite, crisp=crisp)
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


def _wear(fab, rng, k, scale):
    """Ropa usada: tono propio (±4 %), un lado desteñido, motas de pilling y alguna mancha tenue."""
    from satc_intro.noise import smooth_noise
    h, w = fab.shape[:2]
    tone = 1 + rng.normal(0, 0.04) * k
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = ((xx / max(w, 1) - 0.5) * np.cos(ang) + (yy / max(h, 1) - 0.5) * np.sin(ang))
    fade = 1 + 0.07 * k * grad
    out = fab * (tone * fade)[..., None]
    # desteñido: el color se va hacia el gris claro donde más le dio el sol
    lum = out.mean(axis=-1, keepdims=True)
    wash = np.clip(0.10 * k * (grad + 0.5), 0, 0.12)[..., None]
    out = out * (1 - wash) + (lum * 0.9 + 0.08) * wash
    # pilling: bolitas de fibra
    n = int(h * w / (900 * max(0.5, scale) ** 2) * k)
    if n > 0:
        pil = np.zeros((h, w), np.float32)
        for _ in range(n):
            cv2.circle(pil, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), 1, float(rng.uniform(0.3, 0.7)), -1,
                       cv2.LINE_AA)
        pil = cv2.GaussianBlur(pil, (0, 0), 0.6)
        out = out * (1 - 0.25 * pil[..., None]) + (lum * 1.25 + 0.03) * 0.25 * pil[..., None]
    # una mancha tenue (agua, té) en algunos retazos
    if rng.random() < 0.35 * k:
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.15, 0.4) * max(w, h)
        d = np.hypot(xx - cx, yy - cy) / r
        ring = np.exp(-((d - 1) / 0.08) ** 2) * 0.05 + (d < 1) * 0.015
        out *= (1 - ring * (0.8 + 0.4 * smooth_noise((h, w), 12, rng)))[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


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
