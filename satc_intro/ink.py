"""Plumilla: trazos de presión variable con charcos de tinta, sombreado (hatching) y punteado.

Cada trazo se guarda como polilínea + perfil de grosor y se dibuja con Cairo (antialias de alta
calidad) de forma progresiva, como si la pluma lo estuviera trazando.
"""
import cairo
import numpy as np

from .geometry import resample, catmull_rom
from .noise import smoothstep


class Mask:
    """Lienzo de cobertura (alfa) en el que se dibuja con Cairo."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.surf = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
        self.ctx = cairo.Context(self.surf)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)
        self.ctx.set_fill_rule(cairo.FILL_RULE_WINDING)

    def clear(self):
        c = self.ctx
        c.save()
        c.set_operator(cairo.OPERATOR_CLEAR)
        c.paint()
        c.restore()

    def array(self):
        self.surf.flush()
        stride = self.surf.get_stride()
        buf = np.ndarray((self.H, stride), np.uint8, buffer=self.surf.get_data())
        return buf[:, :self.W].astype(np.float32) / 255.0

    def set_alpha(self, a):
        self.ctx.set_source_rgba(0, 0, 0, float(np.clip(a, 0, 1)))

    def fill_poly(self, pts, alpha=1.0):
        pts = np.asarray(pts)
        if len(pts) < 3:
            return
        c = self.ctx
        self.set_alpha(alpha)
        c.move_to(*pts[0])
        for p in pts[1:]:
            c.line_to(*p)
        c.close_path()
        c.fill()

    def dot(self, x, y, r, alpha=1.0):
        self.set_alpha(alpha)
        self.ctx.arc(float(x), float(y), float(max(r, 0.05)), 0, 2 * np.pi)
        self.ctx.fill()


class Stroke:
    """Trazo de plumilla. `t0`, `t1`: intervalo en que la pluma lo recorre."""

    def __init__(self, pts, width, rng=None, t0=0.0, t1=0.0, taper=(0.12, 0.2), pool=0.35,
                 jitter=0.18, alpha=1.0, smooth=True, step=1.4, min_w=0.25):
        rng = rng if rng is not None else np.random.default_rng(0)
        pts = np.asarray(pts, np.float64)
        if smooth and len(pts) > 2:
            pts = catmull_rom(pts, samples=8)
        self.pts = resample(pts, step)
        n = len(self.pts)
        s = np.linspace(0, 1, n)
        # Perfil de presión: entra y sale afinado, con un leve temblor y charcos en los extremos.
        tin, tout = taper
        prof = smoothstep(0, max(tin, 1e-3), s) * smoothstep(1, 1 - max(tout, 1e-3), s)
        prof = 0.25 + 0.75 * prof
        wob = np.interp(s, np.linspace(0, 1, 7), rng.uniform(1 - jitter, 1 + jitter, 7))
        pool_curve = 1 + pool * (np.exp(-(s / 0.025) ** 2) + np.exp(-((1 - s) / 0.03) ** 2))
        self.w = np.maximum(width * prof * wob * pool_curve, min_w)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.arc = np.concatenate([[0], np.cumsum(seg)])
        self.t0, self.t1 = t0, t1
        self.alpha = alpha
        self.pool = pool

    def progress(self, t):
        if self.t1 <= self.t0:
            return 1.0 if t >= self.t0 else 0.0
        u = np.clip((t - self.t0) / (self.t1 - self.t0), 0, 1)
        return 1 - (1 - u) ** 1.6  # la pluma desacelera al final

    def draw(self, mask, t, alpha_mul=1.0, offset=(0.0, 0.0), transform=None):
        p = self.progress(t)
        if p <= 0 or len(self.pts) < 2:
            return
        L = self.arc[-1] * p
        k = int(np.searchsorted(self.arc, L))
        k = max(2, min(k + 1, len(self.pts)))
        pts = self.pts[:k].copy()
        w = self.w[:k].copy()
        if p < 1:  # punta húmeda: la tinta se acumula donde está la pluma
            w[-1] *= 1.25
        if transform is not None:
            pts = transform(pts)
        pts = pts + np.asarray(offset)
        tang = np.gradient(pts, axis=0)
        nrm = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        tang /= nrm
        normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
        left = pts + normal * (w[:, None] / 2)
        right = pts - normal * (w[:, None] / 2)
        a = self.alpha * alpha_mul
        mask.fill_poly(np.vstack([left, right[::-1]]), a)
        mask.dot(pts[0, 0], pts[0, 1], w[0] / 2, a)
        mask.dot(pts[-1, 0], pts[-1, 1], w[-1] / 2, a)


def hatch(poly, angle, spacing, rng, width=1.0, t0=0.0, t1=0.0, jitter=0.25, gap=0.12,
          min_len=4.0, **kw):
    """Sombreado de líneas paralelas recortadas al polígono (intersección analítica)."""
    poly = np.asarray(poly, np.float64)
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    rp = poly @ R.T
    ymin, ymax = rp[:, 1].min(), rp[:, 1].max()
    strokes = []
    Rinv = R.T
    ys = np.arange(ymin + spacing * rng.uniform(0.2, 0.8), ymax, spacing)
    n = len(rp)
    for i, y in enumerate(ys):
        y = y + rng.normal(0, spacing * 0.12)
        xs = []
        for j in range(n):
            a, b = rp[j], rp[(j + 1) % n]
            if (a[1] <= y < b[1]) or (b[1] <= y < a[1]):
                u = (y - a[1]) / (b[1] - a[1])
                xs.append(a[0] + u * (b[0] - a[0]))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            x0, x1 = xs[k], xs[k + 1]
            Lh = x1 - x0
            if Lh < min_len:
                continue
            x0 += Lh * rng.uniform(0, gap)
            x1 -= Lh * rng.uniform(0, gap)
            tilt = rng.normal(0, jitter)
            p0 = np.array([x0, y - tilt * 2]) @ Rinv.T
            p1 = np.array([x1, y + tilt * 2]) @ Rinv.T
            mid = (p0 + p1) / 2 + rng.normal(0, 0.6, 2)
            frac = i / max(1, len(ys) - 1)
            strokes.append(Stroke([p0, mid, p1], width * rng.uniform(0.75, 1.15), rng,
                                  t0=t0 + (t1 - t0) * frac * 0.7, t1=t0 + (t1 - t0) * (frac * 0.7 + 0.3),
                                  taper=(0.2, 0.3), pool=0.15, **kw))
    return strokes


def stipple(poly_mask_fn, bbox, n, rng, r=(0.5, 1.4)):
    """Puntos de tinta dentro de una región (función de pertenencia)."""
    x0, y0, x1, y1 = bbox
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if poly_mask_fn(x, y):
            pts.append((x, y, rng.uniform(*r)))
    return pts
