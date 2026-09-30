"""Acuarela y gouache procedurales.

Técnica de lavado inspirada en el ensayo de Tyler Hobbs «A generative approach to simulating
watercolor paints» (polígonos deformados recursivamente y apilados con baja opacidad), más
oscurecimiento de bordes, granulación sobre la textura del papel, *backruns* (coliflores) y un
mapa de «tiempo de llegada» que permite animar el florecimiento húmedo del pigmento.
"""
import cv2
import numpy as np

from .color import lin
from .noise import fbm, smooth_noise, smoothstep


def deform(poly, depth, var, rng, vv=None):
    """Subdivide y desplaza aleatoriamente los puntos medios (deformación recursiva)."""
    pts = np.asarray(poly, np.float64)
    if vv is None:
        vv = np.full(len(pts), var, np.float64)
    for _ in range(depth):
        n = len(pts)
        a, b = pts, np.roll(pts, -1, axis=0)
        seg = np.linalg.norm(b - a, axis=1)
        mid = (a + b) / 2 + rng.standard_normal((n, 2)) * (seg * vv)[:, None] * 0.5
        new = np.empty((2 * n, 2))
        new[0::2], new[1::2] = a, mid
        nv = np.empty(2 * n)
        nv[0::2] = vv
        nv[1::2] = (vv + np.roll(vv, -1)) / 2 * rng.uniform(0.7, 1.3, n)
        pts, vv = new, nv
    return pts, vv


def _fill(poly, shape, offset):
    img = np.zeros(shape, np.uint8)
    p = np.round((poly - offset) * 16).astype(np.int32)
    cv2.fillPoly(img, [p], 255, lineType=cv2.LINE_AA, shift=4)
    return img.astype(np.float32) / 255.0


class Layer:
    """Capa pictórica con su propio mapa de densidad y su mapa de llegada (animación)."""

    def __init__(self, density, arrival, bbox, color, mode="glaze", strength=1.0,
                 t0=0.0, t1=1.0, soft=0.08, rim=0.6, wet_boost=0.35, tau=0.35, opacity=1.0,
                 ease=None, motion=None, occluder=None):
        self.D = density.astype(np.float32)
        self.A = arrival.astype(np.float32)
        self.bbox = bbox  # (x0, y0, x1, y1)
        self.color = lin(color) if isinstance(color, str) else np.asarray(color, np.float32)
        self.mode = mode
        self.strength = strength
        self.t0, self.t1 = t0, t1
        self.soft, self.rim, self.wet_boost, self.tau = soft, rim, wet_boost, tau
        self.opacity = opacity
        self.ease = ease or (lambda x: 1 - (1 - x) ** 2.2)
        self.motion = motion        # t -> (dx, dy): desplazamiento de la capa (p. ej. el sol que sube)
        self.occluder = occluder    # máscara HxW en [0,1]: 1 = oculto (detrás de las montañas)
        if mode == "glaze":
            # Transmitancia del pigmento: el color «a plena carga» sobre papel blanco.
            self.logT = np.log(np.clip(self.color, 0.02, 1.0)).astype(np.float32)

    def coverage(self, t):
        """Densidad efectiva en el tiempo t (None si aún no empieza)."""
        if t <= self.t0:
            return None
        p = np.clip((t - self.t0) / max(1e-6, self.t1 - self.t0), 0, 1)
        prog = self.ease(p)
        f = prog * (1 + self.soft)
        m = smoothstep(f, f - self.soft, self.A)
        if p >= 1.0 and t > self.t1 + 4 * self.tau:
            return self.D * m
        # Frente húmedo más oscuro que avanza, y pintura mojada algo más saturada que al secar.
        front = np.exp(-((self.A - f) / 0.035) ** 2) * (1 - prog) ** 0.6
        t_arr = self.t0 + np.clip(self.A, 0, 1) * (self.t1 - self.t0)
        wet = np.exp(-np.maximum(0.0, t - t_arr) / self.tau)
        return self.D * (m * (1 + self.wet_boost * wet) + self.rim * front * (self.D > 0.05))

    def apply(self, canvas, t):
        cov = self.coverage(t)
        if cov is None:
            return
        x0, y0, x1, y1 = self.bbox
        if self.motion is not None:
            dx, dy = self.motion(t)
            ix, iy = int(np.floor(dx)), int(np.floor(dy))
            fx, fy = dx - ix, dy - iy
            M = np.float32([[1, 0, fx], [0, 1, fy]])
            cov = cv2.warpAffine(cov, M, (cov.shape[1], cov.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            x0, x1, y0, y1 = x0 + ix, x1 + ix, y0 + iy, y1 + iy
            H, W = canvas.shape[:2]
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                return
            cov = cov[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            x0, y0, x1, y1 = cx0, cy0, cx1, cy1
        if self.occluder is not None:
            cov = cov * (1 - self.occluder[y0:y1, x0:x1])
        region = canvas[y0:y1, x0:x1]
        if self.mode == "glaze":
            region *= np.exp(cov[..., None] * (self.strength * self.logT)[None, None, :])
        else:  # gouache opaco
            a = np.clip(cov * self.opacity, 0, 1)[..., None]
            region *= (1 - a)
            region += a * self.color[None, None, :]


def _densify(poly, max_seg):
    """Subdivide aristas largas para que la deformación sea local."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / max_seg)))
        for j in range(k):
            out.append(a + (b - a) * j / k)
    return np.array(out)


def wash(poly, color, canvas_shape, paper_height, rng, *, mode="glaze", strength=1.0,
         layers=36, base_depth=5, base_var=0.35, layer_depth=3, layer_var=0.45,
         softness=0.55, flat=0.45, edge_dark=0.55, edge_width=2.5, granulation=0.35,
         pooling=0.25, blooms=0, texture=0.35, max_seg=None, var_fn=None,
         grade=None, soft_blur=0.0, mask=None,
         reveal="bloom", seed_pt=None, sweep_angle=0.0, reveal_noise=0.22,
         t0=0.0, t1=1.0, margin=40, **layer_kw):
    """Crea una capa de acuarela a partir de un polígono base.

    grade: (y_denso, y_claro, minimo) degrada la densidad verticalmente (lavado graduado).
    var_fn: función de los puntos -> varianza de deformación por vértice.
    mask: máscara global HxW (viñeta de la lámina) que multiplica la densidad.
    """
    H, W = canvas_shape
    poly = np.asarray(poly, np.float64)
    if max_seg:
        poly = _densify(poly, max_seg)
    vv0 = var_fn(poly) if var_fn is not None else None
    base, vv = deform(poly, base_depth, base_var, rng, vv=vv0)
    x0 = int(max(0, np.floor(base[:, 0].min()) - margin))
    y0 = int(max(0, np.floor(base[:, 1].min()) - margin))
    x1 = int(min(W, np.ceil(base[:, 0].max()) + margin))
    y1 = int(min(H, np.ceil(base[:, 1].max()) + margin))
    shape = (y1 - y0, x1 - x0)
    off = np.array([x0, y0], np.float64)

    acc = np.zeros(shape, np.float32)
    for _ in range(layers):
        p, _ = deform(base, layer_depth, layer_var, rng, vv=vv * rng.uniform(0.6, 1.1))
        m = _fill(p, shape, off)
        if texture > 0:
            tex = smooth_noise(shape, rng.uniform(18, 60), rng)
            m *= (1 - texture) + texture * smoothstep(0.3, 0.7, tex)
        acc += m
    D = acc / layers
    S = smoothstep(0.30, 0.55, D)  # «charco» de pintura: forma nítida
    rim = np.clip(S - cv2.GaussianBlur(S, (0, 0), edge_width), 0, 1)
    if soft_blur > 0:
        D = cv2.GaussianBlur(D, (0, 0), soft_blur)
        S = smoothstep(0.30, 0.55, D)
        rim = np.zeros_like(D)
    dens = softness * D + flat * S + edge_dark * rim * 2.2
    yy_, xx_ = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    if grade is not None:
        gy0, gy1, gmin = grade[:3]
        gpow = grade[3] if len(grade) > 3 else 1.0
        gn = (fbm(shape, 60, rng, octaves=3) - 0.5) * (gy1 - gy0) * 0.35
        u = np.clip((yy_ + gn - gy0) / (gy1 - gy0), 0, 1) ** gpow
        dens *= 1 - (1 - gmin) * u
    if mask is not None:
        dens *= mask[y0:y1, x0:x1]

    # Acumulación irregular de pigmento y granulación sobre los valles del papel.
    ph = paper_height[y0:y1, x0:x1]
    dens *= (1 - pooling) + pooling * 2 * fbm(shape, 140, rng, octaves=3)
    dens *= (1 - granulation * 0.5) + granulation * (1 - ph) + granulation * 0.35 * (rng.random(shape) - 0.5)

    # Backruns (coliflores): zonas más claras con un borde oscuro fractal.
    for _ in range(blooms):
        ys, xs = np.nonzero(S > 0.8)
        if len(xs) == 0:
            break
        i = rng.integers(len(xs))
        cx, cy = xs[i], ys[i]
        r = rng.uniform(25, 80)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        d += (fbm(shape, 12, rng, octaves=3) - 0.5) * r * 0.8
        inside = smoothstep(r, r - 3, d)
        ring = np.exp(-((d - r) / 2.2) ** 2)
        dens *= 1 - 0.45 * inside
        dens += 0.35 * ring * S

    dens = np.clip(dens, 0, 1.6).astype(np.float32)

    # Mapa de llegada para el florecimiento animado.
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    if reveal == "bloom":
        sx, sy = (seed_pt if seed_pt is not None else base.mean(axis=0)) - off
        A = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    elif reveal == "sweep":
        A = xx * np.cos(sweep_angle) + yy * np.sin(sweep_angle)
    else:  # "dab": manchas que aparecen en desorden
        A = smooth_noise(shape, 90, rng) * 1000.0
    n1 = fbm(shape, 90, rng, octaves=4) - 0.5
    support = dens > 0.02
    if support.any():
        lo, hi = A[support].min(), A[support].max()
        A = (A - lo) / max(1e-6, hi - lo)
    A = np.clip(A + reveal_noise * n1 * 2, -0.2, 1.2)
    if support.any():
        lo, hi = A[support].min(), A[support].max()
        A = (A - lo) / max(1e-6, hi - lo)

    return Layer(dens, A, (x0, y0, x1, y1), color, mode=mode, strength=strength,
                 t0=t0, t1=t1, **layer_kw)
