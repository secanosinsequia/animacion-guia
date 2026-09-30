"""Hilos y puntadas en raster: cada puntada es un «tubo» de hilo con torsión, luz, brillo y sombra.

* Stitch: una puntada (el hilo por encima de la tela entre dos agujeros).
* Yarn: una lana continua echada sobre la tela (punto de «couching»), con torsión, pelusa y sombra;
  se puede revelar hasta cualquier largo (la aguja avanzando).
* Knot: nudo francés.
Todo se pre-renderiza una vez como sprites y luego solo se compone.
"""
import cv2
import numpy as np

from satc_intro.color import lin
from satc_intro.geometry import resample
from satc_intro.noise import smoothstep

LIGHT = np.array([-0.64, -0.56, 0.42])     # luz rasante (~26°)
LIGHT = LIGHT / np.linalg.norm(LIGHT)
SH_OFF = np.array([0.62, 0.54]) / 0.42 * 0.42   # dirección de la sombra (opuesta a la luz)


class Sprite:
    """rgb premultiplicado, alfa, sombra (alfa) y transmisión (qué deja pasar a contraluz, RGB)."""
    __slots__ = ("x0", "y0", "rgb", "a", "sh", "smap", "tr")

    def __init__(self, x0, y0, rgb, a, sh=None, smap=None, tr=None):
        self.x0, self.y0, self.rgb, self.a, self.sh, self.smap, self.tr = x0, y0, rgb, a, sh, smap, tr


def composite_T_piece(T, T0, C, sp, crisp=0.3, hem=True):
    """Transmisión de un retazo: la luz pasa por el yute (T0) y por la tela de más arriba; donde ya
    había otra tela (C), la costura superpuesta queda más oscura (como el plomo de un vitral)."""
    H, W = T.shape[:2]
    h, w = sp.a.shape
    x0, y0 = sp.x0, sp.y0
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    a = sp.a[sl][..., None]
    reg = T[cy0:cy1, cx0:cx1]
    c = C[cy0:cy1, cx0:cx1]
    seam = (1 - 0.5 * c)[..., None]
    # la tela difunde la luz: los huecos del yute se ven blandos a través de ella (no como un mosquitero)
    T0r = T0[cy0:cy1, cx0:cx1]
    T0b = getattr(composite_T_piece, "blur", None)
    under = T0r if T0b is None else crisp * T0r + (1 - crisp) * T0b[cy0:cy1, cx0:cx1]
    if hem:
        # el dobladillo: la orilla de cada retazo va doblada (doble tela, más oscura a contraluz)
        a2 = sp.a[sl]
        k = max(3, int(round(min(a2.shape) * 0.0 + 8)))
        inner = cv2.erode(a2, np.ones((k, k), np.uint8))
        seam = seam * (1 - 0.45 * np.clip(a2 - inner, 0, 1))[..., None]
    reg[:] = reg * (1 - a) + a * sp.tr[sl] * under * seam
    c[:] = np.maximum(c, sp.a[sl])


def composite_T(T, sp, reveal=None, dx=0, dy=0):
    """Transmisión: la luz de atrás que queda tras este sprite (opaco salvo que tenga `tr`)."""
    H, W = T.shape[:2]
    h, w = sp.a.shape
    x0, y0 = sp.x0 + dx, sp.y0 + dy
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    a = sp.a[sl]
    if reveal is not None:
        a = a * reveal[sl]
    reg = T[cy0:cy1, cx0:cx1]
    if sp.tr is None:
        reg *= (1 - 0.96 * a)[..., None]
    else:
        reg *= ((1 - a)[..., None] + a[..., None] * sp.tr[sl])


def composite(canvas, sp, shadow=0.5, alpha_mul=1.0, reveal=None, dx=0, dy=0):
    """Compone un sprite (rgb premultiplicado) con su sombra. reveal: máscara extra (misma forma)."""
    H, W = canvas.shape[:2]
    h, w = sp.a.shape
    x0, y0 = sp.x0 + dx, sp.y0 + dy
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    reg = canvas[cy0:cy1, cx0:cx1]
    a = sp.a[sl] * alpha_mul
    rgb = sp.rgb[sl] * alpha_mul
    if reveal is not None:
        rv = reveal[sl]
        a = a * rv
        rgb = rgb * rv[..., None]
    if sp.sh is not None and shadow > 0:
        s = sp.sh[sl] * alpha_mul
        if reveal is not None:
            s = s * rv
        reg *= (1 - shadow * s)[..., None]
    reg *= (1 - a)[..., None]
    reg += rgb


def _shade_tube(e, r, s_len, w, twist, strands, spec, fiber, nt=0.0):
    """Sombreado de un tubo de hilo. e: distancia con signo al eje; r: radio local."""
    dz = np.clip(e / np.maximum(r, 1e-3), -1, 1)
    nz = np.sqrt(np.clip(1 - dz * dz, 0, 1))
    return dz, nz


def _tube_rgb(color, e, r, n_vec, t_vec, s_px, w, twist_scale, strands, spec, fiber_noise, tilt=None,
              strand_contrast=0.30, spec_pow=18):
    dz = np.clip(e / np.maximum(r, 1e-3), -1, 1)
    nz = np.sqrt(np.clip(1 - dz * dz, 0, 1))
    nx = n_vec[0] * dz
    ny = n_vec[1] * dz
    if tilt is not None:
        nx = nx + t_vec[0] * tilt
        ny = ny + t_vec[1] * tilt
        nrm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        nx, ny, nz = nx / nrm, ny / nrm, nz / nrm
    lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    # torsión: surcos diagonales entre las hebras (cabos), y fibras finas dentro de cada cabo
    ph = s_px / (w * twist_scale) + dz * 0.55
    groove = 0.5 + 0.5 * np.cos(2 * np.pi * ph * strands)
    stripe = (1 - strand_contrast) + strand_contrast * smoothstep(0.10, 0.62, groove)
    fine = 0.92 + 0.08 * np.cos(2 * np.pi * (s_px / (w * 0.22) + dz * 2.6))
    shade = (0.34 + 0.86 * lam) * stripe * fine * fiber_noise
    # brillo especular (hilo mercerizado: brilla; lana: casi nada)
    hx, hy, hz = LIGHT[0], LIGHT[1], LIGHT[2] + 1.0
    hn = np.sqrt(hx * hx + hy * hy + hz * hz)
    sp_ = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0, 1) ** spec_pow * spec * smoothstep(0.3, 0.8, groove)
    rgb = color[None, None, :] * shade[..., None] + sp_[..., None]
    return rgb


class Stitch:
    """Una puntada de p0 a p1 (el hilo entra y sale de la tela en los extremos)."""

    @staticmethod
    def render(p0, p1, w, color, rng, kind="floss", hole=True, bow=None):
        color = lin(color) if isinstance(color, str) else np.asarray(color, np.float32)
        color = color * rng.uniform(0.94, 1.06)          # cada hebra con su tono
        p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
        d = p1 - p0
        L = float(np.hypot(*d))
        if L < 0.5:
            L = 0.5
            d = np.array([0.5, 0.0])
        t = d / L
        n = np.array([-t[1], t[0]])
        # una puntada larga no es una regla: se comba 1–2 px cada 100 px
        if bow is None:
            bow = (rng.uniform(0.008, 0.02) * L * rng.choice([-1, 1])) if L > 12 * w else 0.0
        pad = w * 1.6 + 3 + abs(bow)
        x0 = int(np.floor(min(p0[0], p1[0]) - pad))
        y0 = int(np.floor(min(p0[1], p1[1]) - pad))
        x1 = int(np.ceil(max(p0[0], p1[0]) + pad + w * 0.6))
        y1 = int(np.ceil(max(p0[1], p1[1]) + pad + w * 0.6))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
        qx, qy = xx - p0[0], yy - p0[1]
        s_px = qx * t[0] + qy * t[1]
        e = qx * n[0] + qy * n[1]
        s = s_px / L
        if bow:
            e = e - bow * 4 * np.clip(s, 0, 1) * (1 - np.clip(s, 0, 1))
        end = np.minimum(s_px, L - s_px)
        r = (w / 2) * np.sqrt(np.clip(end / (w * 0.55) + 0.15, 0, 1))
        r = np.where((s_px < -0.2) | (s_px > L + 0.2), 0, r)
        a = np.clip(r - np.abs(e) + 0.5, 0, 1).astype(np.float32)
        tilt = np.clip((0.25 - s) / 0.25, 0, 1) * -0.7 + np.clip((s - 0.75) / 0.25, 0, 1) * 0.7
        fib = 1 + 0.07 * (rng.random(a.shape).astype(np.float32) - 0.5)
        strands = 1.0
        spec = {"wool": 0.03, "synthetic": 0.95}.get(kind, 0.30)
        rgb = _tube_rgb(color, e, r, n, t, s_px, w, 1.05 if kind != "wool" else 1.25, strands, spec, fib, tilt,
                        strand_contrast={"wool": 0.45, "synthetic": 0.10}.get(kind, 0.30),
                        spec_pow=40 if kind == "synthetic" else 18)
        # agujeros donde el hilo entra: la tela se hunde un poco
        sh = cv2.GaussianBlur(a, (0, 0), max(0.6, w * 0.35))
        M = np.float32([[1, 0, w * 0.55], [0, 1, w * 0.50]])
        sh = cv2.warpAffine(sh, M, (sh.shape[1], sh.shape[0]))
        if hole:
            for p in (p0, p1):
                hx, hy = p[0] - x0, p[1] - y0
                hole_m = np.exp(-(((xx - x0 - 0.5 - hx) ** 2 + (yy - y0 - 0.5 - hy) ** 2) / (w * 0.45) ** 2))
                sh = np.maximum(sh, 0.8 * hole_m.astype(np.float32))
        return Sprite(x0, y0, (rgb * a[..., None]).astype(np.float32), a, np.clip(sh, 0, 1).astype(np.float32))


class Yarn:
    """Lana continua echada a lo largo de un camino, revelable hasta cualquier largo."""

    def __init__(self, path, w, color, rng, step=None, chunk=48, fuzz=1.0, couch_every=None, couch_color=None,
                 kind="wool"):
        self.color = lin(color) if isinstance(color, str) else np.asarray(color, np.float32)
        self.w = w
        pts = resample(np.asarray(path, np.float64), step or max(1.0, w * 0.5))
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.arc = np.concatenate([[0], np.cumsum(seg)])
        self.pts = pts
        self.length = float(self.arc[-1])
        self.chunks = []
        n = len(pts)
        i = 0
        while i < n - 1:
            j = min(n - 1, i + chunk)
            self.chunks.append(self._render_chunk(i, j, rng, fuzz, kind))
            i = j
        # puntadas de fijación (couching) cada cierto largo
        self.couch = []
        if couch_every:
            cc = lin(couch_color) if isinstance(couch_color, str) else (couch_color if couch_color is not None
                                                                         else self.color * 0.7)
            s = couch_every * 0.5
            while s < self.length:
                k = int(np.searchsorted(self.arc, s))
                k = min(max(k, 1), n - 1)
                p = pts[k]
                tng = pts[k] - pts[k - 1]
                tng /= np.linalg.norm(tng) + 1e-9
                nrm = np.array([-tng[1], tng[0]])
                sp = Stitch.render(p - nrm * w * 0.75, p + nrm * w * 0.75, max(1.2, w * 0.28), cc, rng, kind="floss")
                self.couch.append((s, sp))
                s += couch_every

    def _render_chunk(self, i, j, rng, fuzz, kind):
        w = self.w
        P = self.pts[max(0, i - 1):j + 2]
        A = self.arc[max(0, i - 1):j + 2]
        pad = w * 1.4 + 4
        x0 = int(np.floor(P[:, 0].min() - pad))
        y0 = int(np.floor(P[:, 1].min() - pad))
        x1 = int(np.ceil(P[:, 0].max() + pad + w * 0.6))
        y1 = int(np.ceil(P[:, 1].max() + pad + w * 0.6))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
        best_d = np.full(xx.shape, np.inf, np.float32)
        best_s = np.zeros(xx.shape, np.float32)
        best_e = np.zeros(xx.shape, np.float32)
        best_nx = np.zeros(xx.shape, np.float32)
        best_ny = np.zeros(xx.shape, np.float32)
        for k in range(len(P) - 1):
            a0, a1 = P[k], P[k + 1]
            d = a1 - a0
            L = float(np.hypot(*d)) + 1e-9
            t = d / L
            nrm = np.array([-t[1], t[0]])
            qx, qy = xx - a0[0], yy - a0[1]
            proj = np.clip(qx * t[0] + qy * t[1], 0, L)
            cx, cy = a0[0] + t[0] * proj, a0[1] + t[1] * proj
            dist = np.hypot(xx - cx, yy - cy)
            better = dist < best_d
            best_d = np.where(better, dist, best_d)
            best_s = np.where(better, A[k] + proj, best_s)
            best_e = np.where(better, qx * nrm[0] + qy * nrm[1], best_e)
            best_nx = np.where(better, nrm[0], best_nx)
            best_ny = np.where(better, nrm[1], best_ny)
        r = w / 2 * (1 + 0.06 * np.sin(best_s / (w * 3.1)))          # la lana engrosa y adelgaza
        a = np.clip(r - best_d + 0.6, 0, 1).astype(np.float32)
        fib = 1 + 0.10 * (rng.random(a.shape).astype(np.float32) - 0.5)
        n_vec = (best_nx, best_ny)
        t_vec = (best_ny, -best_nx)
        rgb = _tube_rgb(self.color, best_e, r, n_vec, t_vec, best_s, w, 1.25 if kind == "wool" else 1.05,
                        1.0, 0.03 if kind == "wool" else 0.3, fib, strand_contrast=0.5 if kind == "wool" else 0.3)
        # pelusa de lana: fibras sueltas alrededor
        if fuzz > 0:
            fz = np.zeros(a.shape, np.float32)
            nf = int(len(P) * 5.0 * fuzz)
            for _ in range(nf):
                k = rng.integers(0, len(P))
                p = P[k] - np.array([x0, y0])
                side = rng.choice([-1, 1])
                kk = min(k + 1, len(P) - 1)
                tng = P[kk] - P[max(k - 1, 0)]
                tng /= np.linalg.norm(tng) + 1e-9
                nrm = np.array([-tng[1], tng[0]])
                base = p + nrm * side * w * rng.uniform(0.35, 0.55)
                ang = np.arctan2(nrm[1] * side, nrm[0] * side) + rng.normal(0, 0.7)
                Lf = rng.uniform(1.2, 3.8) * (w / 7)
                end = base + np.array([np.cos(ang), np.sin(ang)]) * Lf
                cv2.line(fz, (int(base[0] * 4), int(base[1] * 4)), (int(end[0] * 4), int(end[1] * 4)),
                         float(rng.uniform(0.25, 0.6)), 1, cv2.LINE_AA, shift=2)
            fz = np.clip(fz, 0, 1) * (1 - a)
            fcol = self.color * 1.15
            rgb = rgb * a[..., None] + fcol[None, None, :] * fz[..., None] * 0.9
            a = np.clip(a + fz * 0.7, 0, 1)
            # la pelusa comparte el largo de su punto más cercano (para revelar)
        else:
            rgb = rgb * a[..., None]
        sh = cv2.GaussianBlur(a, (0, 0), max(0.8, w * 0.35))
        M = np.float32([[1, 0, w * 0.60], [0, 1, w * 0.55]])
        sh = cv2.warpAffine(sh, M, (sh.shape[1], sh.shape[0]))
        smap = np.where(best_d < w * 1.3, best_s, np.inf).astype(np.float32)
        return Sprite(x0, y0, rgb.astype(np.float32), a, np.clip(sh, 0, 1).astype(np.float32), smap)

    def head(self, L):
        """Punto y tangente en el largo L del camino."""
        L = float(np.clip(L, 0, self.length))
        k = int(np.clip(np.searchsorted(self.arc, L), 1, len(self.pts) - 1))
        a0, a1 = self.pts[k - 1], self.pts[k]
        u = (L - self.arc[k - 1]) / max(1e-9, self.arc[k] - self.arc[k - 1])
        p = a0 + (a1 - a0) * u
        tng = (a1 - a0) / (np.linalg.norm(a1 - a0) + 1e-9)
        return p, tng

    def draw(self, canvas, L, shadow=0.5):
        """Dibuja la lana hasta el largo L (con punta redondeada)."""
        if L <= 0:
            return
        for sp in self.chunks:
            smin = float(np.min(sp.smap))
            if smin > L:
                break
            if np.max(sp.smap[np.isfinite(sp.smap)]) <= L:
                composite(canvas, sp, shadow)
            else:
                rv = np.clip((L - sp.smap) / 1.5 + 0.5, 0, 1).astype(np.float32)
                rv[~np.isfinite(sp.smap)] = 0
                # la punta: redonda (se permiten los píxeles a menos de un radio del extremo)
                p, _ = self.head(L)
                yy, xx = np.mgrid[sp.y0:sp.y0 + sp.a.shape[0], sp.x0:sp.x0 + sp.a.shape[1]].astype(np.float32) + 0.5
                cap = np.clip(self.w / 2 - np.hypot(xx - p[0], yy - p[1]) + 0.6, 0, 1)
                rv = np.maximum(rv, cap * (sp.smap < L + self.w))
                composite(canvas, sp, shadow, reveal=rv)
        for s, sp in self.couch:
            if s + self.w < L:
                composite(canvas, sp, shadow * 0.8)


def knot(center, rad, color, rng):
    """Nudo francés: un botoncito de hilo enrollado."""
    color = lin(color) if isinstance(color, str) else np.asarray(color, np.float32)
    cx, cy = center
    pad = rad * 2 + 3
    x0, y0 = int(cx - pad), int(cy - pad)
    x1, y1 = int(cx + pad), int(cy + pad)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    dx, dy = (xx - cx) / rad, (yy - cy) / rad
    rr = np.sqrt(dx * dx + dy * dy)
    a = np.clip((1 - rr) * rad + 0.5, 0, 1).astype(np.float32)
    nz = np.sqrt(np.clip(1 - rr * rr, 0, 1))
    lam = np.clip(dx * LIGHT[0] + dy * LIGHT[1] + nz * LIGHT[2], 0, 1)
    ang = np.arctan2(dy, dx)
    spiral = 0.5 + 0.5 * np.cos(ang * 3 + rr * 9)
    shade = (0.35 + 0.8 * lam) * (0.72 + 0.28 * spiral)
    rgb = color[None, None, :] * shade[..., None] * a[..., None]
    sh = cv2.GaussianBlur(a, (0, 0), rad * 0.4)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, rad * 0.35], [0, 1, rad * 0.45]]), (sh.shape[1], sh.shape[0]))
    return Sprite(x0, y0, rgb.astype(np.float32), a, sh)


def running_stitch(path, stitch, gap, w, color, rng, jitter=0.6, kind="floss", closed=False):
    """Puntada corrida (hilván) a lo largo de un camino: lista de (largo_en_camino, sprite)."""
    pts = np.asarray(path, np.float64)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    pts = resample(pts, 1.0)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    out = []
    s = rng.uniform(0, gap)
    while s + stitch < arc[-1]:
        a = int(np.searchsorted(arc, s))
        b = int(np.searchsorted(arc, s + stitch * rng.uniform(0.72, 1.28)))
        b = min(b, len(pts) - 1)
        p0 = pts[a] + rng.normal(0, jitter * 1.6, 2)
        p1 = pts[b] + rng.normal(0, jitter * 1.6, 2)
        out.append((s, Stitch.render(p0, p1, w * rng.uniform(0.88, 1.12), color, rng, kind=kind)))
        s += stitch + gap * rng.uniform(0.65, 1.4)
    return out


def backstitch(path, stitch, w, color, rng, jitter=0.35, kind="floss"):
    """Pespunte: puntadas seguidas (para letras bordadas)."""
    return running_stitch(path, stitch, 0.9, w, color, rng, jitter=jitter, kind=kind)


def satin_fill(poly, angle, spacing, w, color, rng):
    """Punto de relleno (satén): hilos paralelos que cruzan el polígono."""
    poly = np.asarray(poly, np.float64)
    c, s_ = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s_], [s_, c]])
    rp = poly @ R.T
    ys = np.arange(rp[:, 1].min() + spacing * 0.5, rp[:, 1].max(), spacing)
    out = []
    n = len(rp)
    for i, y in enumerate(ys):
        xs = []
        for j in range(n):
            a, b = rp[j], rp[(j + 1) % n]
            if (a[1] <= y < b[1]) or (b[1] <= y < a[1]):
                u = (y - a[1]) / (b[1] - a[1])
                xs.append(a[0] + u * (b[0] - a[0]))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            p0 = np.array([xs[k] - 0.5, y]) @ R
            p1 = np.array([xs[k + 1] + 0.5, y]) @ R
            out.append((i, Stitch.render(p0 + rng.normal(0, 0.2, 2), p1 + rng.normal(0, 0.2, 2), w, color, rng,
                                         hole=False)))
    return out


def blanket_stitch(path, spacing, leg, w, color, rng, inward=1):
    """Punto festón: la lana corre por el borde y baja en «patitas» perpendiculares."""
    pts = resample(np.asarray(path, np.float64), 1.0)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    out = []
    s = 0.0
    prev = None
    while s < arc[-1]:
        k = int(np.clip(np.searchsorted(arc, s), 1, len(pts) - 1))
        p = pts[k]
        tng = pts[k] - pts[k - 1]
        tng /= np.linalg.norm(tng) + 1e-9
        nrm = np.array([-tng[1], tng[0]]) * inward
        foot = p + nrm * leg * rng.uniform(0.8, 1.2) + tng * rng.normal(0, leg * 0.08)
        p = p + rng.normal(0, 0.9, 2)
        out.append((s, Stitch.render(p, foot + rng.normal(0, 0.8, 2), w, color, rng, kind="wool")))
        if prev is not None:
            out.append((s, Stitch.render(prev, p, w * rng.uniform(0.9, 1.1), color, rng, kind="wool", hole=False)))
        prev = p
        s += spacing * rng.uniform(0.75, 1.28)
    return out
