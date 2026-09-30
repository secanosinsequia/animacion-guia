"""Geometría del paisaje: crestas fractales, conos volcánicos y utilidades de curvas."""
import numpy as np


def ridge(x0, x1, y, amp, rng, n=9, rough=0.55, peaks=None):
    """Línea de cresta por desplazamiento de punto medio (1D). Devuelve Nx2."""
    xs = np.array([x0, x1], np.float64)
    ys = np.array([y, y], np.float64)
    if peaks:
        for px, py in peaks:
            i = np.searchsorted(xs, px)
            xs = np.insert(xs, i, px)
            ys = np.insert(ys, i, py)
    a = amp
    for _ in range(n):
        mx = (xs[:-1] + xs[1:]) / 2
        my = (ys[:-1] + ys[1:]) / 2 + rng.normal(0, a, len(mx))
        nx = np.empty(len(xs) + len(mx))
        ny = np.empty_like(nx)
        nx[0::2], nx[1::2] = xs, mx
        ny[0::2], ny[1::2] = ys, my
        xs, ys = nx, ny
        a *= rough
    return np.stack([xs, ys], axis=1)


def volcano_profile(cx, base_y, peak_y, half_w, rng, crater=0.035, n=80, concave=1.45, skew=0.0):
    """Perfil de cono volcánico con laderas cóncavas y un cráter pequeño."""
    left, right = [], []
    for i in range(n + 1):
        u = i / n  # 0 en la base, 1 en la cima
        h = peak_y + (base_y - peak_y) * (1 - u) ** concave
        dx = half_w * (1 - u) + half_w * crater
        jitter = rng.normal(0, 1.2) * (1 - u) ** 0.5
        left.append((cx - dx * (1 - skew) + jitter, h))
        right.append((cx + dx * (1 + skew) + jitter, h))
    top = [(cx - half_w * crater * 0.6, peak_y + 2), (cx, peak_y + 5), (cx + half_w * crater * 0.6, peak_y + 1)]
    return np.array(left + top + right[::-1], np.float64)


def close_down(line, bottom_y):
    """Cierra una línea de cresta hacia abajo para formar un polígono relleno."""
    line = np.asarray(line, np.float64)
    return np.vstack([line, [[line[-1, 0], bottom_y], [line[0, 0], bottom_y]]])


def resample(pts, step=1.5):
    """Remuestrea una polilínea a paso constante."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return pts[:1]
    n = max(2, int(total / step) + 1)
    si = np.linspace(0, total, n)
    x = np.interp(si, s, pts[:, 0])
    y = np.interp(si, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def catmull_rom(pts, samples=12, closed=False):
    """Curva suave que pasa por los puntos de control."""
    p = np.asarray(pts, np.float64)
    if closed:
        p = np.vstack([p[-1], p, p[0], p[1]])
    else:
        p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-2])
    return np.array(out)
