"""Texturas textiles procedurales con relieve y luz: yute (arpillera), telas, fieltro, organza.

Todo es raster: cada tela es un campo de altura (hilos que pasan por arriba y por abajo de la trama)
más color, iluminado como una foto cenital con luz rasante suave. Nada es un relleno plano.
"""
import cv2
import numpy as np

from satc_intro.color import lin
from satc_intro.noise import fbm, smooth_noise

LIGHT = np.array([-0.64, -0.56, 0.42])     # luz rasante (~26°) desde arriba a la izquierda
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def shade(height, strength=1.0, ao=0.0, ambient=0.55):
    """Luz de un campo de altura (0..1): lambert + oclusión en los valles."""
    gy, gx = np.gradient(height.astype(np.float32))
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(gx)
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    lam = (nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]) / n
    lam = np.clip(lam, 0, 1)
    s = ambient + (1 - ambient) * lam / LIGHT[2]
    if ao:
        s *= (1 - ao) + ao * np.clip(height, 0, 1) ** 0.6
    return s.astype(np.float32)


def _streaks(shape, rng, along="x", length=40, width=1.2):
    """Fibras: ruido estirado en la dirección del hilo."""
    h, w = shape
    n = rng.standard_normal((h, w)).astype(np.float32)
    k = (int(length) | 1, 1) if along == "x" else (1, int(length) | 1)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=length / 4 if along == "x" else width,
                         sigmaY=width if along == "x" else length / 4)
    return n / (n.std() + 1e-6)


def weave(shape, rng, pitch=7.0, gap=0.28, wobble=1.6, slub=0.25, angle=0.0, jitter=0.12, irregular=0.0):
    """Tejido tafetán (plano). Devuelve (altura 0..1, id_hilo_urdimbre, id_hilo_trama, urdimbre_arriba).

    pitch: distancia media entre hilos (px). gap: fracción vacía entre hilos. wobble: ondulación de los
    hilos. irregular: 0..1, variación del espaciado y grosor de cada hilo (yute: alto; algodón: bajo).
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        xx, yy = xx * c + yy * s, -xx * s + yy * c
    wx = (smooth_noise(shape, 90, rng) - 0.5) * 2 * wobble * 2.5
    wy = (smooth_noise(shape, 90, rng) - 0.5) * 2 * wobble * 2.5

    def warp_coord(coord, n_lines):
        """Coordenada continua -> índice de hilo con espaciado irregular (cada hilo tiene su ancho)."""
        if irregular <= 0:
            return coord / pitch
        widths = pitch * np.clip(1 + irregular * rng.normal(0, 0.45, n_lines), 0.45, 1.9)
        edges = np.concatenate([[0], np.cumsum(widths)]) - pitch * 4
        i = np.clip(np.searchsorted(edges, coord) - 1, 0, n_lines - 1)
        return i + (coord - edges[i]) / widths[i]

    lo_x, hi_x = float(np.min(xx + wx)), float(np.max(xx + wx))
    lo_y, hi_y = float(np.min(yy + wy)), float(np.max(yy + wy))
    u = warp_coord(xx + wx - lo_x, int((hi_x - lo_x) / pitch * 2.5) + 20)
    v = warp_coord(yy + wy - lo_y, int((hi_y - lo_y) / pitch * 2.5) + 20)
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    # grosor por hilo (algunos más gruesos) y a lo largo del hilo («slubs»)
    per_u = rng.normal(0, 1, int(iu.max()) + 3)[iu.astype(np.int32)]
    per_v = rng.normal(0, 1, int(iv.max()) + 3)[iv.astype(np.int32)]
    along_u = smooth_noise(shape, pitch * 6, rng) - 0.5
    along_v = smooth_noise(shape, pitch * 6, rng) - 0.5
    thick_u = 1 - gap + slub * along_u + irregular * 0.12 * per_u
    thick_v = 1 - gap + slub * along_v + irregular * 0.12 * per_v
    du = np.abs(fu - 0.5) * 2 / np.clip(thick_u, 0.3, 1.25)
    dv = np.abs(fv - 0.5) * 2 / np.clip(thick_v, 0.3, 1.25)
    prof_u = np.sqrt(np.clip(1 - du * du, 0, 1))
    prof_v = np.sqrt(np.clip(1 - dv * dv, 0, 1))
    over = ((iu + iv) % 2 == 0)
    und_u = 0.775 + 0.225 * np.cos(np.pi * (v - iv - 0.5) * 2 + np.where(over, 0, np.pi))
    und_v = 0.775 + 0.225 * np.cos(np.pi * (u - iu - 0.5) * 2 + np.where(over, np.pi, 0))
    # torsión del hilo: rayitas diagonales a lo largo de cada hebra
    twist_u = 0.08 * np.sin((yy + wy) * 2.2 + fu * 5.0)
    twist_v = 0.08 * np.sin((xx + wx) * 2.2 + fv * 5.0)
    hu = prof_u * und_u * (1 + twist_u)
    hv = prof_v * und_v * (1 + twist_v)
    height = np.maximum(hu, hv)
    top_u = hu >= hv
    height += jitter * (rng.random(shape).astype(np.float32) - 0.5) * (height > 0.05)
    return np.clip(height, 0, 1).astype(np.float32), iu.astype(np.int32), iv.astype(np.int32), top_u


def burlap(W, H, seed=3, ss=2, stencil=None):
    """Arpillera de yute calculada al doble de resolución (sin moiré).

    Devuelve (rgb lineal, altura, transmisión RGB): la transmisión es cuánta luz pasa desde atrás
    (los huecos de la trama brillan; el yute deja pasar un poco, cálido). `stencil`: máscara HxW con el
    estarcido del saco harinero (tinta en el revés: se ve en espejo solo a contraluz).
    """
    rng = np.random.default_rng(seed)
    W2, H2 = W * ss, H * ss
    shape = (H2, W2)
    pitch = 6.6 * ss * max(W, H) / 1920
    hgt, iu, iv, top_u = weave(shape, rng, pitch=pitch, gap=0.26, wobble=2.4 * ss, slub=0.45, irregular=0.9)
    pal_u = rng.normal(0, 1, iu.max() - iu.min() + 3)
    pal_v = rng.normal(0, 1, iv.max() - iv.min() + 3)
    var = np.where(top_u, pal_u[iu - iu.min()], pal_v[iv - iv.min()]).astype(np.float32)
    fib = np.where(top_u, _streaks(shape, rng, "y", length=pitch * 4), _streaks(shape, rng, "x", length=pitch * 4))
    base = lin("#b99a6b")
    col = np.empty((H2, W2, 3), np.float32)
    tone = 1 + 0.10 * var + 0.10 * fib + 0.08 * (fbm(shape, 260 * ss, rng, octaves=3) - 0.5)
    col[..., 0] = base[0] * tone * (1 + 0.02 * var)
    col[..., 1] = base[1] * tone
    col[..., 2] = base[2] * tone * (1 - 0.03 * var)
    s = shade(hgt * 2.2 / ss, strength=1.3 * ss, ao=0.55, ambient=0.45)
    hole = np.clip(1 - hgt / 0.12, 0, 1)
    col = col * s[..., None] * (1 - hole[..., None] * 0.80)
    col += hole[..., None] * lin("#3a2a1a") * 0.22
    # transmisión: los huecos dejan pasar toda la luz; el yute, un poco y cálido; más delgado = más luz
    thin = 1 - np.clip(hgt, 0, 1)
    tr = hole * 1.0 + (1 - hole) * (0.10 + 0.22 * thin * (0.9 + 0.2 * (var * 0.1)))
    trc = np.stack([tr, tr * 0.86, tr * 0.62], -1).astype(np.float32)
    col = cv2.resize(col, (W, H), interpolation=cv2.INTER_AREA)
    hgt = cv2.resize(hgt, (W, H), interpolation=cv2.INTER_AREA)
    trc = cv2.resize(trc, (W, H), interpolation=cv2.INTER_AREA)
    # pelusa: fibras sueltas claras sobre la trama
    fuzz = np.zeros((H, W), np.float32)
    for _ in range(int(W * H / 420)):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        L = rng.uniform(4, 22) * max(W, H) / 1920
        a = rng.uniform(0, np.pi)
        cv2.line(fuzz, (int(x * 4), int(y * 4)), (int((x + L * np.cos(a)) * 4), int((y + L * np.sin(a)) * 4)),
                 float(rng.uniform(0.2, 0.6)), 1, cv2.LINE_AA, shift=2)
    col = col * (1 - 0.35 * fuzz[..., None]) + lin("#d9c29a")[None, None, :] * 0.35 * fuzz[..., None]
    if stencil is not None:     # tinta en el revés: apenas sugerida a contraluz
        trc *= (1 - 0.30 * stencil)[..., None]
    return np.clip(col, 0, 1).astype(np.float32), hgt.astype(np.float32), np.clip(trc, 0, 1.2).astype(np.float32)


def fabric(shape, rng, color, kind="plain", color2=None, pitch=2.6, scale=1.0, angle=0.0):
    """Tela para un retazo: devuelve (rgb lineal, altura) del tamaño `shape`.

    kind: plain (algodón), gingham (cuadrillé), dots (lunares), stripes (rayas), cord (pana),
    felt (fieltro), satin (raso), print (estampado de flores menudas).
    """
    h, w = shape
    base = lin(color) if isinstance(color, str) else np.asarray(color, np.float32)
    c2 = (lin(color2) if isinstance(color2, str) else np.asarray(color2, np.float32)) if color2 is not None else base * 0.6
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if angle:
        c, s_ = np.cos(angle), np.sin(angle)
        xr, yr = xx * c + yy * s_, -xx * s_ + yy * c
    else:
        xr, yr = xx, yy
    mix = np.zeros(shape, np.float32)
    if kind == "felt":
        hgt = 0.5 + 0.30 * (fbm(shape, 6 * scale, rng, octaves=3) - 0.5) + 0.15 * (rng.random(shape) - 0.5)
        hgt = cv2.GaussianBlur(hgt.astype(np.float32), (0, 0), 0.6)
        fib = np.zeros(shape, np.float32)
        for _ in range(int(h * w / 60)):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            L = rng.uniform(2, 7)
            a = rng.uniform(0, np.pi)
            cv2.line(fib, (int(x), int(y)), (int(x + L * np.cos(a)), int(y + L * np.sin(a))), float(rng.uniform(-1, 1)),
                     1, cv2.LINE_AA)
        hgt = np.clip(hgt + 0.12 * fib, 0, 1)
        tone = 1 + 0.10 * (fbm(shape, 30 * scale, rng, octaves=3) - 0.5) + 0.08 * fib
        col = base[None, None, :] * tone[..., None]
        s = shade(hgt, strength=1.2, ambient=0.72)
        return np.clip(col * s[..., None], 0, 1).astype(np.float32), hgt.astype(np.float32)
    if kind == "cord":
        rib = pitch * 3.2 * scale
        ph = (xr / rib) % 1.0
        hgt = np.sqrt(np.clip(1 - ((ph - 0.5) * 2.1) ** 2, 0, 1)) * (0.8 + 0.2 * smooth_noise(shape, 40, rng))
        fuzzn = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 0.7)
        hgt = np.clip(hgt + 0.06 * fuzzn, 0, 1)
        tone = 1 + 0.06 * (fbm(shape, 50, rng, octaves=2) - 0.5)
        col = base[None, None, :] * tone[..., None] * (0.85 + 0.25 * hgt[..., None])
        s = shade(hgt, strength=1.6, ao=0.35, ambient=0.6)
        return np.clip(col * s[..., None], 0, 1).astype(np.float32), hgt.astype(np.float32)
    hgt, iu, iv, top_u = weave(shape, rng, pitch=pitch * scale, gap=0.16, wobble=0.5, slub=0.12, angle=angle,
                               jitter=0.06)
    if kind == "gingham":
        cell = pitch * scale * 7
        a = ((xr // cell) % 2).astype(np.float32)
        b = ((yr // cell) % 2).astype(np.float32)
        mix = (a + b) / 2
    elif kind == "flannel":
        cell = pitch * scale * 9
        a = ((xr // cell) % 3 == 0).astype(np.float32)
        b = ((yr // cell) % 3 == 0).astype(np.float32)
        thin_a = ((xr // (cell / 3)) % 9 == 4).astype(np.float32) * 0.5
        mix = np.clip(0.55 * a + 0.55 * b + thin_a, 0, 1)
        hgt = np.clip(hgt * 0.6 + 0.4 * smooth_noise(shape, 3, rng), 0, 1)   # franela: perchada, suave
    elif kind == "stripes":
        cell = pitch * scale * 5
        mix = ((xr // cell) % 3 == 0).astype(np.float32)
    elif kind == "dots":
        cell = pitch * scale * 8
        gx = (xr / cell) % 1.0 - 0.5
        gy_ = ((yr / cell) + 0.5 * ((xr // cell) % 2)) % 1.0 - 0.5
        mix = (np.sqrt(gx * gx + gy_ * gy_) < 0.2).astype(np.float32)
    elif kind == "print":
        cell = pitch * scale * 11
        mix = np.zeros(shape, np.float32)
        r2 = np.random.default_rng(int(rng.integers(1 << 30)))
        for _ in range(int(h * w / (cell * cell) * 1.6)):
            cx, cy = r2.uniform(0, w), r2.uniform(0, h)
            rr = cell * r2.uniform(0.10, 0.16)
            for k in range(5):     # florcita de 5 pétalos
                a = k * 2 * np.pi / 5 + r2.uniform(0, 1)
                cv2.circle(mix, (int(cx + rr * 1.1 * np.cos(a)), int(cy + rr * 1.1 * np.sin(a))), max(1, int(rr * 0.7)),
                           1.0, -1, cv2.LINE_AA)
        mix = cv2.GaussianBlur(mix, (0, 0), 0.5)
    elif kind == "satin":
        hgt = 0.6 + 0.25 * np.sin(yr / (pitch * scale * 0.9)) * 0.15 + 0.1 * smooth_noise(shape, 30, rng)
        hgt = hgt.astype(np.float32)
        sheen = np.clip(smooth_noise(shape, 60, rng) * 1.6 - 0.4, 0, 1)
        col = base[None, None, :] * (0.9 + 0.35 * sheen[..., None])
        s = shade(hgt, strength=0.6, ambient=0.8)
        return np.clip(col * s[..., None], 0, 1).astype(np.float32), hgt
    # el estampado se imprime sobre la trama: se nota el hilo debajo
    thread_var = 1 + 0.05 * np.where(top_u, 1.0, -1.0)
    col = base[None, None, :] * (1 - mix[..., None]) + c2[None, None, :] * mix[..., None]
    col = col * thread_var[..., None] * (1 + 0.05 * (fbm(shape, 40, rng, octaves=2)[..., None] - 0.5))
    s = shade(hgt * 1.2, strength=0.9, ao=0.35, ambient=0.66)
    return np.clip(col * s[..., None], 0, 1).astype(np.float32), hgt


TRANS_K = dict(plain=0.60, gingham=0.52, dots=0.50, stripes=0.52, print=0.46, flannel=0.40, cord=0.16, felt=0.02,
               satin=0.42)


def transmission(rgb_lin, hgt, kind):
    """Luz que atraviesa un retazo: su color (saturado) por la densidad de su tejido."""
    k = TRANS_K.get(kind, 0.3)
    dens = 0.55 + 0.9 * (1 - np.clip(hgt, 0, 1)) if kind not in ("felt", "cord") else 1.0
    # a contraluz el tinte se satura (la luz atraviesa el teñido): color normalizado ** 1.8
    c = np.clip(rgb_lin, 0, 1)
    mx = np.maximum(c.max(axis=-1, keepdims=True), 1e-3)
    luma = (c * np.array([0.3, 0.55, 0.15], np.float32)).sum(-1, keepdims=True)
    tint = (c / mx) ** 1.8 * np.clip(luma, 0, 1) ** 0.35
    return (tint * (k * dens if np.ndim(dens) == 0 else k * dens[..., None])).astype(np.float32)


def fray_mask(poly, shape, rng, offset=(0, 0), fray=1.0, felt=False, supersample=2):
    """Máscara de un retazo cortado con tijeras: borde levemente irregular + hilos sueltos."""
    h, w = shape
    ss = supersample
    m = np.zeros((h * ss, w * ss), np.uint8)
    pts = (np.asarray(poly, np.float64) - np.asarray(offset)) * ss
    cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # borde irregular: desplazamiento de ruido fino
    dx = (smooth_noise(shape, 6, rng) - 0.5) * 2.2 * fray
    dy = (smooth_noise(shape, 6, rng) - 0.5) * 2.2 * fray
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = cv2.remap(m, gx + dx, gy + dy, cv2.INTER_LINEAR)
    if felt:   # el fieltro no se deshilacha: borde algo difuso y velloso
        edge = np.clip(m - cv2.GaussianBlur(m, (0, 0), 1.5), 0, 1)
        m = np.clip(m + 0.6 * edge * (rng.random(shape) - 0.3), 0, 1)
        return m.astype(np.float32)
    # hilos sueltos que asoman del borde
    edge = (m > 0.4) & (m < 0.8)
    ys, xs = np.nonzero(edge)
    if len(xs):
        loose = np.zeros(shape, np.float32)
        n = int(len(xs) * 0.04 * fray)
        idx = rng.choice(len(xs), max(1, n), replace=False)
        gyy, gxx = np.gradient(cv2.GaussianBlur(m, (0, 0), 2.0))
        for i in idx:
            x, y = xs[i], ys[i]
            nx, ny = -gxx[y, x], -gyy[y, x]
            nn = np.hypot(nx, ny) + 1e-6
            L = rng.uniform(2, 6) * fray
            ex, ey = x + nx / nn * L + rng.normal(0, 1), y + ny / nn * L + rng.normal(0, 1)
            cv2.line(loose, (int(x), int(y)), (int(ex), int(ey)), float(rng.uniform(0.5, 0.9)), 1, cv2.LINE_AA)
        m = np.maximum(m, loose)
    return m.astype(np.float32)


def organza(shape, rng):
    """Organza: malla finísima y transparente, con brillo que cambia con los pliegues."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mesh = 0.5 + 0.25 * np.sin(xx * 2.4) * np.sin(yy * 2.4)
    folds = smooth_noise(shape, 70, rng)
    sheen = np.clip((folds - 0.45) * 3.0, 0, 1)
    return mesh.astype(np.float32), sheen.astype(np.float32), folds.astype(np.float32)
