"""Papel kraft claro procedural: formación nubosa, fibras, motas, grano y viñeta."""
import cv2
import numpy as np

from .color import lin, PALETTE
from .noise import fbm, smooth_noise


def _fibers(h, w, rng, n, len_range, width_range, straightness=0.85, aspect_bias=0.25):
    """Dibuja fibras cortas y curvas en una capa float (0..1) con antialias."""
    ss = 2  # sobremuestreo para fibras finas
    layer = np.zeros((h * ss, w * ss), np.float32)
    # Dirección predominante levemente horizontal (sentido de máquina del papel).
    for _ in range(n):
        x, y = rng.random() * w * ss, rng.random() * h * ss
        L = rng.uniform(*len_range) * ss
        ang = rng.normal(0.0, 1.0) * (1.2 - aspect_bias) + (rng.random() < 0.5) * np.pi
        steps = max(3, int(L / (3 * ss)))
        pts = []
        curv = rng.normal(0, 1 - straightness) * 0.25
        for i in range(steps):
            pts.append((x, y))
            ang += curv + rng.normal(0, 0.06)
            x += np.cos(ang) * L / steps
            y += np.sin(ang) * L / steps
        wpx = max(1, int(round(rng.uniform(*width_range) * ss)))
        val = rng.uniform(0.35, 1.0)
        cv2.polylines(layer, [np.array(pts, np.int32)], False, float(val), wpx, cv2.LINE_AA)
    layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_AREA)
    return layer


def make_kraft(w, h, seed=7, lift_n=0):
    """Devuelve (rgb_lineal HxWx3, altura_papel HxW en [0,1], motas_sueltas Nx4).

    `lift_n` motas NO se hornean en el papel: se dibujan aparte porque se levantarán (el murmullo).
    Cada fila de motas_sueltas es (x, y, radio, intensidad)."""
    rng = np.random.default_rng(seed)
    base = lin(PALETTE["kraft"])

    # Formación del papel: manchas grandes + nubosidad media.
    mottling = fbm((h, w), 520, rng, octaves=3) - 0.5
    cloud = fbm((h, w), 70, rng, octaves=3) - 0.5
    grain = rng.standard_normal((h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.6)
    grain /= grain.std() + 1e-6

    # Fibras oscuras (corteza) y claras (celulosa blanqueada).
    dark = _fibers(h, w, rng, n=int(w * h / 140), len_range=(3, 18), width_range=(0.4, 0.9))
    dark *= smooth_noise((h, w), 60, rng) * 1.4
    light = _fibers(h, w, rng, n=int(w * h / 200), len_range=(4, 22), width_range=(0.5, 1.2))
    light *= smooth_noise((h, w), 60, rng) * 1.4
    long_dark = _fibers(h, w, rng, n=int(w * h / 16000), len_range=(20, 70), width_range=(0.5, 1.0),
                        straightness=0.95)

    # Motas / inclusiones.
    specks = np.zeros((h, w), np.float32)
    for _ in range(int(w * h / 5000)):
        x, y = int(rng.random() * w), int(rng.random() * h)
        r = rng.choice([1, 1, 1, 2, 2, 3])
        cv2.circle(specks, (x, y), int(r), float(rng.uniform(0.3, 1.0)), -1, cv2.LINE_AA)
    specks = cv2.GaussianBlur(specks, (0, 0), 0.7)
    # Motas que se levantarán: algo más grandes y oscuras, repartidas por toda la hoja.
    lifters = np.stack([rng.uniform(40, w - 40, lift_n), rng.uniform(40, h - 40, lift_n),
                        rng.uniform(1.1, 2.2, lift_n), rng.uniform(0.55, 0.9, lift_n)], 1) if lift_n else np.zeros((0, 4))

    # Micro-textura fibrosa: ruido estirado en direcciones aleatorias (fieltro de celulosa).
    felt = np.zeros((h, w), np.float32)
    for ang in rng.uniform(0, np.pi, 5):
        k = 9
        ker = np.zeros((k, k), np.float32)
        c = k // 2
        dx, dy = np.cos(ang), np.sin(ang)
        for t in np.linspace(-c, c, 4 * k):
            ker[int(round(c + dy * t)), int(round(c + dx * t))] = 1
        ker /= ker.sum()
        n = rng.standard_normal((h, w)).astype(np.float32)
        felt += cv2.filter2D(n, -1, ker)
    felt /= felt.std() + 1e-6
    floc = fbm((h, w), 34, rng, octaves=3) - 0.5

    lum = (1.0
           + 0.060 * mottling
           + 0.050 * cloud
           + 0.060 * floc
           + 0.020 * grain
           + 0.026 * felt
           - 0.085 * dark
           - 0.14 * long_dark
           + 0.065 * light
           - 0.24 * specks)

    # Variación de tono: zonas algo más rojizas / amarillentas.
    hue = smooth_noise((h, w), 380, rng) - 0.5
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0] = base[0] * lum * (1 + 0.035 * hue)
    rgb[..., 1] = base[1] * lum * (1 + 0.005 * hue)
    rgb[..., 2] = base[2] * lum * (1 - 0.05 * hue)
    # Las fibras oscuras son pardo-rojizas: restan más azul.
    rgb[..., 2] *= (1 - 0.10 * dark)

    # Viñeta suave y luz rasante desde arriba a la izquierda.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx / w - 0.5) * 2, (yy / h - 0.5) * 2
    r2 = (nx ** 2 * 0.9 + ny ** 2 * 1.1)
    vign = 1.0 - 0.16 * np.clip(r2, 0, 2.2) ** 1.6 / 2.2 ** 1.6
    light_dir = 1.0 + 0.05 * (-(nx * 0.6 + ny * 0.8))
    rgb *= (vign * light_dir)[..., None]

    height = np.clip(0.5 + 0.10 * grain / 3.0 + 0.12 * felt / 3.0 + 0.9 * floc + 0.6 * cloud
                     + 0.35 * light - 0.35 * dark, 0, 1)
    height = cv2.GaussianBlur(height.astype(np.float32), (0, 0), 0.8)
    return np.clip(rgb, 0, 1).astype(np.float32), height.astype(np.float32), lifters.astype(np.float32)
