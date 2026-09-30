"""Ruido procedural rápido con numpy + OpenCV (sin dependencias nativas extra)."""
import cv2
import numpy as np


def smooth_noise(shape, scale, rng, oversample=1.0):
    """Ruido suave isotrópico de ~`scale` px de longitud de correlación, en [0, 1]."""
    h, w = shape
    s = max(float(scale), 1.0)
    # Ruido blanco a baja resolución, filtrado gaussiano y reescalado bicúbico.
    lh = int(np.ceil(h / s * 2 * oversample)) + 6
    lw = int(np.ceil(w / s * 2 * oversample)) + 6
    g = rng.standard_normal((lh, lw)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=1.0 * oversample, borderType=cv2.BORDER_REFLECT)
    up = cv2.resize(g, (int(lw * s / (2 * oversample)), int(lh * s / (2 * oversample))),
                    interpolation=cv2.INTER_CUBIC)
    oy = int(rng.integers(0, max(1, up.shape[0] - h)))
    ox = int(rng.integers(0, max(1, up.shape[1] - w)))
    up = up[oy:oy + h, ox:ox + w]
    if up.shape != (h, w):
        up = cv2.resize(up, (w, h), interpolation=cv2.INTER_CUBIC)
    up -= up.mean()
    sd = up.std() + 1e-6
    return np.clip(0.5 + up / (6 * sd), 0, 1).astype(np.float32)


def fbm(shape, scale, rng, octaves=5, persistence=0.5, lacunarity=2.0):
    """Movimiento browniano fraccional (suma de octavas de ruido suave), en [0, 1]."""
    total = np.zeros(shape, np.float32)
    amp, norm, s = 1.0, 0.0, float(scale)
    for _ in range(octaves):
        if s < 1.5:
            n = rng.random(shape).astype(np.float32)
        else:
            n = smooth_noise(shape, s, rng)
        total += amp * (n - 0.5)
        norm += amp
        amp *= persistence
        s /= lacunarity
    total /= norm
    total -= total.mean()
    return np.clip(0.5 + total / (6 * (total.std() + 1e-6)), 0, 1).astype(np.float32)


def warp(img, dx, dy, border=cv2.BORDER_REFLECT):
    """Deforma `img` con los campos de desplazamiento dx, dy (en píxeles)."""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + dx.astype(np.float32), ys + dy.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=border)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)
