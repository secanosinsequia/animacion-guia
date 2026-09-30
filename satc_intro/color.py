"""Conversión de color y paleta (combinaciones de Sanzo Wada)."""
import numpy as np


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def lin(h):
    """Color hex (sRGB) -> RGB lineal."""
    return srgb_to_linear(hex_to_rgb(h))


# Paleta: combinaciones n.º 241, 243 y 126 del «Dictionary of Color Combinations» (Sanzo Wada),
# vía github.com/mattdesl/dictionary-of-colour-combinations
PALETTE = {
    "kraft": "#cdb189",          # kraft claro (entre Isabella #c5a56e e Ivory Buff #ebd3a2)
    "isabella": "#c5a56e",
    "alerta": "#dd4027",         # Red Orange (241)
    "alba": "#ffefae",           # Pale Lemon Yellow (241)
    "alba_gouache": "#f7ebc6",   # blanco cálido de gouache (Sulpher Yellow #f5ecc2)
    "medici": "#547076",         # Dark Medici Blue (241)
    "slate": "#34454c",          # Slate Color (243)
    "oliva": "#6b7140",          # Olive Green (243)
    "bosque": "#253122",         # Deep Slate Olive
    "cosaco": "#437742",         # Cossack Green
    "siena": "#bb7125",          # Raw Sienna (243)
    "ocre": "#e2b540",           # Yellow Ocher (126)
    "lyons": "#1c4286",          # Deep Lyons Blue (126)
    "tinta": "#2a2019",          # negro cálido de plumilla
    "sepia": "#4b3317",          # Vandyke Brown
    "rojo_ingles": "#d96629",    # English Red (190)
}
