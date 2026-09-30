"""fig. 2 — Torre de medición de viento (mástil anemométrico).

Es la única línea que «no es propia del lugar»: se traza con regla y tiralíneas, de grosor
constante, sin charcos ni temblor, a velocidad mecánica (lineal). Mástil de celosía, cables
tensores rectos a tres anclajes, brazos laterales con anemómetros de cazoletas que giran y una veleta.
"""
import numpy as np


class Tower:
    def __init__(self, base, height, scale=1.0, t0=0.20, t1=1.0, anchors=None):
        self.bx, self.by = base
        self.Ht = height
        self.u = scale
        self.t0, self.t1 = t0, t1
        bx, by, H = self.bx, self.by, height
        self.w = 6.0 * scale                      # ancho aparente del mástil
        self.top = by - H
        self.segments = []                        # (p0, p1, ancho, ts, te)
        span = t1 - t0

        def T(a, b):
            return t0 + a * span, t0 + b * span

        # Mástil: dos montantes que suben a la vez (lineal, frío)
        ts, te = T(0.0, 0.45)
        for side in (-1, 1):
            self.segments.append(((bx + side * self.w / 2, by), (bx + side * self.w / 2, self.top), 1.5 * scale, ts, te))
        # Celosía en zigzag que sigue a los montantes
        n = int(H / (7.5 * scale))
        for i in range(n):
            y0 = by - H * i / n
            y1 = by - H * (i + 1) / n
            s0 = -1 if i % 2 == 0 else 1
            a, b = i / n, (i + 1) / n
            self.segments.append(((bx + s0 * self.w / 2, y0), (bx - s0 * self.w / 2, y1), 0.8 * scale,
                                  *T(0.02 + 0.45 * a, 0.02 + 0.45 * b)))
        # Pararrayos
        self.segments.append(((bx, self.top), (bx, self.top - 14 * scale), 1.0 * scale, *T(0.45, 0.5)))
        # Tensores: 4 alturas x 3 anclajes, líneas rectas que «disparan» desde el mástil
        if anchors is None:
            anchors = [(bx - 0.50 * H, by + 34 * scale), (bx + 0.44 * H, by + 40 * scale),
                       (bx + 0.07 * H, by + 0.19 * H)]
        self.anchors = anchors
        levels = [0.30, 0.55, 0.78, 0.97]
        k = 0
        for li, f in enumerate(levels):
            for ai, (ax, ay) in enumerate(anchors):
                py = by - H * f
                a = 0.48 + 0.06 * li + 0.02 * ai
                self.segments.append(((bx, py), (ax, ay), 0.75 * scale, *T(a, a + 0.14)))
                k += 1
        # Bloques de anclaje
        for (ax, ay) in anchors:
            self.segments.append(((ax - 4 * scale, ay), (ax + 4 * scale, ay), 1.4 * scale, *T(0.72, 0.76)))
        # Base
        self.segments.append(((bx - 12 * scale, by + 1), (bx + 12 * scale, by + 1), 1.6 * scale, *T(0.0, 0.05)))
        # Brazos con anemómetros y veleta
        self.booms = []
        for j, (f, side) in enumerate([(0.42, 1), (0.66, -1), (0.9, 1), (0.995, -1)]):
            py = by - H * f
            L = 26 * scale
            ex = bx + side * (self.w / 2 + L)
            a = 0.80 + 0.035 * j
            self.segments.append(((bx + side * self.w / 2, py), (ex, py), 1.1 * scale, *T(a, a + 0.05)))
            self.segments.append(((ex, py), (ex, py - 9 * scale), 1.0 * scale, *T(a + 0.05, a + 0.07)))
            self.booms.append((ex, py - 9 * scale, t0 + (a + 0.08) * span, j == 3))

    def draw(self, mask, t, alpha=1.0):
        c = mask.ctx
        c.set_line_cap(0)  # BUTT: cortes secos, de tiralíneas
        mask.set_alpha(alpha)
        for (p0, p1, w, ts, te) in self.segments:
            if t <= ts:
                continue
            u = min(1.0, (t - ts) / max(1e-6, te - ts))  # lineal: sin aceleración humana
            x = p0[0] + (p1[0] - p0[0]) * u
            y = p0[1] + (p1[1] - p0[1]) * u
            c.set_line_width(w)
            c.move_to(*p0)
            c.line_to(x, y)
            c.stroke()
        # Anemómetros de cazoletas (giran) y veleta
        for (x, y, ta, is_vane) in self.booms:
            if t <= ta:
                continue
            appear = min(1.0, (t - ta) / 0.06)
            if is_vane:
                c.set_line_width(1.0 * self.u)
                c.move_to(x - 7 * self.u * appear, y)
                c.line_to(x + 7 * self.u * appear, y)
                c.stroke()
                mask.fill_poly([(x + 7 * self.u, y - 3 * self.u), (x + 11 * self.u, y), (x + 7 * self.u, y + 3 * self.u)],
                               alpha * appear)
                continue
            rot = (t - ta) * 2.2 * 2 * np.pi
            for k in range(3):
                a = rot + k * 2 * np.pi / 3
                cx = x + 6.5 * self.u * np.cos(a) * appear
                cy = y + 2.0 * self.u * np.sin(a) * appear
                c.set_line_width(0.7 * self.u)
                c.move_to(x, y)
                c.line_to(cx, cy)
                c.stroke()
                mask.dot(cx, cy, 1.7 * self.u * appear, alpha)
