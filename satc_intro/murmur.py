"""El murmullo: las motas del propio kraft se levantan y convergen para formar el indicio.

«Señales débiles que cobran sentido al juntarse» (Guía SATC, 3.1.3). Las motas estuvieron en el
papel desde el primer cuadro; se despegan con una sombrita, giran juntas como una bandada y se
funden en las letras en tinta.
"""
import numpy as np


class Murmur:
    def __init__(self, specks, targets, rng, center, u=1.0, t_lift=(1.10, 1.38), travel=(0.38, 0.52)):
        self.p0 = specks[:, :2].astype(np.float64)
        self.r = specks[:, 2] * u
        self.dark = specks[:, 3]
        n = len(self.p0)
        idx = rng.permutation(len(targets))[:n] if len(targets) >= n else rng.integers(0, len(targets), n)
        self.p3 = targets[idx].astype(np.float64)
        # las más cercanas al título salen después (onda que se cierra sobre él)
        d = np.hypot(*(self.p0 - center).T)
        dn = (d - d.min()) / max(1e-6, d.max() - d.min())
        self.t0 = t_lift[0] + (t_lift[1] - t_lift[0]) * (1 - dn) * 0.6 + rng.uniform(0, 0.4, n) * (t_lift[1] - t_lift[0])
        self.t1 = self.t0 + rng.uniform(*travel, n)
        # remolino colectivo: todas giran en el mismo sentido alrededor del título
        mid = (self.p0 + self.p3) / 2
        v = self.p3 - self.p0
        perp = np.stack([-v[:, 1], v[:, 0]], 1)
        self.c1 = self.p0 + (mid - self.p0) * 0.6 + perp * 0.35
        self.c2 = self.p3 + (mid - self.p3) * 0.4 + perp * 0.15
        self.wob = rng.uniform(0, 2 * np.pi, (n, 2))
        self.u = u

    def draw(self, mask, t):
        e = np.clip((t - self.t0) / (self.t1 - self.t0), 0, 1)
        s = e * e * (3 - 2 * e)
        p = ((1 - s) ** 3)[:, None] * self.p0 + (3 * s * (1 - s) ** 2)[:, None] * self.c1 + \
            (3 * s * s * (1 - s))[:, None] * self.c2 + (s ** 3)[:, None] * self.p3
        env = np.sin(np.pi * s)
        p[:, 0] += 6 * self.u * env * np.sin(t * 9 + self.wob[:, 0])
        p[:, 1] += 6 * self.u * env * np.cos(t * 8 + self.wob[:, 1])
        fade = np.clip(1 - (t - self.t1) / 0.12, 0, 1)
        for i in range(len(p)):
            if fade[i] <= 0:
                continue
            lifted = 0 < e[i] < 1
            wake = np.clip(e[i] / 0.15, 0, 1) if e[i] < 1 else 1.0
            a = self.dark[i] * fade[i] * (0.30 + 0.65 * wake)
            if lifted:  # sombrita sobre el papel: se nota que se despegó
                mask["shadow"].dot(p[i, 0] + 2.2 * self.u, p[i, 1] + 2.6 * self.u, self.r[i] * 1.3, 0.35 * a)
            mask["speck"].dot(p[i, 0], p[i, 1], self.r[i] * (1 + 0.25 * env[i]), a)
