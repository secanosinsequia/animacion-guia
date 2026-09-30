"""El murmullo: las motas del propio kraft se levantan y forman las palabras como densidad de puntos.

«Señales débiles que cobran sentido al juntarse» (Guía SATC, 3.1.3). Cada mota ya se ve en el
cuadro 0; al despegarse deja un hueco claro en el papel, gira junto a las demás en espiral (todas en
el mismo sentido, como una murmuración) y se asienta dentro de una letra. Hasta el clic, la letra
existe SOLO como puntillado que tiembla «en dos»; sin fundidos.
"""
import numpy as np

FPS = 30.0


def blue_noise_in_mask(mask, n, rng, rmin):
    """Muestreo con distancia mínima (tirar dardos sobre una grilla) dentro de una máscara booleana."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2))
    order = rng.permutation(len(xs))
    cell = rmin / np.sqrt(2)
    grid = {}
    pts = []
    for i in order:
        x, y = xs[i] + rng.random(), ys[i] + rng.random()
        gx, gy = int(x / cell), int(y / cell)
        ok = True
        for ax in range(gx - 2, gx + 3):
            for ay in range(gy - 2, gy + 3):
                q = grid.get((ax, ay))
                if q is not None and (q[0] - x) ** 2 + (q[1] - y) ** 2 < rmin * rmin:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[(gx, gy)] = (x, y)
            pts.append((x, y))
            if len(pts) >= n:
                break
    return np.array(pts)


class Murmur:
    def __init__(self, sources, targets, rng, center, u=1.0, t_lift=(0.86, 1.12), travel=(0.40, 0.52),
                 t_print=1.933, turns=(0.10, 0.35), fill=None, fill_frames=8):
        n = len(targets)
        self.p0 = sources[:n, :2].astype(np.float64)
        self.r = sources[:n, 2] * u
        self.dark = sources[:n, 3]
        self.p3 = targets.astype(np.float64)
        self.C = np.asarray(center, np.float64)
        d0, d1 = self.p0 - self.C, self.p3 - self.C
        self.r0 = np.hypot(d0[:, 0], d0[:, 1])
        self.r1 = np.hypot(d1[:, 0], d1[:, 1])
        self.a0 = np.arctan2(d0[:, 1], d0[:, 0])
        a1 = np.arctan2(d1[:, 1], d1[:, 0])
        self.da = np.mod(a1 - self.a0, 2 * np.pi) + 2 * np.pi * rng.uniform(*turns, n)
        rn = (self.r0 - self.r0.min()) / max(1e-6, self.r0.max() - self.r0.min())
        span = t_lift[1] - t_lift[0]
        self.t0 = t_lift[0] + span * (1 - rn) * 0.5 + rng.uniform(0, 0.5, n) * span
        self.t1 = self.t0 + rng.uniform(*travel, n)
        self.t_print = t_print
        self.wob = rng.uniform(0, 2 * np.pi, (n, 2))
        self.squash = rng.uniform(0.55, 0.75, n)
        self.u = u
        self.seed = int(rng.integers(1 << 30))
        # tinta que se asienta: puntos finos que aparecen «en dos» dentro de los trazos antes del clic,
        # para que el puntillado ya se lea como palabra y el clic lo confirme
        self.fill = np.zeros((0, 3)) if fill is None else np.asarray(fill, np.float64)
        if len(self.fill):
            k = rng.integers(1, fill_frames // 2 + 1, len(self.fill)) * 2
            self.fill_t = t_print - k / FPS
            self.fill_r = rng.uniform(1.0, 1.45, len(self.fill)) * u

    def _pos(self, t):
        e = np.clip((t - self.t0) / (self.t1 - self.t0), 0, 1)
        s = e * e * (3 - 2 * e)
        sr = 1 - (1 - s) ** 1.6
        r = self.r0 + (self.r1 - self.r0) * sr
        a = self.a0 + self.da * s
        k = 1 - (1 - self.squash) * np.sin(np.pi * s)
        p = np.stack([self.C[0] + r * np.cos(a), self.C[1] + r * np.sin(a) * k], 1)
        w = np.clip((s - 0.85) / 0.15, 0, 1)[:, None]
        p = p * (1 - w) + self.p3 * w
        env = np.sin(np.pi * s)
        p[:, 0] += 4 * self.u * env * np.sin(t * 9 + self.wob[:, 0])
        p[:, 1] += 4 * self.u * env * np.cos(t * 8 + self.wob[:, 1])
        p = np.where((e <= 0)[:, None], self.p0, p)
        return p, e, env

    def draw(self, masks, t):
        if t >= self.t_print - 1e-6:
            # las motas ya son tinta impresa: solo quedan los huecos claros en el papel
            for i in range(len(self.p0)):
                masks["holes"].dot(self.p0[i, 0], self.p0[i, 1], self.r[i] * 1.05, 0.55)
            return
        p, e, env = self._pos(t)
        pp, _, _ = self._pos(t - 1.0 / 60)
        # temblor «en dos» (muy leve) de las motas ya asentadas
        step = int(np.floor(t * FPS / 2))
        jr = np.random.default_rng((self.seed + step * 7919) % (1 << 31))
        jit = jr.normal(0, 0.45 * self.u, p.shape)
        for i in np.nonzero(t >= self.fill_t)[0] if len(self.fill) else []:
            masks["speck"].dot(self.fill[i, 0], self.fill[i, 1], self.fill_r[i], 0.96)
        for i in range(len(p)):
            ei = e[i]
            if ei > 0:
                masks["holes"].dot(self.p0[i, 0], self.p0[i, 1], self.r[i] * 1.05, 0.55 * min(1.0, ei / 0.1))
            if ei <= 0:
                masks["speck"].dot(p[i, 0], p[i, 1], self.r[i], self.dark[i] * 0.62)
            elif ei < 1:
                a = self.dark[i] * (0.62 + 0.36 * min(1.0, ei / 0.12))
                masks["shadow"].dot(p[i, 0] + 2.2 * self.u, p[i, 1] + 2.6 * self.u, self.r[i] * 1.3, 0.3 * a)
                masks["speck"].dot(pp[i, 0], pp[i, 1], self.r[i] * 0.75, 0.45 * a)
                masks["speck"].dot(p[i, 0], p[i, 1], self.r[i] * (1 + 0.3 * env[i]), a)
            else:
                q = p[i] + jit[i]
                masks["speck"].dot(q[0], q[1], float(np.clip(self.r[i] * 0.9, 1.3 * self.u, 2.1 * self.u)), 0.98)
