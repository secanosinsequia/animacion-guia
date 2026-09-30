"""La bandada: cada letra del título se vuelve un queltehue que vuela y se posa en el potrero.

* vuelo: alas con proyección 3D simple (ala cercana y lejana), 8 poses cuantizadas (dibujo a mano),
  copias arrastradas (smear) cuando el ave va rápido, aleteo de aterrizaje con patas abajo;
* en el suelo: queltehue pequeño, despierto o dormido (cabeza escondida); uno queda de guardia;
* la posta: el rojo de ALERTA se separa en gotas, se junta en un solo punto y llega al ojo de la
  vigía de turno.
"""
import numpy as np

from . import queltehue as Q
from .geometry import catmull_rom

FPS = 30.0

WING = [(0.06, 0.0), (0.078, 0.22), (0.052, 0.46), (0.004, 0.7), (-0.036, 0.9), (-0.09, 1.0),
        (-0.16, 0.98), (-0.205, 0.84), (-0.215, 0.62), (-0.19, 0.36), (-0.12, 0.0)]
HAND = [(0.004, 0.7), (-0.036, 0.9), (-0.09, 1.0), (-0.16, 0.98), (-0.205, 0.84), (-0.214, 0.62),
        (0.03, 0.6)]
BAND = [(0.056, 0.38), (0.032, 0.59), (-0.214, 0.61), (-0.196, 0.4)]


def _bez(p0, p1, p2, p3, e):
    return ((1 - e) ** 3) * p0 + 3 * e * (1 - e) ** 2 * p1 + 3 * e * e * (1 - e) * p2 + e ** 3 * p3


def _poisson(rng, box, n, rmin_fn, tries=6000, avoid=None):
    x0, y0, x1, y1 = box
    pts = []
    for _ in range(tries):
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if avoid is not None and avoid(*p):
            continue
        r = rmin_fn(p[1])
        if all(np.hypot(*(p - q)) > max(r, rmin_fn(q[1])) for q in pts):
            pts.append(p)
            if len(pts) >= n:
                break
    return pts


class Bird:
    def __init__(self, letter, land, h_land, t_pop, dur, facing, rng, u):
        self.letter = letter
        self.p0 = np.array(letter.center, np.float64)
        self.p3 = np.array(land, np.float64)
        self.h_land = h_land
        self.t_pop = t_pop
        self.t_fly = t_pop + 2.0 / FPS
        self.t_land = self.t_fly + dur
        self.facing = facing
        lh = letter.box[3] - letter.box[1]
        self.span0 = float(np.clip(lh * 0.62, 38 * u, 105 * u))
        self.span1 = h_land * 1.9
        up = rng.uniform(0.6, 1.0)
        self.p1 = self.p0 + np.array([rng.uniform(30, 120), -rng.uniform(50, 110) * up]) * u
        self.p2 = self.p3 + np.array([rng.uniform(-90, 60), -rng.uniform(110, 170)]) * u
        self.phase0 = rng.uniform(0, 2 * np.pi)
        self.freq = rng.uniform(4.2, 5.2)
        self.sleep_t = self.t_land + rng.uniform(0.18, 0.34)
        self.duty = False
        self.color = letter.color

    def state(self, t):
        if t < self.t_fly:
            return "letter"
        if t < self.t_land:
            return "fly"
        return "stand"

    def pos(self, t):
        u = np.clip((t - self.t_fly) / (self.t_land - self.t_fly), 0, 1)
        e = 1 - (1 - u) ** 1.7            # sale rápido, frena al posarse
        e = e * e * (3 - 2 * e) * 0.35 + e * 0.65
        return _bez(self.p0, self.p1, self.p2, self.p3, e), u


def _wing_pts(theta, near, flex, psi=np.deg2rad(24)):
    pts = []
    for poly in (WING, HAND, BAND):
        arr = np.array(poly, np.float64)
        x, w = arr[:, 0], arr[:, 1] * np.where(arr[:, 1] > 0.5, flex, 1.0)
        Y = w * np.sin(theta)
        Z = w * np.cos(theta) * (1 if near else -1)
        sy = -Y * np.cos(psi) + Z * np.sin(psi)
        pts.append(np.stack([x, sy], 1))
    return pts


class Flock:
    def __init__(self, title, lay, rng, t_release=3.30, queltehue_head=None):
        self.title = title
        self.lay = lay
        self.u = lay["u"]
        u = self.u
        letters = list(title.letters)
        n = len(letters)
        box = lay["land_box"]
        hmin, hmax = lay["land_h"]
        y0, y1 = box[1], box[3]
        hfun = lambda y: hmin + (hmax - hmin) * np.clip((y - y0) / (y1 - y0), 0, 1)
        spots = _poisson(rng, box, n, lambda y: hfun(y) * 1.35, avoid=lay.get("land_avoid"))
        while len(spots) < n:
            spots.append(np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])]))
        spots = sorted(spots, key=lambda p: p[0])
        # Onda de alarma: desde la vigía (izquierda) hacia la derecha
        src = np.asarray(queltehue_head if queltehue_head is not None else (0, 0), np.float64)
        order = sorted(range(n), key=lambda i: np.hypot(*(letters[i].center - src)))
        dmin = np.hypot(*(letters[order[0]].center - src))
        dmax = np.hypot(*(letters[order[-1]].center - src))
        # asignación por x (evita cruces), con algo de desorden
        by_x = sorted(range(n), key=lambda i: letters[i].center[0] + rng.normal(0, 40 * u))
        self.birds = []
        for rank, i in enumerate(by_x):
            L = letters[i]
            d = np.hypot(*(L.center - src))
            t_pop = t_release + 0.30 * (d - dmin) / max(1.0, dmax - dmin)
            sp = spots[rank]
            h = hfun(sp[1])
            facing = 1 if rng.random() < 0.6 else -1
            b = Bird(L, sp, h, t_pop, rng.uniform(0.52, 0.66), facing, rng, u)
            title.pop_time[id(L)] = t_pop
            self.birds.append(b)
        # Vigía de turno: la que se posa más cerca del punto elegido
        duty_pt = np.array(lay["duty_pt"])
        self.duty = min(self.birds, key=lambda b: np.hypot(*(b.p3 - duty_pt)))
        self.duty.duty = True
        self.duty.facing = 1
        # Gotas rojas de ALERTA
        self.drops = [b for b in self.birds if b.color == "red"]
        self.merge_pt = np.array(lay["merge_pt"])
        self.t_merge = max(b.t_pop for b in self.drops) + 0.26
        self.t_eye = max(self.t_merge + 0.34, self.duty.t_land + 0.08)

    # --- geometría del ave de pie -------------------------------------------------------------------
    def duty_eye(self):
        b = self.duty
        s = b.h_land / 0.82
        e = np.array(Q.EYE) * np.array([b.facing, 1.0]) * s + b.p3
        return e, 0.0135 * s

    def _stand(self, masks, b, t):
        s = b.h_land / 0.82
        f = b.facing
        X = lambda pts: np.asarray(pts, np.float64) * np.array([f * s, s]) + b.p3
        asleep = (t >= b.sleep_t) and not b.duty
        # asentarse: los 2 primeros cuadros con alas aún abiertas
        k = int(np.floor((t - b.t_land) * FPS))
        grey, white, ink = masks["grey"], masks["white"], masks["ink"]
        body = catmull_rom(np.array(Q.BODY), 6, closed=True)
        grey.fill_poly(X(body), 1.0)
        white.fill_poly(X(catmull_rom(np.array(Q.BELLY), 5, closed=True)), 1.0)
        for poly in (Q.BIB, Q.PRIMARIES, Q.TAILBAND):
            ink.fill_poly(X(catmull_rom(np.array(poly), 4, closed=True)), 1.0)
        c = ink.ctx
        c.set_line_cap(1)
        lw = max(0.8, 0.012 * s)
        for leg in (Q.LEG_FAR, Q.LEG_NEAR):
            p = X(leg)
            c.set_line_width(lw)
            c.move_to(*p[0])
            for q in p[1:]:
                c.line_to(*q)
            c.stroke()
        # contorno fino del cuerpo
        c.set_line_width(max(0.6, 0.008 * s))
        pb = X(body)
        c.move_to(*pb[0])
        for q in pb[1:]:
            c.line_to(*q)
        c.close_path()
        ink.set_alpha(0.85)
        c.stroke()
        ink.set_alpha(1.0)
        if asleep:
            hc = np.array([0.07, -0.672])
            hr = Q.HEAD_R * 0.95
            grey.dot(*X([hc])[0], hr * s, 1.0)
            ink.set_alpha(1.0)
            cr = X([(0.05, -0.72), (-0.02, -0.745), (-0.08, -0.742)])
            c.set_line_width(max(0.7, 0.01 * s))
            c.move_to(*cr[0])
            c.line_to(*cr[1])
            c.line_to(*cr[2])
            c.stroke()
            return
        hc = X([Q.HEAD_C])[0]
        grey.dot(hc[0], hc[1], Q.HEAD_R * s, 1.0)
        grey.fill_poly(X([Q.NECK_BACK, Q.HEAD_BACK, Q.HEAD_THROAT, Q.NECK_FRONT]), 1.0)
        for poly in (Q.FOREHEAD, Q.THROAT):
            ink.fill_poly(X(poly), 1.0)
        ink.fill_poly(X(Q.BILL), 1.0)
        cr = X(Q.CREST[0])
        c.set_line_width(max(0.7, 0.011 * s))
        c.move_to(*cr[0])
        for q in cr[1:]:
            c.line_to(*q)
        c.stroke()
        if b.duty:
            e, r = self.duty_eye()
            ink.dot(e[0], e[1], r * 1.35, 1.0)
            if t >= self.t_eye:
                masks["white_over"].dot(e[0], e[1], max(r * 1.6, 3.2), 1.0)
                masks["red_over"].dot(e[0], e[1], max(r * 1.2, 2.4), 1.0)
                if t >= self.t_eye + 0.05:
                    masks["white_over"].dot(e[0] - r * 0.35, e[1] - r * 0.4, max(r * 0.35, 0.5), 1.0)
        else:
            e = X([Q.EYE])[0]
            ink.dot(e[0], e[1], max(0.9, 0.012 * s), 1.0)

    # --- vuelo ------------------------------------------------------------------------------------------
    @staticmethod
    def _wing2d(alpha_deg, bend, length):
        """Ala en 2D (estilo sumi-e): siempre muestra su largo. Devuelve (brazo, barra, mano, contorno)."""
        a = np.deg2rad(alpha_deg)
        n = 14
        sgrid = np.linspace(0, 1, n)
        # ancho: angosto en el hombro, ancho en la mano, punta redondeada
        wprof = 0.07 + 0.15 * np.sin(np.clip(sgrid, 0, 1) * np.pi * 0.6) - 0.9 * np.clip(sgrid - 0.84, 0, 1) ** 1.6
        wprof = np.maximum(wprof, 0.02)
        pts_c = []
        ang = a
        p = np.zeros(2)
        for i in range(n):
            if i:
                seg = length / (n - 1)
                if sgrid[i] > 0.5:           # la mano se dobla hacia atrás en la subida
                    ang = a + np.deg2rad(bend) * (sgrid[i] - 0.5) * 2
                p = p + seg * np.array([-np.cos(ang), -np.sin(ang)])
            pts_c.append(p.copy())
        pts_c = np.array(pts_c)
        tang = np.gradient(pts_c, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nor = np.stack([-tang[:, 1], tang[:, 0]], 1)
        lead = pts_c + nor * (wprof[:, None] * length * 0.35)
        trail = pts_c - nor * (wprof[:, None] * length * 0.65)   # el borde de fuga es más ancho

        def band(s0, s1, edge=lead):
            i0, i1 = int(s0 * (n - 1)), int(np.ceil(s1 * (n - 1)))
            return np.vstack([edge[i0:i1 + 1], trail[i0:i1 + 1][::-1]])
        outline = np.vstack([lead, trail[::-1]])
        # mancha blanca solo en el borde de fuga del brazo (secundarias), no una franja completa
        return band(0.0, 0.62), band(0.22, 0.62, edge=pts_c), band(0.58, 1.0), outline

    def _fly(self, masks, b, t, alpha=1.0):
        p, u = b.pos(t)
        p_prev, _ = b.pos(t - 1.0 / FPS)
        v = p - p_prev
        span = b.span0 + (b.span1 - b.span0) * (u ** 0.8)
        f = 1 if v[0] >= 0 else -1
        ang = np.arctan2(v[1], abs(v[0]) + 1e-6)
        ang = float(np.clip(ang, np.deg2rad(-35), np.deg2rad(40)))
        # 8 poses de aleteo (M -> V), cuantizadas como dibujos
        phase = b.phase0 + 2 * np.pi * b.freq * (t - b.t_fly)
        pose = np.round((phase % (2 * np.pi)) / (2 * np.pi) * 8) % 8
        ph = pose / 8 * 2 * np.pi
        wa = 35 + 62 * np.cos(ph)                 # de 97° (arriba) a -27° (abajo)
        bend = -38 if np.sin(ph) > 0 else 8       # subida: mano doblada
        landing = u > 0.8
        if landing:
            wa, bend = 100, -20
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[ca, -sa], [sa, ca]])
        L = span * 0.6

        def T(pts, scale=True):
            q = np.asarray(pts, np.float64) * (span if scale else 1.0)
            q = q @ R.T
            q[:, 0] *= f
            return q + p

        ink, white, grey = masks["ink"], masks["white"], masks["grey"]
        sh = np.array([0.03, -0.02]) * span
        # ala lejana: más corta y más alta, en sombra
        far = self._wing2d(wa + 14, bend, L * 0.78)
        for k, poly in enumerate(far[:3]):
            q = poly + sh
            (ink if k != 1 else grey).fill_poly(T(q, False), alpha * (0.9 if k != 1 else 1.0))
        # cuerpo pequeño con babero y vientre blanco
        a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        body = np.stack([0.14 * np.cos(a), 0.042 * np.sin(a)], 1)
        grey.fill_poly(T(body), alpha)
        white.fill_poly(T(np.stack([0.11 * np.cos(a[4:16]), 0.042 * np.abs(np.sin(a[4:16]))], 1)), alpha * 0.95)
        ink.fill_poly(T([(0.09, -0.028), (0.16, -0.03), (0.17, 0.018), (0.09, 0.03)]), alpha)   # babero
        ink.dot(*T([(0.155, -0.024)])[0], 0.036 * span, alpha)                                # cabeza
        ink.fill_poly(T([(0.18, -0.03), (0.23, -0.022), (0.18, -0.014)]), alpha)               # pico
        ink.fill_poly(T([(-0.12, -0.018), (-0.2, -0.006), (-0.2, 0.01), (-0.12, 0.02)]), alpha)  # cola
        c = ink.ctx
        cr = T([(0.14, -0.05), (0.09, -0.072), (0.045, -0.076)])
        c.set_line_width(max(0.7, 0.011 * span))
        ink.set_alpha(alpha)
        c.move_to(*cr[0])
        c.line_to(*cr[1])
        c.line_to(*cr[2])
        c.stroke()
        if landing:
            legs = T([(0.02, 0.03), (0.0, 0.17)])
            c.set_line_width(max(0.7, 0.01 * span))
            c.move_to(*legs[0])
            c.line_to(*legs[1])
            c.stroke()
        # ala cercana: brazo pardo, barra blanca, mano negra y contorno de plumilla
        near = self._wing2d(wa, bend, L)
        grey.fill_poly(T(near[0] + sh, False), alpha)
        white.fill_poly(T(near[1] + sh, False), alpha * 0.95)
        ink.fill_poly(T(near[2] + sh, False), alpha)
        q = T(near[3] + sh, False)
        c.set_line_width(max(0.6, 0.008 * span))
        c.move_to(*q[0])
        for pt in q[1:]:
            c.line_to(*pt)
        c.close_path()
        ink.set_alpha(alpha * 0.75)
        c.stroke()

    def draw(self, masks, t):
        for b in self.birds:
            st = b.state(t)
            if st == "fly":
                p, u = b.pos(t)
                p_prev, _ = b.pos(t - 1.0 / FPS)
                if np.hypot(*(p - p_prev)) > 26 * self.u:   # muy rápido: una copia arrastrada
                    self._fly(masks, b, t - 0.6 / FPS, alpha=0.16)
                self._fly(masks, b, t)
            elif st == "stand":
                self._stand(masks, b, t)
        self._drops(masks, t)

    # --- la posta del rojo ------------------------------------------------------------------------------
    def _drops(self, masks, t):
        red = masks["red_over"]
        r0 = 5.5 * self.u
        if t >= self.t_eye:
            return
        arrived = 0
        for b in self.drops:
            if t < b.t_pop + 1.0 / FPS:
                continue
            u = np.clip((t - b.t_pop) / max(1e-3, self.t_merge - b.t_pop), 0, 1)
            e = 1 - (1 - u) ** 2
            if u >= 1:
                arrived += 1
                continue
            start = b.p0
            mid = (start + self.merge_pt) / 2 + np.array([0, -40 * self.u])
            p = (1 - e) ** 2 * start + 2 * e * (1 - e) * mid + e * e * self.merge_pt
            rr = r0 * (1.4 - 0.4 * e)
            red.dot(p[0], p[1], rr, 1.0)
            # pequeña estela
            p2 = (1 - max(e - 0.08, 0)) ** 2 * start + 2 * max(e - 0.08, 0) * (1 - max(e - 0.08, 0)) * mid + \
                max(e - 0.08, 0) ** 2 * self.merge_pt
            red.dot(p2[0], p2[1], rr * 0.55, 0.5)
        if arrived > 0:
            k = arrived
            r = r0 * np.sqrt(k) * 0.8
            if t < self.t_merge:
                red.dot(self.merge_pt[0], self.merge_pt[1], r, 1.0)
            else:
                eye, er = self.duty_eye()
                u = np.clip((t - self.t_merge) / max(1e-3, self.t_eye - self.t_merge), 0, 1)
                e = u * u * (3 - 2 * u)
                mid = (self.merge_pt + eye) / 2 + np.array([60 * self.u, -70 * self.u])
                p = (1 - e) ** 2 * self.merge_pt + 2 * e * (1 - e) * mid + e * e * eye
                rr = r * (1 - e) + max(er * 1.2, 2.4) * e
                red.dot(p[0], p[1], rr, 1.0)
