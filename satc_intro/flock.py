"""La bandada: cada letra del título se vuelve un queltehue que vuela y se posa en el potrero.

Metamorfosis (≈0,6 s por letra, en onda circular desde el ojo de la vigía):
  1. la letra se aprieta 2 cuadros (escala Y 0,8) — lo dibuja title.py;
  2. se parte por su eje y sus mitades giran como alas alrededor de la base (V, M, A, T son alas);
  3. las mitades-ala aletean mientras aparece un cuerpo pequeño y la figura se achica;
  4. se funde en el queltehue en vuelo (poses dibujadas), que baja y se posa.

La posta: el rojo de ALERTA escurre en un solo trazo de acuarela hasta el ojo de la vigía de turno,
que queda en primer plano, más grande; las demás duermen con la cabeza escondida, en una pata.
"""
import numpy as np

from . import queltehue as Q
from .geometry import catmull_rom

FPS = 30.0
K_OPEN = 3      # cuadros en que la letra se abre
K_FLAP = 3      # cuadros de aleteo con las mitades de la letra
K_FADE = 3      # cuadros de fundido hacia el pájaro dibujado


def _bez(p0, p1, p2, p3, e):
    return ((1 - e) ** 3) * p0 + 3 * e * (1 - e) ** 2 * p1 + 3 * e * e * (1 - e) * p2 + e ** 3 * p3


def _poisson(rng, box, n, rmin_fn, tries=8000, avoid=None):
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
    def __init__(self, letter, land, h_land, t_pop, dur, facing, rng, u, away):
        self.letter = letter
        self.color = letter.color
        x0, y0, x1, y1 = letter.box
        self.lh = y1 - y0
        self.pivot = np.array([(x0 + x1) / 2, y1 - 0.12 * self.lh])
        self.p3 = np.array(land, np.float64)
        self.h_land = h_land
        self.t_pop = t_pop
        self.t_open = t_pop + 2.0 / FPS
        self.t_land = t_pop + dur
        self.facing = facing
        self.u = u
        # trayectoria: sale empujada hacia afuera de la onda (lejos del ojo) y planea hacia el potrero
        self.p0 = self.pivot.copy()
        self.p1 = self.p0 + away * rng.uniform(50, 110) * u + np.array([0, -rng.uniform(40, 90)]) * u
        self.p2 = self.p3 + np.array([rng.uniform(-80, 60), -rng.uniform(110, 170)]) * u
        self.phase0 = rng.uniform(0, 2 * np.pi)
        self.freq = rng.uniform(4.0, 4.8)
        self.sleep_t = self.t_land + rng.uniform(0.10, 0.34)
        self.duty = False

    def k(self, t):
        return int(np.floor((t - self.t_pop) * FPS + 1e-6))

    def pos(self, t):
        u = np.clip((t - self.t_open) / max(1e-3, self.t_land - self.t_open), 0, 1)
        e = 1 - (1 - u) ** 1.8
        return _bez(self.p0, self.p1, self.p2, self.p3, e), u

    def scale(self, t):
        """Escala de la figura letra-pájaro: nace del tamaño de la letra y se achica rápido."""
        k = (t - self.t_open) * FPS
        return float(np.clip(1.0 - 0.5 * k / (K_OPEN + K_FLAP + K_FADE), 0.5, 1.0))


def _wing2d(alpha_deg, bend, length):
    """Ala en 2D (estilo sumi-e): siempre muestra su largo. Devuelve (brazo, blanco, mano, contorno)."""
    a = np.deg2rad(alpha_deg)
    n = 14
    sgrid = np.linspace(0, 1, n)
    wprof = 0.07 + 0.15 * np.sin(sgrid * np.pi * 0.6) - 0.9 * np.clip(sgrid - 0.84, 0, 1) ** 1.6
    wprof = np.maximum(wprof, 0.02)
    pts_c = []
    ang = a
    p = np.zeros(2)
    for i in range(n):
        if i:
            if sgrid[i] > 0.5:
                ang = a + np.deg2rad(bend) * (sgrid[i] - 0.5) * 2
            p = p + (length / (n - 1)) * np.array([-np.cos(ang), -np.sin(ang)])
        pts_c.append(p.copy())
    pts_c = np.array(pts_c)
    tang = np.gradient(pts_c, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nor = np.stack([-tang[:, 1], tang[:, 0]], 1)
    lead = pts_c + nor * (wprof[:, None] * length * 0.35)
    trail = pts_c - nor * (wprof[:, None] * length * 0.65)

    def band(s0, s1, edge=lead):
        i0, i1 = int(s0 * (n - 1)), int(np.ceil(s1 * (n - 1)))
        return np.vstack([edge[i0:i1 + 1], trail[i0:i1 + 1][::-1]])
    return band(0.0, 0.62), band(0.22, 0.62, edge=pts_c), band(0.58, 1.0), np.vstack([lead, trail[::-1]])


class Flock:
    def __init__(self, title, lay, rng, t_release=3.00, eye=None, relay_t0=3.16):
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
        duty_pt = np.array(lay["duty_pt"])
        duty_h = lay["duty_h"]

        def avoid(x, y):
            return np.hypot(x - duty_pt[0], (y - duty_pt[1]) * 1.6) < duty_h * 1.3 or \
                bool(lay.get("land_avoid") and lay["land_avoid"](x, y))
        spots = _poisson(rng, box, n - 1, lambda y: hfun(y) * 1.35, avoid=avoid)
        while len(spots) < n - 1:
            spots.append(np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])]))
        spots = sorted(spots, key=lambda p: p[0])
        # Onda de alarma circular desde el ojo de la vigía
        src = np.asarray(eye if eye is not None else (0, 0), np.float64)
        dists = [np.hypot(*(L.center - src)) for L in letters]
        dmin, dmax = min(dists), max(dists)
        # La vigía de turno nace de la primera A de ALERTA
        duty_letter = title.alerta.letters[0]
        others = [L for L in letters if L is not duty_letter]
        by_x = sorted(others, key=lambda L: L.center[0] + rng.normal(0, 40 * u))
        self.birds = []
        for L, sp in [(duty_letter, duty_pt)] + list(zip(by_x, spots)):
            d = np.hypot(*(L.center - src))
            t_pop = t_release + 0.42 * (d - dmin) / max(1.0, dmax - dmin)
            away = (L.center - src) / max(1.0, d)
            is_duty = L is duty_letter
            h = duty_h if is_duty else hfun(sp[1])
            facing = 1 if (is_duty or rng.random() < 0.6) else -1
            b = Bird(L, sp, h, t_pop, 0.9 if is_duty else rng.uniform(0.82, 0.95), facing, rng, u, away)
            b.duty = is_duty
            title.pop_time[id(L)] = t_pop
            self.birds.append(b)
        self.duty = self.birds[0]
        # La posta: trazo de acuarela roja desde ALERTA hasta el ojo de la vigía de turno
        A = title.alerta
        start = np.array([A.letters[1].center[0] - 10 * u, A.box[3] + 8 * u])
        eye_pt, er = self.duty_eye()
        mid1 = start + (eye_pt - start) * 0.33 + np.array([70, 0]) * u
        mid2 = start + (eye_pt - start) * 0.66 + np.array([-55, 0]) * u
        path = catmull_rom(np.array([start, start + np.array([8, 30]) * u, mid1, mid2,
                                     eye_pt + np.array([10, -40]) * u, eye_pt]), samples=24)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.relay_arc = np.concatenate([[0], np.cumsum(seg)])
        self.relay_path = path
        self.relay_start = start
        self.relay_t0 = relay_t0
        self.t_eye = max(self.duty.t_land + 0.10, relay_t0 + 0.72)
        self.drips = [b for b in self.birds if b.color == "red"]

    # --- la vigía de turno ----------------------------------------------------------------------------
    def duty_eye(self):
        b = self.duty
        s = b.h_land / 0.82
        e = np.array(Q.EYE) * np.array([b.facing, 1.0]) * s + b.p3
        return e, max(Q.EYE_R * s * 1.25, 5.2 * self.u)

    # --- ave de pie -----------------------------------------------------------------------------------------
    def _stand(self, masks, b, t):
        s = b.h_land / 0.82
        f = b.facing
        X = lambda pts: np.asarray(pts, np.float64).reshape(-1, 2) * np.array([f * s, s]) + b.p3
        asleep = (t >= b.sleep_t) and not b.duty
        grey, white, ink = masks["grey"], masks["white"], masks["ink"]
        c = ink.ctx
        c.set_line_cap(1)
        lw = max(0.8, 0.012 * s)
        if asleep:
            # dormida: cuerpo en gota (pecho redondo, cola en punta), cabeza hundida sin pico, una pata
            body = catmull_rom(np.array([(0.24, -0.50), (0.20, -0.40), (0.06, -0.345), (-0.12, -0.36),
                                         (-0.28, -0.41), (-0.37, -0.45), (-0.26, -0.50), (-0.08, -0.585),
                                         (0.08, -0.625), (0.19, -0.605)]), 6, closed=True)
            grey.fill_poly(X(body), 1.0)
            white.fill_poly(X(catmull_rom(np.array([(0.22, -0.46), (0.18, -0.39), (0.05, -0.35), (-0.12, -0.365),
                                                    (-0.26, -0.41), (-0.1, -0.43), (0.06, -0.45)]), 5, closed=True)), 1.0)
            ink.fill_poly(X(catmull_rom(np.array([(0.24, -0.50), (0.215, -0.445), (0.16, -0.47), (0.15, -0.56),
                                                  (0.2, -0.6)]), 4, closed=True)), 1.0)          # pechera
            ink.fill_poly(X([(-0.2, -0.47), (-0.37, -0.45), (-0.3, -0.49)]), 1.0)             # primarias
            grey.dot(*X([(0.13, -0.635)])[0], 0.09 * s, 1.0)                                  # cabeza hundida
            cr = X([(0.1, -0.705), (0.02, -0.735), (-0.06, -0.73)])
            c.set_line_width(max(0.7, 0.011 * s))
            ink.set_alpha(1.0)
            c.move_to(*cr[0])
            c.line_to(*cr[1])
            c.line_to(*cr[2])
            c.stroke()
            ink.dot(*X([(0.175, -0.645)])[0], max(0.8, 0.01 * s), 1.0)                          # ojo cerrado
            leg = X([(0.05, -0.35), (0.04, -0.2), (0.05, -0.004)])
            c.set_line_width(lw)
            c.move_to(*leg[0])
            for q in leg[1:]:
                c.line_to(*q)
            c.stroke()
            pb = X(body)
            c.set_line_width(max(0.6, 0.008 * s))
            c.move_to(*pb[0])
            for q in pb[1:]:
                c.line_to(*q)
            c.close_path()
            ink.set_alpha(0.8)
            c.stroke()
            ink.set_alpha(1.0)
            return
        body = catmull_rom(np.array(Q.BODY), 6, closed=True)
        grey.fill_poly(X(body), 1.0)
        white.fill_poly(X(catmull_rom(np.array(Q.BELLY), 5, closed=True)), 1.0)
        for poly in (Q.BIB, Q.PRIMARIES, Q.TAILBAND):
            ink.fill_poly(X(catmull_rom(np.array(poly), 4, closed=True)), 1.0)
        for leg in (Q.LEG_FAR, Q.LEG_NEAR):
            p = X(leg)
            c.set_line_width(lw)
            c.move_to(*p[0])
            for q in p[1:]:
                c.line_to(*q)
            c.stroke()
        c.set_line_width(max(0.6, 0.008 * s) * (1.8 if b.duty else 1.0))
        pb = X(body)
        c.move_to(*pb[0])
        for q in pb[1:]:
            c.line_to(*q)
        c.close_path()
        ink.set_alpha(0.85)
        c.stroke()
        ink.set_alpha(1.0)
        hc = X([Q.HEAD_C])[0]
        grey.dot(hc[0], hc[1], Q.HEAD_R * s, 1.0)
        grey.fill_poly(X([Q.NECK_BACK, Q.HEAD_BACK, Q.HEAD_THROAT, Q.NECK_FRONT]), 1.0)
        ink.fill_poly(X(catmull_rom(np.array(Q.FACE_MASK), 3, closed=True)), 1.0)
        ink.fill_poly(X(Q.BILL), 1.0)
        cr = X(Q.CREST[0])
        c.set_line_width(max(0.7, 0.011 * s))
        c.move_to(*cr[0])
        for q in cr[1:]:
            c.line_to(*q)
        c.stroke()
        e = X([Q.EYE])[0]
        if b.duty:
            _, r = self.duty_eye()
            if t >= self.t_eye:
                g = np.clip((t - self.t_eye) / 0.1, 0, 1)
                masks["white_over"].dot(e[0], e[1], r * (1 + 0.5 * g), 1.0)    # halo de gouache
                ink.dot(e[0], e[1], r * 1.12, 1.0)
                masks["red_over"].dot(e[0], e[1], r * (0.4 + 0.6 * g), 1.0)
                if g >= 1:
                    masks["glint"].dot(e[0] - r * 0.35, e[1] - r * 0.38, r * 0.28, 1.0)
            else:
                ink.dot(e[0], e[1], r * 0.8, 1.0)
        else:
            ink.dot(e[0], e[1], max(0.9, 0.012 * s), 1.0)

    # --- letra que se abre como alas -------------------------------------------------------------------
    def _letter_wings(self, masks, b, t, theta, scale, pos, alpha, red_frac):
        L = b.letter
        x0, y0, x1, y1 = L.box
        pad = 6 * self.u
        piv = b.pivot
        targets = []
        if b.color == "red":
            if red_frac > 0.01:
                targets.append((masks["red_over"], alpha * red_frac))
            if red_frac < 0.99:
                targets.append((masks["ink"], alpha * (1 - red_frac)))
        else:
            targets.append((masks["ink"], alpha))
        for m, a in targets:
            ctx = m.ctx
            for side in (-1, 1):
                ctx.save()
                ctx.translate(pos[0], pos[1])
                ctx.scale(scale, scale)
                ctx.rotate(side * theta)
                ctx.translate(-piv[0], -piv[1])
                if side < 0:
                    ctx.rectangle(x0 - pad, y0 - pad, piv[0] - (x0 - pad), (y1 - y0) + 2 * pad)
                else:
                    ctx.rectangle(piv[0], y0 - pad, (x1 + pad) - piv[0], (y1 - y0) + 2 * pad)
                ctx.clip()
                L.draw(ctx)
                m.set_alpha(a)
                ctx.fill()
                ctx.restore()
        # cuerpo pequeño que aparece entre las alas
        body_a = alpha if (t - b.t_open) * FPS >= 0.99 else 0.0
        if body_a > 0.02:
            r = 0.2 * b.lh * scale
            f = b.facing
            a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            body = np.stack([pos[0] + f * r * 0.9 * np.cos(a), pos[1] + r * 0.32 * np.sin(a)], 1)
            masks["grey"].fill_poly(body, body_a)
            masks["white"].fill_poly(np.stack([pos[0] + f * r * 0.7 * np.cos(a[3:17]),
                                               pos[1] + r * 0.32 * np.abs(np.sin(a[3:17]))], 1), body_a)
            ink = masks["ink"]
            ink.dot(pos[0] + f * r * 0.95, pos[1] - r * 0.18, r * 0.27, body_a)
            ink.fill_poly([(pos[0] + f * r * 1.18, pos[1] - r * 0.24), (pos[0] + f * r * 1.55, pos[1] - r * 0.16),
                           (pos[0] + f * r * 1.18, pos[1] - r * 0.1)], body_a)
            ink.fill_poly([(pos[0] - f * r * 0.8, pos[1] - r * 0.08), (pos[0] - f * r * 1.3, pos[1]),
                           (pos[0] - f * r * 0.8, pos[1] + r * 0.1)], body_a)

    # --- vuelo con alas dibujadas -------------------------------------------------------------------------
    def _fly(self, masks, b, t, alpha=1.0):
        p, u = b.pos(t)
        p_prev, _ = b.pos(t - 1.0 / FPS)
        v = p - p_prev
        _, uu = b.pos(t)
        span0 = b.lh * 1.1 * 0.58
        span = span0 + (b.h_land * 1.9 - span0) * min(1.0, uu / 0.85) ** 0.8
        f = 1 if v[0] >= 0 else -1
        ang = np.arctan2(v[1], abs(v[0]) + 1e-6)
        ang = float(np.clip(ang, np.deg2rad(-35), np.deg2rad(40)))
        phase = b.phase0 + 2 * np.pi * b.freq * (t - b.t_open)
        pose = np.round((phase % (2 * np.pi)) / (2 * np.pi) * 8) % 8     # 8 poses dibujadas
        ph = pose / 8 * 2 * np.pi
        wa = 35 + 62 * np.cos(ph)
        bend = -38 if np.sin(ph) > 0 else 8
        landing = u > 0.82
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
        far = _wing2d(wa + 14, bend, L * 0.78)
        grey.fill_poly(T(far[0] + sh, False), alpha)
        ink.fill_poly(T(far[2] + sh, False), alpha * 0.95)
        a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        body = np.stack([0.14 * np.cos(a), 0.042 * np.sin(a)], 1)
        grey.fill_poly(T(body), alpha)
        white.fill_poly(T(np.stack([0.11 * np.cos(a[4:16]), 0.042 * np.abs(np.sin(a[4:16]))], 1)), alpha * 0.95)
        ink.fill_poly(T([(0.09, -0.028), (0.16, -0.03), (0.17, 0.018), (0.09, 0.03)]), alpha)
        ink.dot(*T([(0.155, -0.024)])[0], 0.036 * span, alpha)
        ink.fill_poly(T([(0.18, -0.03), (0.23, -0.022), (0.18, -0.014)]), alpha)
        ink.fill_poly(T([(-0.12, -0.018), (-0.2, -0.006), (-0.2, 0.01), (-0.12, 0.02)]), alpha)
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
        near = _wing2d(wa, bend, L)
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

    # --- dibujo por cuadro --------------------------------------------------------------------------------
    def draw(self, masks, t):
        for b in self.birds:
            if t < b.t_open:
                continue
            if t >= b.t_land:
                self._stand(masks, b, t)
                continue
            p, u = b.pos(t)
            ko = b.k(t) - 2
            # rojo solo los 2 primeros cuadros: luego escurre (ver _relay) y el ala queda en tinta
            red_frac = 1.0 if (b.color == "red" and ko < 2) else 0.0
            if ko < 3:                               # se parte por su eje y se abre en V
                th = np.deg2rad(42) * (1 - (1 - (ko + 1) / 3) ** 2)
                self._letter_wings(masks, b, t, th, 0.8 - 0.07 * ko, p, 1.0, red_frac)
            elif ko < 5:                             # un aletazo con las mitades de la letra
                th = np.deg2rad(90 if ko == 3 else 45)
                self._letter_wings(masks, b, t, th, 0.58, p, 1.0, 0.0)
            else:                                    # ya es un queltehue (reemplazo seco, sin fundido)
                self._fly(masks, b, t)
        self._relay(masks, t)

    # --- la posta del rojo ----------------------------------------------------------------------------------
    @staticmethod
    def _ribbon(m, pts, widths):
        """Cinta de ancho variable (trazo de pincel continuo) con extremos redondos."""
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2:
            return
        tang = np.gradient(pts, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nor = np.stack([-tang[:, 1], tang[:, 0]], 1)
        w = np.asarray(widths)[:, None] / 2
        m.fill_poly(np.vstack([pts + nor * w, (pts - nor * w)[::-1]]), 1.0)
        m.dot(pts[0, 0], pts[0, 1], widths[0] / 2, 1.0)
        m.dot(pts[-1, 0], pts[-1, 1], widths[-1] / 2, 1.0)

    def _relay(self, masks, t):
        m = masks["relay"]
        u = self.u
        # goteo: el rojo de cada letra de ALERTA escurre hacia el inicio del trazo
        for b in self.drips:
            if b.duty:
                continue
            d0 = b.t_open
            if not (d0 <= t < self.relay_t0 + 0.2):
                continue
            e = np.clip((t - d0) / 0.22, 0, 1)
            a = np.array([b.letter.center[0], b.letter.box[3] - 4 * u])
            z = self.relay_start
            head = a + (z - a) * (1 - (1 - e) ** 2)
            tail = a + (z - a) * max(0.0, e - 0.5) * 2 * 0.9
            mid = (head + tail) / 2 + np.array([0, 6 * u])
            pts = np.array([tail, mid, head])
            pts = np.stack([np.interp(np.linspace(0, 2, 16), [0, 1, 2], pts[:, 0]),
                            np.interp(np.linspace(0, 2, 16), [0, 1, 2], pts[:, 1])], 1)
            self._ribbon(m, pts, np.linspace(2.0, 8.0, 16) * u)
        if t < self.relay_t0 or t > self.t_eye + 0.02:
            return
        # trazo principal: la cabeza avanza, la cola se reabsorbe
        e = np.clip((t - self.relay_t0) / (self.t_eye - self.relay_t0), 0, 1)
        e = e * e * (3 - 2 * e) * 0.3 + e * 0.7
        total = self.relay_arc[-1]
        s_head = e * total
        s_tail = max(0.0, s_head - 0.42 * total)
        idx = (self.relay_arc >= s_tail) & (self.relay_arc <= s_head)
        pts = self.relay_path[idx]
        if len(pts) < 2:
            return
        n = len(pts)
        w = (1.5 + 7.0 * np.linspace(0, 1, n) ** 1.2) * u
        self._ribbon(m, pts, w)
        hq = pts[-1]
        m.dot(hq[0], hq[1], 6.5 * u * (1 - 0.35 * e), 1.0)   # gota en la punta
