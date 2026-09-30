"""La bandada: cada letra del título se vuelve un queltehue que vuela y se posa en el potrero.

Metamorfosis (por letra, en una onda circular que sale del ojo de la vigía):
  k 0–1  la letra se aprieta (lo dibuja title.py);
  k 2–3  se parte por su eje y las mitades se abren en V (20°, 40° desde la vertical), en su plancha
         riso (grano y color);
  k 4    un aletazo: las mitades bajan en Λ;
  k 5    sustitución seca por el queltehue dibujado, del mismo tamaño y con las alas arriba.
Vuelo: ciclo de 4 poses «en dos» (V, horizontal, Λ, horizontal); alas anchas y redondeadas, brazo pardo
con borde de fuga negro, franja blanca y mano negra. Seis cuadros antes de posarse: alas arriba en V
mostrando el blanco y patas adelante; al tocar el suelo, squash (Y 0,9) y las alas se pliegan.
En el suelo: dormidas en tres poses (en una pata, en dos, echadas), a escala según la profundidad y en
grupos que se solapan; cada ave se compone por separado, de atrás hacia adelante.
La posta (sin líquido: el rojo es una plancha riso): la «A» de ALERTA se vuelve la vigía de turno y
vuela con su plancha roja encima, en multiply y descuadrada 3 px. Al
posarse, la plancha de la A se contrae hacia el ojo en 6 cuadros y deja solo el ojo rojo, que destella.
Las aves se pintan como la vigía: aguadas transparentes con granulación y borde oscurecido, gouache
blanco y plumilla de grosor variable.
"""
import cv2
import numpy as np

from . import queltehue as Q
from .geometry import catmull_rom
from .ink import Mask, Stroke

FPS = 30.0
K_SUB = 5          # cuadros desde el «pop» hasta la sustitución por el pájaro dibujado
LAND_FRAMES = 6    # cuadros de pose de aterrizaje antes de tocar el suelo
FOLD_FRAMES = 4    # squash (2) + alas que se pliegan (2)
SHRINK_FRAMES = 6  # la plancha roja de la vigía de turno se contrae hacia el ojo

# poses de aleteo: (ángulo del brazo sobre la horizontal, quiebre de la mano, largo relativo)
POSES = [(48, -10, 1.0), (4, 0, 1.0), (-38, 14, 0.95), (16, 34, 0.82)]
POSE_LAND = (70, -6, 1.0)


def _bez(p0, p1, p2, p3, e):
    return ((1 - e) ** 3) * p0 + 3 * e * (1 - e) ** 2 * p1 + 3 * e * e * (1 - e) * p2 + e ** 3 * p3


def _rot(pts, a):
    c, s = np.cos(a), np.sin(a)
    return np.asarray(pts, np.float64) @ np.array([[c, -s], [s, c]]).T


class Stamp:
    """Capas de una sola ave, del tamaño de su caja; se dibuja en coordenadas de pantalla."""
    LAYERS = ("back", "head", "white", "ink", "red")

    def __init__(self, box, W, H, single=False, shift=(0.0, 0.0)):
        x0, y0, x1, y1 = box
        self.x0, self.y0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
        self.x1, self.y1 = int(min(W, np.ceil(x1))), int(min(H, np.ceil(y1)))
        self.ok = self.x1 - self.x0 > 1 and self.y1 - self.y0 > 1
        self.m = {}
        if self.ok:
            if single:   # una sola plancha (silueta completa), p. ej. la plancha roja descuadrada
                m = Mask(self.x1 - self.x0, self.y1 - self.y0)
                m.ctx.translate(-self.x0 + shift[0], -self.y0 + shift[1])
                for k in self.LAYERS:
                    self.m[k] = m
            else:
                for k in self.LAYERS:
                    m = Mask(self.x1 - self.x0, self.y1 - self.y0)
                    m.ctx.translate(-self.x0, -self.y0)
                    self.m[k] = m

    def composite_glaze(self, img, color, k, grain, alpha=1.0):
        """La plancha como tinta transparente (multiply), con grano de papel."""
        if not self.ok:
            return
        a = self.m["back"].array()
        if not a.any():
            return
        reg = img[self.y0:self.y1, self.x0:self.x1]
        g = grain[self.y0:self.y1, self.x0:self.x1]
        a = np.clip(a * alpha * (0.78 + 0.22 * g), 0, 1)
        reg *= np.exp(a[..., None] * k * np.log(np.clip(color, 0.01, 1))[None, None, :])

    def composite(self, img, colors, grain, tex=None):
        """Aguadas transparentes (lomo y cabeza), gouache opaco (blanco), tinta (negro) y rojo.

        tex = dict(ph=altura del papel, dx, dy=desplazamiento de borde irregular, pigments={capa: color})"""
        if not self.ok:
            return
        y0, y1, x0, x1 = self.y0, self.y1, self.x0, self.x1
        reg = img[y0:y1, x0:x1]
        g = grain[y0:y1, x0:x1]
        gxy = None
        for k in self.LAYERS:
            a = self.m[k].array()
            if not a.any():
                continue
            if tex is not None and k in ("back", "head", "white"):
                if gxy is None:
                    hh, ww = a.shape
                    gy_, gx_ = np.mgrid[0:hh, 0:ww].astype(np.float32)
                    gxy = (gx_ + tex["dx"][y0:y1, x0:x1], gy_ + tex["dy"][y0:y1, x0:x1])
                a = cv2.remap(a, gxy[0], gxy[1], cv2.INTER_LINEAR)
            if tex is not None and k in tex["pigments"]:
                # aguada: granulación en los valles del papel y borde que se oscurece al secar
                ph = tex["ph"][y0:y1, x0:x1]
                rim = np.clip(a - cv2.GaussianBlur(a, (0, 0), 1.3), 0, 1)
                dens = np.clip(a * (0.80 + 0.40 * (1 - ph)) * (0.9 + 0.2 * g) + 0.9 * rim, 0, 1.4)
                reg *= np.exp(dens[..., None] * 0.95 * np.log(np.clip(tex["pigments"][k], 0.02, 1))[None, None, :])
                continue
            if k == "white":
                a = a * (0.78 + 0.22 * g)          # gouache: el grano del papel asoma
            aa = np.clip(a, 0, 1)[..., None]
            reg *= (1 - aa)
            reg += colors[k][None, None, :] * aa


class Bird:
    def __init__(self, letter, land, h_land, t_pop, dur, facing, rng, u, away, pose, duty=False):
        self.letter = letter
        self.color = letter.color
        x0, y0, x1, y1 = letter.box
        self.lh = y1 - y0
        self.lw = x1 - x0
        self.pivot = np.array([(x0 + x1) / 2, y1 - 0.06 * self.lh])
        self.feet = np.array(land, np.float64)
        self.h_land = h_land
        self.t_pop = t_pop
        self.t_sub = t_pop + K_SUB / FPS
        self.t_land = self.t_sub + dur
        self.facing = facing
        self.u = u
        self.away = away
        self.pose = pose
        self.duty = duty
        self.phase0 = int(rng.integers(0, 4))
        self.lw_k = rng.uniform(0.75, 1.3)       # grosor de contorno propio de cada ave (±0,5 px)
        self.sleep_t = self.t_land + FOLD_FRAMES / FPS + rng.uniform(0.02, 0.05)
        # trayectoria del centro del cuerpo: se eleva y sale hacia afuera; llega planeando de costado
        self.p0 = self.meta_pos(self.t_sub)
        self.p3 = self.feet + np.array([0.0, -0.5 * h_land])
        dirx = np.sign(self.p3[0] - self.p0[0]) or 1.0
        self.p1 = self.p0 + away * rng.uniform(70, 120) * u + np.array([0, -rng.uniform(40, 80)]) * u
        self.p2 = self.p3 + np.array([-dirx * rng.uniform(120, 180), -rng.uniform(40, 70)]) * u
        self.dirx = dirx
        # tamaños (envergadura) al nacer y al posarse
        self.span0 = 1.25 * self.lh
        self.span1 = (1.25 if duty else 1.55) * h_land

    def k(self, t):
        return int(np.floor((t - self.t_pop) * FPS + 1e-6))

    def meta_pos(self, t):
        k = (t - self.t_pop) * FPS
        d = max(0.0, k - 2)
        return self.pivot + self.away * d * 2.0 * self.u + np.array([0, -d * 5.0 * self.u])   # sube 12–18 px

    def flight(self, t):
        u = np.clip((t - self.t_sub) / max(1e-3, self.t_land - self.t_sub), 0, 1)
        e = u * u * (3 - 2 * u) * 0.45 + u * 0.55
        return _bez(self.p0, self.p1, self.p2, self.p3, e), u


class Flock:
    def __init__(self, title, lay, rng, W, H, t_release=3.00, eye=None, spread=0.50, flight=(0.34, 0.40)):
        self.title = title
        self.lay = lay
        self.W, self.H = W, H
        self.u = u = lay["u"]
        letters = list(title.letters)
        n = len(letters)
        box = lay["land_box"]
        hmin, hmax = lay["land_h"]
        y0, y1 = box[1], box[3]
        self.hfun = hfun = lambda y: hmin + (hmax - hmin) * np.clip((y - y0) / (y1 - y0), 0, 1)
        duty_pt = np.array(lay["duty_pt"])
        duty_h = lay["duty_h"]

        # Lugares de aterrizaje: grupos que se solapan (no una grilla pareja)
        n_cl = 5
        cx = np.linspace(box[0] + 0.08 * (box[2] - box[0]), box[2] - 0.06 * (box[2] - box[0]), n_cl)
        cx += rng.normal(0, 0.04 * (box[2] - box[0]), n_cl)
        cy = rng.uniform(y0 + 0.15 * (y1 - y0), y1 - 0.1 * (y1 - y0), n_cl)
        weights = rng.uniform(0.6, 1.4, n_cl)
        weights /= weights.sum()
        spots = []
        tries = 0
        while len(spots) < n - 1 and tries < 20000:
            tries += 1
            c = rng.choice(n_cl, p=weights)
            p = np.array([cx[c] + rng.normal(0, 70 * u), cy[c] + rng.normal(0, 16 * u)])
            if not (box[0] < p[0] < box[2] and y0 < p[1] < y1):
                continue
            if np.hypot(p[0] - duty_pt[0], (p[1] - duty_pt[1]) * 1.6) < duty_h * 1.0:
                continue
            if lay.get("land_avoid") and lay["land_avoid"](*p):
                continue
            h = hfun(p[1])
            if all(np.hypot(*(p - q)) > 0.42 * max(h, hfun(q[1])) for q in spots):
                spots.append(p)
        while len(spots) < n - 1:
            spots.append(np.array([rng.uniform(box[0], box[2]), rng.uniform(y0, y1)]))
        spots = sorted(spots, key=lambda p: p[0])

        # Onda de alarma circular desde el ojo de la vigía
        src = np.asarray(eye if eye is not None else (0, 0), np.float64)
        # onda desde el ojo: orden por distancia, con espaciado parejo (≤ 6 letras abriéndose a la vez)
        order = sorted(range(n), key=lambda i: np.hypot(*(letters[i].center - src)) + rng.normal(0, 6 * u))
        rank = {id(letters[i]): r for r, i in enumerate(order)}
        duty_letter = title.alerta.letters[0]
        others = [L for L in letters if L is not duty_letter]
        by_x = sorted(others, key=lambda L: L.center[0] + rng.normal(0, 40 * u))
        self.birds = []
        poses = ["one_leg", "two_legs", "sitting"]
        for L, sp in [(duty_letter, duty_pt)] + list(zip(by_x, spots)):
            d = np.hypot(*(L.center - src))
            t_pop = t_release + spread * rank[id(L)] / max(1, n - 1)
            away = (L.center - src) / max(1.0, d)
            is_duty = L is duty_letter
            h = duty_h if is_duty else hfun(sp[1]) * rng.uniform(0.9, 1.1)
            facing = 1 if is_duty else (1 if rng.random() < 0.55 else -1)
            dur = 0.50 if is_duty else rng.uniform(*flight)
            pose = "awake" if is_duty else poses[int(rng.integers(0, 3))]
            b = Bird(L, sp, h, t_pop, dur, facing, rng, u, away, pose, duty=is_duty)
            title.pop_time[id(L)] = t_pop
            self.birds.append(b)
        self.duty = self.birds[0]

        # La posta: la plancha roja viaja con la vigía de turno y se contrae hasta su ojo
        self.red_birds = [b for b in self.birds if b.color == "red"]
        self.t_eye = self.duty.t_land + SHRINK_FRAMES / FPS

    def set_textures(self, ph, dx, dy, pigments):
        self.tex = dict(ph=ph, dx=dx, dy=dy, pigments=pigments)

    # --- la vigía de turno ------------------------------------------------------------------------------
    def duty_eye(self):
        b = self.duty
        s = b.h_land / 0.82
        e = np.array(Q.EYE) * np.array([b.facing, 1.0]) * s + b.feet
        return e, 3.8 * self.u          # 7–8 px de diámetro

    # --- metamorfosis: mitades de la letra en las planchas riso ------------------------------------------
    def _halves(self, m, b, pos, theta, sc, flip, alpha=1.0):
        L = b.letter
        x0, y0, x1, y1 = L.box
        pad = 8 * self.u
        piv = b.pivot
        ctx = m.ctx
        for side in (-1, 1):
            ctx.save()
            ctx.translate(pos[0], pos[1])
            ctx.scale(sc, -sc if flip else sc)
            ctx.rotate(side * theta)
            ctx.translate(-piv[0], -piv[1])
            if side < 0:
                ctx.rectangle(x0 - pad, y0 - pad, piv[0] - (x0 - pad), (y1 - y0) + 2 * pad)
            else:
                ctx.rectangle(piv[0], y0 - pad, (x1 + pad) - piv[0], (y1 - y0) + 2 * pad)
            ctx.clip()
            L.draw(ctx)
            m.set_alpha(alpha)
            ctx.fill()
            ctx.restore()

    def draw_letters(self, plates, t):
        for b in self.birds:
            k = b.k(t)
            if t < b.t_pop or k < 2 or k >= K_SUB:
                continue
            pos = b.meta_pos(t)
            flip = k >= 4                                   # k 2–3: se abre en V · k 4: aletazo en Λ
            theta = np.deg2rad(20 if k == 2 else 40)
            names = ("white", "red") if b.color == "red" else ("ink",)
            for name in names:
                self._halves(plates[name], b, pos, theta, 0.75, flip, alpha=0.85 if flip else 1.0)

    # --- dibujo del ave en vuelo (vista frontal-baja: silueta en M) -------------------------------------
    def _wing(self, st, side, theta_deg, flex_deg, length, sp, center, bank):
        """Ala ancha y redondeada. side = -1 (izquierda) / +1 (derecha)."""
        th = np.deg2rad(theta_deg)
        fl = np.deg2rad(flex_deg)
        L1, L2 = 0.21 * sp * length, 0.25 * sp * length
        c0 = 0.13 * sp
        n = 16
        s = np.linspace(0, 1, n)
        pts = []
        for si in s:
            if si <= 0.45:
                d = si / 0.45 * L1
                p = np.array([np.cos(th), np.sin(th)]) * d
            else:
                w = np.array([np.cos(th), np.sin(th)]) * L1
                d = (si - 0.45) / 0.55 * L2
                p = w + np.array([np.cos(th + fl), np.sin(th + fl)]) * d
            pts.append(p)
        pts = np.array(pts)
        body = 0.62 + 0.38 * np.sin(np.clip(s / 0.62, 0, 1) * np.pi / 2)
        tip = np.sqrt(np.clip(1 - ((s - 0.66) / 0.34) ** 2, 0, 1))          # punta redonda
        chord = c0 * np.where(s < 0.66, body, tip)
        tang = np.gradient(pts, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nor = np.stack([-tang[:, 1], tang[:, 0]], 1)       # hacia el borde de ataque (arriba)
        lead = pts + nor * (0.28 * chord)[:, None]
        trail = pts - nor * (0.72 * chord)[:, None]

        def to_screen(q):
            q = q.copy()
            q[:, 0] *= side
            q[:, 1] *= -1                                  # y hacia abajo en pantalla
            q = q + np.array([side * 0.05 * sp, -0.015 * sp])
            return _rot(q, bank) + center

        def band(a, b_, edge_a=lead, edge_b=trail):
            i0, i1 = int(a * (n - 1)), int(np.ceil(b_ * (n - 1)))
            return to_screen(np.vstack([edge_a[i0:i1 + 1], edge_b[i0:i1 + 1][::-1]]))

        mid = pts - nor * (0.42 * chord)[:, None]           # borde de fuga negro (secundarias)
        st.m["back"].fill_poly(band(0.0, 0.46), 1.0)
        st.m["ink"].fill_poly(band(0.0, 0.46, edge_a=mid), 1.0)
        st.m["white"].fill_poly(band(0.40, 0.56), 1.0)
        st.m["ink"].fill_poly(band(0.54, 1.0), 1.0)
        # plumilla de grosor variable: borde de ataque firme, borde de fuga perdido y encontrado
        rng = np.random.default_rng(int(abs(theta_deg) * 7 + length * 100) + (side > 0))
        w = max(0.7, 0.016 * sp)
        Stroke(to_screen(lead), w, rng, pool=0.35, smooth=False, taper=(0.15, 0.35), jitter=0.2).draw(st.m["ink"], 10)
        tr = to_screen(trail)
        Stroke(tr[:int(len(tr) * 0.55)], w * 0.6, rng, pool=0.2, smooth=False, taper=(0.3, 0.5),
               jitter=0.25).draw(st.m["ink"], 10)

    def _fly_bird(self, st, b, t, pose=None, span=None, center=None, legs=False, upright=0.0):
        p, u = b.flight(t)
        center = p if center is None else center
        if span is None:
            e = min(1.0, u / 0.7) ** 0.6
            span = b.span0 * (1 - e) + b.span1 * e
        pp, _ = b.flight(t - 1.0 / FPS)
        v = p - pp
        f = b.dirx if abs(v[0]) < 0.5 else np.sign(v[0])
        bank = float(np.clip(np.arctan2(v[1], abs(v[0]) + 1e-6) * 0.35, np.deg2rad(-12), np.deg2rad(12))) * f
        if pose is None:
            idx = (int(np.floor((t - b.t_sub) * FPS / 2)) + b.phase0) % 4
            pose = POSES[idx]
        th, fl, ln = pose
        # ala lejana, cuerpo, ala cercana
        for side in (-f, f):
            if side == f:
                self._body(st, center, span, f, bank, legs, upright)
            self._wing(st, side, th + (6 if side == -f else 0), fl, ln * (0.92 if side == -f else 1.0), span,
                       center, bank)

    def _body(self, st, center, sp, f, bank, legs, upright):
        a = np.linspace(0, 2 * np.pi, 22, endpoint=False)
        bw, bh = 0.10 * sp * (1 - 0.3 * upright), 0.075 * sp * (1 + 0.5 * upright)
        body = _rot(np.stack([bw * np.cos(a), bh * np.sin(a)], 1), bank) + center
        st.m["back"].fill_poly(body, 1.0)
        belly = _rot(np.stack([0.8 * bw * np.cos(a[1:11]), 0.9 * bh * np.abs(np.sin(a[1:11]))], 1), bank) + center
        st.m["white"].fill_poly(belly, 1.0)
        # cola blanca con banda negra
        tail = _rot(np.array([(-0.03, 0.06), (0.03, 0.06), (0.045, 0.14), (-0.045, 0.14)]) * sp, bank) + center
        st.m["white"].fill_poly(tail, 1.0)
        band = _rot(np.array([(-0.045, 0.115), (0.045, 0.115), (0.047, 0.14), (-0.047, 0.14)]) * sp, bank) + center
        st.m["ink"].fill_poly(band, 1.0)
        # cabeza hacia el sentido de vuelo, con antifaz, pico y cresta
        hc = center + _rot(np.array([[f * 0.07 * sp, -0.07 * sp - 0.05 * sp * upright]]), bank)[0]
        st.m["head"].dot(hc[0], hc[1], 0.05 * sp, 1.0)
        st.m["ink"].fill_poly(_rot(np.array([(f * 0.01, -0.02), (f * 0.05, -0.035), (f * 0.055, 0.02),
                                            (f * 0.02, 0.05)]) * sp, bank) + hc, 1.0)
        st.m["ink"].fill_poly(_rot(np.array([(f * 0.05, -0.012), (f * 0.10, 0.0), (f * 0.05, 0.01)]) * sp, bank) + hc,
                              1.0)
        bib = _rot(np.array([(-0.05, -0.045), (0.05, -0.045), (0.06, -0.005), (-0.06, -0.005)]) * sp, bank) + center
        st.m["ink"].fill_poly(bib, 1.0)
        rng = np.random.default_rng(int(sp * 10) % 997)
        Stroke(body[1:11], max(0.7, 0.014 * sp), rng, pool=0.3, smooth=False,
               taper=(0.2, 0.3), jitter=0.2).draw(st.m["ink"], 10)
        c = st.m["ink"].ctx
        cr = _rot(np.array([(-f * 0.02, -0.04), (-f * 0.06, -0.07), (-f * 0.1, -0.075)]) * sp, bank) + hc
        c.set_line_width(max(0.6, 0.01 * sp))
        c.move_to(*cr[0])
        c.line_to(*cr[1])
        c.line_to(*cr[2])
        c.stroke()
        if legs:
            for dx in (-0.015, 0.02):
                l0 = center + _rot(np.array([[dx * sp, 0.05 * sp]]), bank)[0]
                l1 = l0 + np.array([f * 0.07 * sp, 0.17 * sp])
                c.set_line_width(max(0.7, 0.011 * sp))
                c.move_to(*l0)
                c.line_to(*l1)
                c.stroke()

    # --- ave en el suelo (vista lateral) --------------------------------------------------------------------
    def _stand(self, st, b, t, squash=1.0, wings_fold=None, about=None, silhouette=False):
        f = b.facing
        s = b.h_land / 0.82
        base = b.feet
        if about is not None:          # escala anclada en un punto (la plancha que se contrae al ojo)
            E, kk = about
            base = E + (base - E) * kk
            s = s * kk

        def X(pts):
            q = np.asarray(pts, np.float64).reshape(-1, 2) * np.array([f * s, s * squash])
            return q + base

        asleep = (t >= b.sleep_t) and not b.duty
        back, head, white, ink = st.m["back"], st.m["head"], st.m["white"], st.m["ink"]
        c = ink.ctx
        c.set_line_cap(1)
        lw = max(0.8, 0.012 * s)
        if asleep:
            drop = 0.30 if b.pose == "sitting" else 0.0     # echada: el cuerpo toca el pasto
            body = catmull_rom(np.array([(0.24, -0.50), (0.20, -0.40), (0.06, -0.345), (-0.12, -0.36),
                                         (-0.28, -0.41), (-0.37, -0.45), (-0.26, -0.50), (-0.08, -0.585),
                                         (0.08, -0.625), (0.19, -0.605)]) + np.array([0, drop]), 6, closed=True)
            if b.pose != "sitting":
                legs = [[(0.05, -0.35), (0.04, -0.2), (0.05, -0.004)]]
                if b.pose == "two_legs":
                    legs.append([(-0.01, -0.35), (-0.02, -0.2), (0.0, -0.004)])
                for leg in legs:
                    lp = X(leg)
                    c.set_line_width(lw)
                    c.move_to(*lp[0])
                    for q in lp[1:]:
                        c.line_to(*q)
                    c.stroke()
            back.fill_poly(X(body), 1.0)
            white.fill_poly(X(catmull_rom(np.array([(0.22, -0.46), (0.18, -0.39), (0.05, -0.35), (-0.12, -0.365),
                                                    (-0.26, -0.41), (-0.1, -0.43), (0.06, -0.45)]) + np.array([0, drop]),
                                          5, closed=True)), 1.0)
            ink.fill_poly(X(catmull_rom(np.array([(0.24, -0.50), (0.215, -0.445), (0.16, -0.47), (0.15, -0.56),
                                                  (0.2, -0.6)]) + np.array([0, drop]), 4, closed=True)), 1.0)
            ink.fill_poly(X(np.array([(-0.2, -0.47), (-0.37, -0.45), (-0.3, -0.49)]) + np.array([0, drop])), 1.0)
            hc = X([(0.13, -0.635 + drop)])[0]
            head.dot(hc[0], hc[1], 0.09 * s, 1.0)
            cr = X(np.array([(0.1, -0.705), (0.02, -0.735), (-0.06, -0.73)]) + np.array([0, drop]))
            c.set_line_width(max(0.7, 0.011 * s))
            c.move_to(*cr[0])
            c.line_to(*cr[1])
            c.line_to(*cr[2])
            c.stroke()
            pb = X(body)
            n = len(pb)
            rng = np.random.default_rng(int(b.feet[0] * 13 + b.feet[1]) % 100003)
            Stroke(pb[:int(n * 0.5)], max(0.8, 0.016 * s) * b.lw_k, rng, pool=0.35, smooth=False, taper=(0.2, 0.3),
                   jitter=0.2).draw(ink, 10)
            Stroke(pb[int(n * 0.6):int(n * 0.9)], max(0.6, 0.009 * s) * b.lw_k, rng, pool=0.2, smooth=False,
                   taper=(0.3, 0.4), jitter=0.2).draw(ink, 10)
            return
        body = catmull_rom(np.array(Q.BODY), 6, closed=True)
        for leg in (() if silhouette else (Q.LEG_FAR, Q.LEG_NEAR)):
            p = X(leg)
            c.set_line_width(lw)
            c.move_to(*p[0])
            for q in p[1:]:
                c.line_to(*q)
            c.stroke()
        back.fill_poly(X(body), 1.0)
        white.fill_poly(X(catmull_rom(np.array(Q.BELLY), 5, closed=True)), 1.0)
        for poly in (Q.BIB, Q.PRIMARIES, Q.TAILBAND):
            ink.fill_poly(X(catmull_rom(np.array(poly), 4, closed=True)), 1.0)
        head.fill_poly(X(catmull_rom(np.array(Q.HEAD), 3, closed=True)), 1.0)
        head.fill_poly(X([Q.NECK_BACK, Q.HEAD_BACK, Q.HEAD_THROAT, Q.NECK_FRONT]), 1.0)
        ink.fill_poly(X(catmull_rom(np.array(Q.FACE_MASK), 3, closed=True)), 1.0)
        ink.fill_poly(X(Q.BILL), 1.0)
        cr = X(Q.CREST[0])
        c.set_line_width(max(0.7, 0.011 * s))
        c.move_to(*cr[0])
        for q in cr[1:]:
            c.line_to(*q)
        c.stroke()
        pb = X(body)
        n = len(pb)
        rng = np.random.default_rng(int(b.feet[0] * 13 + b.feet[1]) % 100003)
        wk = 1.6 if b.duty else b.lw_k
        # vientre y pecho en sombra: trazo grueso; lomo a la luz: trazo fino y cortado
        Stroke(pb[int(n * 0.02):int(n * 0.55)], max(0.8, 0.017 * s) * wk, rng, pool=0.4, smooth=False,
               taper=(0.15, 0.3), jitter=0.2).draw(ink, 10)
        Stroke(pb[int(n * 0.62):int(n * 0.95)], max(0.6, 0.009 * s) * wk, rng, pool=0.25, smooth=False,
               taper=(0.3, 0.4), jitter=0.25).draw(ink, 10)
        if wings_fold is not None:
            # alas aún abiertas en V que se pliegan sobre el lomo
            th, ln = wings_fold
            sp = 1.5 * b.h_land * (about[1] if about is not None else 1.0)
            sh = X([(-0.02, -0.62)])[0]
            for side in (-f, f):
                self._wing(st, side, th + (6 if side == -f else 0), -6, ln, sp, sh, 0.0)
        if silhouette:
            return
        e = X([Q.EYE])[0]
        if b.duty:
            _, r = self.duty_eye()
            if t >= self.t_eye:
                k = int(np.floor((t - self.t_eye) * FPS + 1e-6))
                fl = 1.4 if k < 3 else (1.2 if k == 3 else 1.0)   # destello de 3 cuadros y uno de sostén
                ink.dot(e[0], e[1], r * fl + 1.0 * self.u, 1.0)   # contorno oscuro de 1 px
                st.m["red"].dot(e[0], e[1], r * fl, 1.0)
                ink.dot(e[0] + 0.5 * self.u, e[1] + 0.3 * self.u, r * 0.34 * fl, 1.0)
                st.m["white"].dot(e[0] - r * 0.4, e[1] - r * 0.42, 0.9 * self.u, 1.0)   # brillo de 1 px
            else:
                ink.dot(e[0], e[1], max(1.2, 0.012 * s), 1.0)
        else:
            ink.dot(e[0], e[1], max(0.9, 0.012 * s), 1.0)

    # --- composición de todas las aves ------------------------------------------------------------------
    def render(self, img, t, colors, grain):
        W, H = self.W, self.H
        todo = []
        for b in self.birds:
            if t < b.t_sub:
                continue
            if t < b.t_land:
                p, u = b.flight(t)
                depth = p[1] - 1000 * (1 if u < 0.9 else 0)
                todo.append((depth, b, "fly"))
            else:
                todo.append((b.feet[1], b, "stand"))
        todo.sort(key=lambda x: x[0])
        mis = np.array([3.0, 2.0]) * self.u               # plancha roja descuadrada ~3 px
        for _, b, kind in todo:
            if kind == "fly":
                p, u = b.flight(t)
                e = min(1.0, u / 0.7) ** 0.6
                sp = b.span0 * (1 - e) + b.span1 * e
                box = (p[0] - sp, p[1] - sp, p[0] + sp, p[1] + sp)
                n_left = (b.t_land - t) * FPS
                land = n_left <= LAND_FRAMES
                for red in ((False, True) if b.duty else (False,)):
                    st = Stamp(box, W, H, single=red, shift=mis if red else (0, 0))
                    if not st.ok:
                        continue
                    if land:                          # erguida, alas arriba, patas adelante
                        self._fly_bird(st, b, t, pose=POSE_LAND, legs=not red, upright=1.0)
                    else:
                        self._fly_bird(st, b, t)
                    if red:
                        st.composite_glaze(img, colors["red"], 0.95, grain)
                    else:
                        st.composite(img, colors, grain, getattr(self, "tex", None))
            else:
                k = int(np.floor((t - b.t_land) * FPS + 1e-6))
                s = b.h_land / 0.82
                box = (b.feet[0] - 1.2 * s, b.feet[1] - 1.3 * s, b.feet[0] + 1.2 * s, b.feet[1] + 0.1 * s)
                st = Stamp(box, W, H)
                if not st.ok:
                    continue
                if k < 2:
                    kw = dict(squash=0.9, wings_fold=(70, 1.0))
                elif k < FOLD_FRAMES:
                    j = k - 2
                    kw = dict(wings_fold=(70 - 30 * (j + 1), 0.75 - 0.3 * j))
                else:
                    kw = {}
                self._stand(st, b, t, **kw)
                st.composite(img, colors, grain, getattr(self, "tex", None))
                if b.duty and k < SHRINK_FRAMES:
                    # la plancha roja se contrae hacia el ojo (escala 1 → 0,08, anclada en el ojo)
                    E, _ = self.duty_eye()
                    kk = 1.0 - 0.92 * ((k + 1) / SHRINK_FRAMES) ** 1.3
                    rs = Stamp(box, W, H, single=True, shift=mis * kk)
                    if rs.ok:
                        self._stand(rs, b, t, about=(E, kk), silhouette=True, **kw)
                        rs.composite_glaze(img, colors["red"], 0.95, grain)

    def draw_relay(self, m, t):
        """Sin líquido: la posta ya no gotea (ver render)."""
        return
