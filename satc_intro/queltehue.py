"""fig. 1 — Queltehue (*Vanellus chilensis*), la vigía.

Ilustración de lámina naturalista construida a mano con curvas: silueta, aguadas (gris pardo del
dorso, blanco de gouache del vientre, negro del babero y la cara), plumilla de contorno con líneas
«perdidas y encontradas», sombreado de plumas y el ojo rojo: el único rojo de la lámina.

Coordenadas locales en «unidades de ave»: x hacia la derecha (el ave mira a la derecha), y hacia
abajo como en pantalla; los pies están en (0, 0) y la cresta llega a y ≈ -1.
"""
import cv2
import numpy as np

from .color import lin
from .geometry import catmull_rom
from .ink import Mask, Stroke
from .noise import smoothstep, fbm


def _curve(pts, samples=10, closed=False):
    return catmull_rom(np.asarray(pts, np.float64), samples=samples, closed=closed)


# --- Geometría (unidades de ave) --------------------------------------------------------------
BODY = [(0.205, -0.662), (0.235, -0.615), (0.248, -0.56), (0.238, -0.50), (0.20, -0.44),
        (0.13, -0.385), (0.04, -0.352), (-0.06, -0.35), (-0.15, -0.37), (-0.23, -0.40),
        (-0.29, -0.425), (-0.312, -0.44), (-0.29, -0.456), (-0.20, -0.49), (-0.09, -0.545),
        (0.02, -0.595), (0.10, -0.635), (0.145, -0.66)]
BELLY = [(0.24, -0.515), (0.20, -0.44), (0.13, -0.385), (0.04, -0.352), (-0.06, -0.35),
         (-0.15, -0.37), (-0.23, -0.40), (-0.27, -0.418), (-0.18, -0.428), (-0.08, -0.442),
         (0.02, -0.472), (0.08, -0.50), (0.13, -0.518), (0.19, -0.505)]
WING = [(0.11, -0.63), (0.092, -0.57), (0.05, -0.51), (-0.02, -0.47), (-0.10, -0.444),
        (-0.20, -0.428), (-0.30, -0.44), (-0.318, -0.447), (-0.29, -0.457), (-0.20, -0.49),
        (-0.09, -0.545), (0.02, -0.595), (0.09, -0.628)]
PRIMARIES = [(-0.13, -0.436), (-0.22, -0.428), (-0.318, -0.447), (-0.25, -0.466), (-0.16, -0.472)]
SHOULDER = [(0.10, -0.615), (0.072, -0.565), (0.012, -0.525), (-0.05, -0.52), (0.0, -0.57),
            (0.06, -0.605)]
TAILBAND = [(-0.24, -0.412), (-0.30, -0.43), (-0.31, -0.442), (-0.25, -0.44)]
BIB = [(0.212, -0.672), (0.236, -0.62), (0.249, -0.565), (0.241, -0.515), (0.19, -0.498),
       (0.13, -0.515), (0.103, -0.568), (0.12, -0.628), (0.16, -0.664)]
LEG_NEAR = [(0.062, -0.36), (0.045, -0.27), (0.03, -0.185), (0.04, -0.09), (0.05, -0.004)]
LEG_FAR = [(0.016, -0.355), (0.002, -0.268), (-0.012, -0.19), (-0.006, -0.095), (0.0, -0.006)]
TOES_NEAR = [[(0.05, -0.004), (0.13, 0.002)], [(0.05, -0.004), (0.11, 0.013)],
             [(0.05, -0.004), (0.024, 0.004)]]
TOES_FAR = [[(0.0, -0.006), (0.075, -0.011)], [(0.0, -0.006), (-0.022, -0.006)]]
NECK_STUB = [(0.15, -0.655), (0.152, -0.676), (0.203, -0.68), (0.205, -0.666)]

# Cabeza y cuello (se transforman con la pose). Pivote en la base del cuello.
PIVOT = np.array([0.175, -0.662])
HEAD_C = np.array([0.200, -0.748])
HEAD_R = 0.090
BILL_UP = [(0.284, -0.762), (0.368, -0.747), (0.366, -0.742), (0.30, -0.748)]
BILL_LO = [(0.30, -0.748), (0.362, -0.743), (0.354, -0.738), (0.286, -0.735)]
BILL_HINGE = np.array([0.288, -0.745])
BILL = [(0.284, -0.762), (0.368, -0.747), (0.354, -0.738), (0.286, -0.735)]
EYE = np.array([0.236, -0.766])
EYE_R = 0.0145
CREST = [[(0.16, -0.822), (0.11, -0.850), (0.05, -0.868), (-0.005, -0.872), (-0.05, -0.863)],
         [(0.165, -0.818), (0.115, -0.840), (0.07, -0.852), (0.03, -0.852)]]
FACE_MASK = [(0.290, -0.762), (0.285, -0.784), (0.273, -0.803), (0.256, -0.818), (0.236, -0.826),
             (0.232, -0.812), (0.247, -0.800), (0.260, -0.785), (0.268, -0.766), (0.271, -0.748),
             (0.266, -0.727), (0.254, -0.705), (0.238, -0.687), (0.224, -0.673), (0.246, -0.667),
             (0.264, -0.683), (0.279, -0.703), (0.288, -0.725), (0.290, -0.737)]
WHITE_BORDER = [(0.222, -0.838), (0.214, -0.814), (0.230, -0.797), (0.244, -0.783), (0.250, -0.766)]
NECK_BACK = (0.132, -0.655)    # base del cuello (dorso)
NECK_FRONT = (0.218, -0.668)   # base del cuello (pecho)
HEAD_BACK_A = np.deg2rad(152)  # ángulo en la cabeza donde nace la nuca (abajo-atrás)
HEAD_THROAT_A = np.deg2rad(62)
# compatibilidad con la bandada (aves pequeñas)
HEAD_BACK = (0.115, -0.779)
HEAD_THROAT = (0.242, -0.668)
FOREHEAD = FACE_MASK[:10]
THROAT = FACE_MASK[9:]


class Queltehue:
    """La vigía. El cuerpo es un sprite fijo; cabeza, cuello, cresta, pico y ojo se recalculan por cuadro.

    Todo se pinta sobre el papel limpio (reserva) y luego se compone sobre el paisaje, como hace
    una ilustradora que deja el ave en blanco al pintar el fondo.
    """

    def __init__(self, feet_xy, size, seed=4, t_alert=0.70, t_eye=0.80, t_shout=1.40, t_shout_end=1.72,
                 t_sleep=4.18):
        self.fx, self.fy = feet_xy
        self.S = size
        self.seed = seed
        self.t_alert, self.t_eye, self.t_sleep = t_alert, t_eye, t_sleep
        self.t_shout, self.t_shout_end = t_shout, t_shout_end
        s = size
        self.bbox = (int(self.fx - 0.50 * s), int(self.fy - 0.98 * s), int(self.fx + 0.52 * s),
                     int(self.fy + 0.06 * s))
        x0, y0, x1, y1 = self.bbox
        self.w, self.h = x1 - x0, y1 - y0
        self.mask = Mask(self.w, self.h)
        r = np.random.default_rng(seed)
        self.tex = fbm((self.h, self.w), 16, r, octaves=4)
        self.tex2 = fbm((self.h, self.w), 5, r, octaves=3)
        self.dx = (fbm((self.h, self.w), 9, r, octaves=3) - 0.5) * 2.6
        self.dy = (fbm((self.h, self.w), 9, r, octaves=3) - 0.5) * 2.6
        self._gx, self._gy = np.meshgrid(np.arange(self.w, dtype=np.float32),
                                         np.arange(self.h, dtype=np.float32))
        self.u = self.S / 480.0  # escala de grosor de línea

    # --- pose ---------------------------------------------------------------------------------------
    def to_screen(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        x0, y0 = self.bbox[:2]
        return np.stack([self.fx - x0 + pts[:, 0] * self.S, self.fy - y0 + pts[:, 1] * self.S], 1)

    def head_angle(self, t):
        relaxed, alert = np.deg2rad(14), np.deg2rad(-6)
        if t < self.t_alert - 0.1:
            a = relaxed
        elif t < self.t_alert:  # anticipación: se agacha un poco más
            u = (t - (self.t_alert - 0.1)) / 0.1
            a = relaxed + np.deg2rad(6) * np.sin(u * np.pi / 2)
        else:
            u = t - self.t_alert
            a = alert + (relaxed + np.deg2rad(6) - alert) * np.exp(-u * 10.0) * np.cos(u * 24.0)
        if t > self.t_sleep:  # vuelve a descansar
            v = np.clip((t - self.t_sleep) / 0.3, 0, 1)
            v = v * v * (3 - 2 * v)
            a = a + (relaxed - a) * v
        return a

    def shout(self, t):
        """0..1: grito (cabeza hacia adelante y pico abierto), con anticipación."""
        a, b = self.t_shout, self.t_shout_end
        if t < a - 0.07 or t > b + 0.12:
            return 0.0, 0.0
        if t < a:  # anticipación: recoge la cabeza
            return -0.35 * np.sin((t - (a - 0.07)) / 0.07 * np.pi / 2), 0.0
        if t <= b:
            u = np.clip((t - a) / 0.05, 0, 1)
            return 1 - (1 - u) ** 3, u
        v = np.clip((t - b) / 0.12, 0, 1)
        return 1 - v * v * (3 - 2 * v), 1 - v

    def neck_stretch(self, t):
        a = self.head_angle(t)
        return 1.0 + 0.10 * np.clip((np.deg2rad(14) - a) / np.deg2rad(20), -0.3, 1.3)

    def head_tf(self, t):
        a = self.head_angle(t)
        st = self.neck_stretch(t)
        fwd, _ = self.shout(t)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        lift = np.array([0.008, -0.034]) * (st - 1.0) / 0.10 + np.array([0.034, -0.004]) * fwd

        def f(pts):
            p = np.asarray(pts, np.float64).reshape(-1, 2) - PIVOT
            return p @ R.T + PIVOT + lift
        return f

    def crest_erect(self, t):
        u = np.clip((t - self.t_alert) / 0.22, 0, 1)
        e = 1 - (1 - u) ** 3
        if t > self.t_sleep:
            e *= 1 - np.clip((t - self.t_sleep) / 0.3, 0, 1)
        return e

    def eye_open(self, t):
        if t >= self.t_sleep + 0.08:
            return float(np.clip(1 - (t - self.t_sleep - 0.08) / 0.12, 0, 1))
        return float(np.clip((t - self.t_eye) / 0.06, 0, 1))

    # --- utilidades de pintura ------------------------------------------------------------------------
    def _fill(self, pts_screen, alpha=1.0):
        self.mask.clear()
        self.mask.fill_poly(pts_screen, alpha)
        return self.mask.array()

    def _wc(self, m, edge=0.5, rough=True, tex=0.32):
        """Relleno plano -> mancha de acuarela: borde irregular, oscurecido y pigmento irregular."""
        if rough:
            m = cv2.remap(m, self._gx + self.dx, self._gy + self.dy, cv2.INTER_LINEAR)
        rim = np.clip(m - cv2.GaussianBlur(m, (0, 0), 1.8 * max(self.u, 0.6)), 0, 1)
        return np.clip(m * ((1 - tex * 0.6) + tex * self.tex) + edge * rim * 1.8, 0, 1.4)

    @staticmethod
    def _glaze(reg, m, col, k):
        reg *= np.exp(m[..., None] * k * np.log(np.clip(col, 0.02, 1))[None, None, :])

    @staticmethod
    def _over(reg, m, col, a=1.0):
        aa = np.clip(m * a, 0, 1)[..., None]
        reg *= (1 - aa)
        reg += col[None, None, :] * aa

    def _strokes_mask(self, strokes, t=10.0, transform=None):
        m = self.mask
        m.clear()
        for st in strokes:
            st.draw(m, t, transform=transform)
        a = m.array()
        return cv2.remap(a, self._gx + self.dx * 0.3, self._gy + self.dy * 0.3, cv2.INTER_LINEAR)

    def _S(self, pts, w, rng, **kw):
        kw.setdefault("pool", 0.3)
        return Stroke(self.to_screen(pts), w * self.u, rng, **kw)

    # --- cuerpo fijo ----------------------------------------------------------------------------------
    def body_sprite(self, paper_rgb, paper_h, ink, white):
        x0, y0, x1, y1 = self.bbox
        reg = paper_rgb[y0:y1, x0:x1].copy()
        ph = paper_h[y0:y1, x0:x1]
        rng = np.random.default_rng(self.seed + 1)
        grey = lin("#8a857b")
        brown = lin("#7d7465")
        sepia = lin("#6b5541")
        bronze = lin("#5a5334")
        cover = np.zeros((self.h, self.w), np.float32)

        # sombra en el suelo
        ang = np.linspace(0, 2 * np.pi, 48)
        sh = self.to_screen(np.stack([0.03 + 0.30 * np.cos(ang), 0.004 + 0.026 * np.sin(ang)], 1))
        m_sh = cv2.GaussianBlur(self._fill(sh), (0, 0), 5 * self.u)
        self._glaze(reg, m_sh, sepia, 0.6)
        cover = np.maximum(cover, m_sh * 0.9)

        m_body = self._fill(self.to_screen(_curve(BODY, 10, closed=True)))
        m_belly = self._fill(self.to_screen(_curve(BELLY, 8, closed=True)))
        m_wing = self._fill(self.to_screen(_curve(WING, 8, closed=True)))
        cover = np.maximum(cover, m_body)
        self._glaze(reg, self._wc(m_body, 0.55), brown, 0.95)
        dry = 0.80 + 0.20 * smoothstep(0.3, 0.62, ph)
        self._over(reg, self._wc(m_belly, 0.0) * dry, white, 0.94)
        m_bshadow = self._fill(self.to_screen(_curve([(0.20, -0.47), (0.10, -0.39), (-0.05, -0.35),
                                                      (-0.20, -0.375), (-0.30, -0.41), (-0.10, -0.40),
                                                      (0.08, -0.42)], 6, closed=True)))
        self._glaze(reg, cv2.GaussianBlur(m_bshadow, (0, 0), 4 * self.u) * m_belly, grey, 0.55)
        self._glaze(reg, self._wc(m_wing, 0.7), sepia, 0.55)
        self._glaze(reg, self._wc(self._fill(self.to_screen(_curve(SHOULDER, 6, closed=True))), 0.35),
                    bronze, 0.9)
        # negros limpios (tinta plena con borde seco apenas irregular)
        blk = np.zeros_like(m_body)
        for poly in (PRIMARIES, TAILBAND, BIB):
            blk = np.maximum(blk, self._fill(self.to_screen(_curve(poly, 6, closed=True))))
        self._over(reg, self._wc(blk, 0.1, tex=0.06), ink, 0.95)
        # patas
        for leg, w in ((LEG_FAR, 0.010), (LEG_NEAR, 0.012)):
            lp = self.to_screen(_curve(leg, 6))
            self.mask.clear()
            Stroke(lp, w * self.S, rng, taper=(0.02, 0.02), pool=0.0, jitter=0.03).draw(self.mask, 10)
            m_leg = self.mask.array()
            self._glaze(reg, m_leg, lin("#8a6b58"), 0.9)
            cover = np.maximum(cover, m_leg)

        # --- plumilla de grosor variable (1–4 px): gruesa en la sombra, fina a la luz ---
        strokes = []
        bc = _curve(BODY, 12, closed=True)
        n = len(bc)
        for (a, b, w, jit) in [(0.02, 0.30, 4.0, 0.25), (0.33, 0.55, 3.4, 0.25), (0.60, 0.74, 1.6, 0.3),
                               (0.78, 0.97, 1.3, 0.35)]:
            strokes.append(Stroke(self.to_screen(bc[int(a * n):int(b * n)]), w * self.u, rng, pool=0.45,
                                  smooth=False, jitter=jit))
        wc_ = _curve(WING, 10, closed=False)
        k = len(wc_)
        strokes.append(Stroke(self.to_screen(wc_[:k // 2]), 2.6 * self.u, rng, pool=0.35, smooth=False, jitter=0.3))
        strokes.append(Stroke(self.to_screen(wc_[k // 2 - 2:k - 3]), 1.8 * self.u, rng, pool=0.3, smooth=False))
        for row in range(3):
            for j in range(7 - row):
                u = j / 6.0
                x = 0.09 - 0.30 * u - row * 0.05 + rng.normal(0, 0.006)
                y = -0.585 + 0.10 * u + row * 0.032 + rng.normal(0, 0.006)
                pts = [(x + 0.028, y - 0.012), (x + 0.004, y + 0.006), (x - 0.03, y + 0.01)]
                strokes.append(self._S(pts, rng.uniform(0.9, 1.6), rng, pool=0.12, taper=(0.3, 0.4)))
        for j in range(4):
            y0_ = -0.448 - 0.006 * j
            strokes.append(self._S([(-0.13 - 0.02 * j, y0_ - 0.01), (-0.28, y0_ + 0.006),
                                    (-0.43 + 0.01 * j, -0.436 - 0.001 * j)], 1.3, rng, pool=0.15))
        for leg, w in ((LEG_FAR, 1.6), (LEG_NEAR, 2.1)):
            lp = np.asarray(leg)
            for side, ww in ((-1, 1.0), (1, 0.6)):
                off = np.array([0.0055 * side, 0.0])
                strokes.append(self._S(lp + off, w * ww, rng, taper=(0.05, 0.1), pool=0.2, jitter=0.05))
        for toe in TOES_NEAR + TOES_FAR:
            strokes.append(self._S(toe, 1.7, rng, pool=0.35, taper=(0.05, 0.5)))
        for j in range(30):
            x = rng.uniform(-0.38, 0.42)
            hh = rng.uniform(0.02, 0.10) * (1.4 if rng.random() < 0.15 else 1.0)
            lean = rng.normal(0.01, 0.035)
            strokes.append(self._S([(x, 0.01), (x + lean * 0.4, -hh * 0.5), (x + lean, -hh)], rng.uniform(0.9, 1.7),
                                   rng, taper=(0.02, 0.7), pool=0.1))
        ink_a = self._strokes_mask(strokes)
        self._over(reg, ink_a, ink, 0.95)
        cover = np.maximum(cover, ink_a)
        return reg, np.clip(cover * 1.4, 0, 1)

    # --- cabeza animada ---------------------------------------------------------------------------------
    def _head_outline(self, t):
        """Silueta continua cuello+cabeza (sin «tubo»): nuca, arco del cráneo, frente, garganta."""
        f = self.head_tf(t)
        nb, nf = np.array(NECK_BACK), np.array(NECK_FRONT)
        a_back, a_th = HEAD_BACK_A, HEAD_THROAT_A
        hb_ = f([HEAD_C + HEAD_R * np.array([np.cos(a_back), np.sin(a_back)])])[0]
        ht = f([HEAD_C + HEAD_R * np.array([np.cos(a_th), np.sin(a_th)])])[0]
        back = _curve([nb + (0.006, 0.03), nb, (nb + hb_) / 2 + (0.010, 0.0), hb_], 8)
        arc_a = np.linspace(a_back, 2 * np.pi + a_th, 60)
        arc = f(np.stack([HEAD_C[0] + HEAD_R * np.cos(arc_a), HEAD_C[1] + HEAD_R * np.sin(arc_a)], 1))
        front = _curve([ht, (ht + nf) / 2 + (0.012, 0.0), nf], 8)
        return back, arc, front

    def head_sprite(self, t, paper_rgb, ink, white, red):
        x0, y0, x1, y1 = self.bbox
        reg = paper_rgb[y0:y1, x0:x1].copy()
        rng = np.random.default_rng(self.seed + 2)
        f = self.head_tf(t)
        hs = lambda p: self.to_screen(f(p))
        grey = lin("#8f8a80")
        fwd, gape = self.shout(t)

        back, arc, front = self._head_outline(t)
        # cierre por encima de la pechera (no la tapa) y hacia adentro del dorso
        close = np.array([(0.19, -0.669), (0.158, -0.663), (0.14, -0.642)])
        sil = self.to_screen(np.vstack([back, arc, front, close]))
        m_sil = self._fill(sil)
        self._glaze(reg, self._wc(m_sil, 0.5), grey, 0.95)
        # mejilla algo más clara (volumen)
        chk = hs(np.stack([HEAD_C[0] + 0.012 + 0.045 * np.cos(np.linspace(0, 6.3, 20)),
                           HEAD_C[1] + 0.012 + 0.035 * np.sin(np.linspace(0, 6.3, 20))], 1))
        self._over(reg, cv2.GaussianBlur(self._fill(chk), (0, 0), 3 * self.u) * m_sil, white, 0.18)
        # antifaz negro continuo: frente -> garganta -> (cuello) -> pechera
        m_mask = self._fill(hs(_curve(FACE_MASK, 5, closed=True)))
        self.mask.clear()
        Stroke(self.to_screen(np.vstack([front, [(0.212, -0.655)]]) + np.array([-0.012, 0.0])), 0.036 * self.S, rng,
               pool=0.2, jitter=0.03, taper=(0.02, 0.02),
               smooth=False).draw(self.mask, 10)
        m_mask = np.maximum(m_mask, self.mask.array())
        self._over(reg, self._wc(m_mask, 0.1, tex=0.05), ink, 0.95)
        # borde blanco del antifaz
        self.mask.clear()
        Stroke(hs(WHITE_BORDER), 1.7 * self.u, rng, pool=0.1, jitter=0.05, taper=(0.3, 0.5)).draw(self.mask, 10)
        self._over(reg, self.mask.array() * m_sil, white, 0.9)

        # pico: el inferior se abre 20° al gritar
        g = np.deg2rad(22) * gape
        cg, sg = np.cos(g), np.sin(g)
        lo = (np.asarray(BILL_LO) - BILL_HINGE) @ np.array([[cg, -sg], [sg, cg]]).T + BILL_HINGE
        m_up = self._fill(hs(BILL_UP))
        m_lo = self._fill(hs(lo))
        if gape > 0.05:  # interior de la boca
            mouth = np.vstack([np.asarray(BILL_UP)[[0, 3]], lo[[0]]])
            self._over(reg, self._fill(hs(mouth)), lin("#4a2a22"), 0.9)
        self._glaze(reg, np.maximum(m_up, m_lo), lin("#7a5a4a"), 1.0)
        tip_up = [(0.34, -0.752), (0.368, -0.747), (0.366, -0.742), (0.34, -0.745)]
        self._over(reg, self._fill(hs(tip_up)), ink, 0.95)
        cover = np.clip(m_sil + m_up + m_lo + m_mask, 0, 1)

        # plumilla: nuca y cráneo con grosor variable (fino arriba, grueso en la nuca)
        strokes = [Stroke(self.to_screen(back[len(back) // 3:]), 3.2 * self.u, rng, pool=0.3, smooth=False,
                          jitter=0.12, taper=(0.3, 0.1))]
        na = len(arc)
        strokes.append(Stroke(self.to_screen(arc[:na // 2]), 1.5 * self.u, rng, pool=0.3, smooth=False, jitter=0.15))
        strokes.append(Stroke(self.to_screen(arc[na // 2:int(na * 0.78)]), 1.2 * self.u, rng, pool=0.2,
                              smooth=False, taper=(0.3, 0.5)))
        strokes.append(Stroke(hs(BILL_UP[:2]), 1.4 * self.u, rng, pool=0.25, smooth=False))
        strokes.append(Stroke(hs(lo[1:]), 1.2 * self.u, rng, pool=0.25, smooth=False))
        # cresta: se eriza en la alerta (trazos que se afinan)
        e = self.crest_erect(t)
        for j, cr in enumerate(CREST):
            cr = np.asarray(cr, np.float64)
            base = cr[0]
            rel = cr - base
            a = np.deg2rad(-8 + 36 * e - 4 * j)
            c, s = np.cos(a), np.sin(a)
            rel = rel @ np.array([[c, -s], [s, c]]).T
            strokes.append(Stroke(hs(rel + base), (3.4 - 1.1 * j) * self.u, rng, taper=(0.03, 0.92), pool=0.25,
                                  jitter=0.05))
        # grito: tres trazos de tinta que salen del pico (no «ondas»)
        if gape > 0.05:
            tip = f([(0.37, -0.742)])[0]
            for j, ang in enumerate((-32, -4, 24)):
                r = np.deg2rad(ang)
                d = np.array([np.cos(r), np.sin(r)])
                p0 = tip + d * (0.03 + 0.01 * j % 2)
                p1 = tip + d * (0.085 + 0.012 * (j == 1))
                strokes.append(Stroke(self.to_screen(np.array([p0, (p0 + p1) / 2, p1])), 3.0 * self.u * gape, rng,
                                      pool=0.4, smooth=False, taper=(0.05, 0.9)))
        ink_a = self._strokes_mask(strokes)
        self._over(reg, ink_a, ink, 0.95)
        cover = np.maximum(cover, ink_a)

        # el ojo
        o = self.eye_open(t)
        ec = hs([EYE])[0]
        r = EYE_R * self.S
        m = self.mask
        if o > 0.02:
            m.clear()
            m.dot(ec[0], ec[1], r * 1.32, 1.0)
            self._over(reg, m.array(), ink, 0.92)
            m.clear()
            c = m.ctx
            c.save()
            c.translate(ec[0], ec[1])
            c.scale(1.0, max(o, 0.05))
            c.arc(0, 0, r, 0, 2 * np.pi)
            c.restore()
            m.set_alpha(1.0)
            c.fill()
            self._over(reg, m.array(), red, 1.0)
            m.clear()
            m.dot(ec[0] + r * 0.08, ec[1] + r * 0.05, r * 0.42 * o, 1.0)
            self._over(reg, m.array(), ink, 1.0)
            m.clear()
            m.dot(ec[0] - r * 0.38, ec[1] - r * 0.42, r * 0.26, 1.0)
            self._over(reg, m.array(), white, 0.95 * o)
        else:
            m.clear()
            lid = np.array([[ec[0] - r * 1.3, ec[1] - r * 0.1], [ec[0], ec[1] + r * 0.5],
                            [ec[0] + r * 1.3, ec[1] - r * 0.15]])
            Stroke(lid, 1.8 * self.u, rng, pool=0.25).draw(m, 10)
            self._over(reg, m.array(), ink, 0.92)
        return reg, np.clip(cover * 1.4, 0, 1)

    def eye_screen(self, t):
        """Posición del ojo (pupila) en coordenadas de pantalla."""
        x0, y0 = self.bbox[:2]
        p = self.to_screen(self.head_tf(t)([EYE]))[0]
        return np.array([p[0] + x0, p[1] + y0])
