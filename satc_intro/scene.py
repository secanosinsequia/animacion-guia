"""La lámina completa y su línea de tiempo (5 s, 30 fps).

Acto I   (0,00–1,20) La vigía y la torre: la torre se traza con regla; el queltehue despierta.
Acto II  (1,20–3,30) El murmullo y el clic: las motas forman el indicio; ALERTA sale del ojo y todo
                     encaja en registro en el mismo cuadro, con abolladura del papel.
Acto III (3,30–5,00) La suelta y la posta: cada letra es un queltehue que se posa en el potrero; el
                     rojo se junta en el ojo de la vigía de turno; el queltehue cierra el suyo.
"""
import cv2
import numpy as np

from . import landscape
from .color import lin, linear_to_srgb, PALETTE
from .flock import Flock
from .ink import Mask, Stroke
from .murmur import Murmur
from .noise import fbm, smoothstep
from .paper import make_kraft
from .plate import Plate
from .queltehue import Queltehue
from .title import Title
from .tower import Tower
from .typography import Font, Word

FPS = 30
DURATION = 5.0

T = dict(tower=(0.20, 1.00), legend2=0.95, alert=0.95, eye=1.08, lift=(1.10, 1.38), ghost=1.45,
         alerta=1.80, click=2.10, release=3.30, sleep=4.12, note=(4.10, 4.42))


def layout(W, H):
    """Maquetación. 16:9 (escritorio) o 4:5 (móvil)."""
    if W / H > 1.2:
        u = W / 1920.0
        s = lambda x, y: (x * u, y * H / 1080.0)
        v = H / 1080.0
        return dict(
            u=u, frame=26 * u, margin=60 * u, head_base=66 * v, head_rule=84 * v, head_cap=13.5 * u,
            foot_rule=958 * v, foot_l1=990 * v, foot_l2=1022 * v, foot_cap=13.5 * u,
            plate_box=(150 * u, 470 * v, 1770 * u, 950 * v), horizon=690 * v, tower_base=s(1590, 640),
            tower_h=310 * v, meadow_top=772 * v, bird_feet=s(430, 905), bird_size=480 * v,
            num1=s(186, 900), num2=s(1616, 322), huellas=s(840, 982), no_confundir=s(1232, 982),
            map_x=1824 * u, map_top=136 * v, map_bottom=430 * v,
            title_cx=1075 * u, title_w=560 * u, title_top=150 * v, title_gap=14 * v, title_small_cap=40 * v,
            land_box=(700 * u, 800 * v, 1700 * u, 938 * v), land_h=(24 * v, 46 * v),
            duty_pt=s(1210, 925), merge_pt=s(1150, 520),
            note_cx=1075 * u, note_y=(300 * v, 352 * v), note_cap=26 * u,
        )
    # 4:5 vertical
    u = W / 1080.0
    v = H / 1350.0
    s = lambda x, y: (x * u, y * v)
    return dict(
        u=u, frame=22 * u, margin=48 * u, head_base=60 * v, head_rule=76 * v, head_cap=12.5 * u,
        foot_rule=1238 * v, foot_l1=1268 * v, foot_l2=1296 * v, foot_cap=12.5 * u,
        plate_box=(60 * u, 700 * v, 1020 * u, 1225 * v), horizon=905 * v, tower_base=s(850, 870),
        tower_h=300 * v, meadow_top=985 * v, bird_feet=s(270, 1190), bird_size=400 * v,
        num1=s(66, 1180), num2=s(870, 560), huellas=s(560, 1322), no_confundir=s(820, 1322),
        map_x=1000 * u, map_top=120 * v, map_bottom=380 * v,
        title_cx=500 * u, title_w=700 * u, title_top=118 * v, title_gap=14 * v, title_small_cap=46 * v,
        land_box=(420 * u, 1010 * v, 1000 * u, 1215 * v), land_h=(22 * v, 40 * v),
        duty_pt=s(700, 1120), merge_pt=s(640, 760),
        note_cx=500 * u, note_y=(330 * v, 380 * v), note_cap=26 * u,
    )


class Scene:
    def __init__(self, W=1920, H=1080, seed=7):
        self.W, self.H = W, H
        L = self.L = layout(W, H)
        u = L["u"]
        rng = np.random.default_rng(seed)
        self.fonts = dict(anton=Font("Anton-Regular.ttf"), rm=Font("IMFellEnglish-Roman.ttf"),
                          it=Font("IMFellEnglish-Italic.ttf"), sc=Font("IMFellEnglish-SC.ttf"))
        self.col = dict(ink=lin("#221a14"), sepia_ink=lin("#2c2118"), white=lin("#f4ead0"),
                        red=lin(PALETTE["alerta"]), grey=lin("#8f8a80"), speck=lin("#3b2c1f"))

        # --- papel y paisaje -------------------------------------------------------------------------
        paper, ph, lifters = make_kraft(W, H, seed=seed, lift_n=int(260 * (W * H) / (1920 * 1080)))
        self.paper, self.ph = paper, ph
        layers, geo = landscape.build(W, H, ph, seed=seed + 4, lay=dict(
            plate_box=L["plate_box"], horizon=L["horizon"], tower_base=L["tower_base"], meadow_top=L["meadow_top"]))
        self.geo = geo
        canvas = paper.copy()
        for lyr in layers:
            lyr.apply(canvas, 10.0)

        # --- queltehue (reserva: pintado sobre papel limpio) ---------------------------------------
        self.q = Queltehue(L["bird_feet"], L["bird_size"], seed=seed + 1, t_alert=T["alert"], t_eye=T["eye"],
                           t_sleep=T["sleep"])
        qx0, qy0, qx1, qy1 = self.q.bbox

        def avoid(x, y):
            return qx0 + 40 * u < x < qx1 - 20 * u and qy0 < y < qy1 - 30 * u

        m = Mask(W, H)
        for st in landscape.ink_strokes(geo, W, H, seed=seed + 9, avoid=avoid):
            st.draw(m, 10.0)
        self._over(canvas, self._rough(m.array(), rng), self.col["sepia_ink"], 0.88)
        body_rgb, body_a = self.q.body_sprite(paper, ph, self.col["ink"], self.col["white"])
        reg = canvas[qy0:qy1, qx0:qx1]
        reg *= (1 - body_a[..., None])
        reg += body_rgb * body_a[..., None]

        # --- aparato de lámina --------------------------------------------------------------------------
        self.plate = Plate(W, H, L, self.fonts, rng)
        m.clear()
        self.plate.draw_static(m)
        self._over(canvas, m.array(), self.col["sepia_ink"], 0.95)
        self.base = canvas

        # --- elementos animados ---------------------------------------------------------------------------
        self.tower = Tower(L["tower_base"], L["tower_h"], scale=u, t0=T["tower"][0], t1=T["tower"][1])
        self.title = Title(L, self.fonts["anton"], self.q.eye_screen, rng, t_ghost=T["ghost"],
                           t_click=T["click"], t_alerta=T["alerta"])
        # objetivos del murmullo: puntos dentro de las palabras en tinta
        m.clear()
        for w in self.title.words:
            if w.color == "ink":
                for Lt in w.letters:
                    Lt.draw(m.ctx)
        m.set_alpha(1.0)
        m.ctx.fill()
        ys, xs = np.nonzero(m.array() > 0.6)
        targets = np.stack([xs, ys], 1)
        blk = self.title.block
        self.murmur = Murmur(lifters, targets, rng, center=np.array([(blk[0] + blk[2]) / 2, (blk[1] + blk[3]) / 2]),
                             u=u, t_lift=T["lift"])
        head = self.q.eye_screen(T["release"])
        self.flock = Flock(self.title, L, rng, t_release=T["release"], queltehue_head=head)

        # --- texturas fijas al papel (el grano no se mueve con las planchas) ----------------------------
        r2 = np.random.default_rng(seed + 50)
        fine = fbm((H, W), 1.6, r2, octaves=2)
        self.g_ink = (0.55 + 0.45 * smoothstep(0.18, 0.42, 0.65 * fine + 0.35 * ph)).astype(np.float32)
        self.g_ink = np.clip(self.g_ink * (0.93 + 0.1 * fbm((H, W), 40, r2, octaves=2)), 0, 1)
        self.g_white = smoothstep(0.16, 0.5, 0.5 * fbm((H, W), 1.4, r2, octaves=2) + 0.5 * ph).astype(np.float32)
        self.g_white = 0.35 + 0.65 * self.g_white
        # abolladura del papel: campo de altura bajo el bloque del título
        bx0, by0, bx1, by1 = [int(v) for v in blk]
        dent = np.zeros((H, W), np.float32)
        cv2.rectangle(dent, (bx0 - int(10 * u), by0 - int(10 * u)), (bx1 + int(10 * u), by1 + int(10 * u)), 1.0, -1)
        dent = cv2.GaussianBlur(dent, (0, 0), 38 * u)
        self.dent_gx = cv2.Sobel(dent, cv2.CV_32F, 1, 0, ksize=5) / 32.0
        self.dent_gy = cv2.Sobel(dent, cv2.CV_32F, 0, 1, ksize=5) / 32.0
        self.gx, self.gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.rough_dx = (fbm((H, W), 8, r2, octaves=3) - 0.5) * 1.6
        self.rough_dy = (fbm((H, W), 8, r2, octaves=3) - 0.5) * 1.6

        self.masks = {k: Mask(W, H) for k in ("ink", "white", "ink_plate", "red", "grey", "red_over",
                                              "white_over", "speck", "shadow", "tower", "note")}

    # --- utilidades de composición -------------------------------------------------------------------
    @staticmethod
    def _over(canvas, a, col, alpha=1.0):
        aa = np.clip(a * alpha, 0, 1)[..., None]
        canvas *= (1 - aa)
        canvas += col[None, None, :] * aa

    @staticmethod
    def _glaze(canvas, a, col, k):
        canvas *= np.exp(a[..., None] * k * np.log(np.clip(col, 0.01, 1))[None, None, :])

    def _rough(self, a, rng=None):
        if not hasattr(self, "rough_dx"):
            H, W = a.shape
            r = np.random.default_rng(3)
            self.gx, self.gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            self.rough_dx = (fbm((H, W), 8, r, octaves=3) - 0.5) * 1.6
            self.rough_dy = (fbm((H, W), 8, r, octaves=3) - 0.5) * 1.6
        return cv2.remap(a, self.gx + self.rough_dx, self.gy + self.rough_dy, cv2.INTER_LINEAR)

    # --- cuadro -------------------------------------------------------------------------------------------
    def frame(self, t):
        W, H, L = self.W, self.H, self.L
        u = L["u"]
        t2 = np.floor(t * FPS / 2) * 2 / FPS      # capas dibujadas: «en dos»
        img = self.base.copy()
        M = self.masks
        for m in M.values():
            m.clear()

        # I. La torre, trazada con regla (y la línea 2 de la leyenda)
        self.tower.draw(M["tower"], t2)
        self.plate.draw_legend2(M["tower"], t, T["legend2"])
        self._over(img, M["tower"].array(), self.col["ink"], 0.95)

        # La vigía: cabeza, cuello, cresta y ojo
        q = self.q
        hr, ha = q.head_sprite(t, self.paper, self.col["ink"], self.col["white"], self.col["red"])
        x0, y0, x1, y1 = q.bbox
        reg = img[y0:y1, x0:x1]
        reg *= (1 - ha[..., None])
        reg += hr * ha[..., None]

        # II. El murmullo: motas del kraft (las que aún no se levantan también se dibujan aquí)
        self.murmur.draw(M, t)
        self._over(img, cv2.GaussianBlur(M["shadow"].array(), (0, 0), 1.2), lin("#6d5a40"), 0.8)
        self._over(img, M["speck"].array(), self.col["speck"], 0.9)

        # El título: plancha blanca -> tinta -> rojo (tintas que se superponen, no canales RGB)
        tm = {"white": M["white"], "ink": M["ink_plate"], "red": M["red"]}
        self.title.draw(tm, t)
        a_w = M["white"].array()
        if a_w.any():
            self._over(img, a_w * self.g_white, self.col["white"], 0.94)
        a_i = M["ink_plate"].array()
        if a_i.any():
            self._glaze(img, a_i * self.g_ink, self.col["ink"], 1.35)
        a_r = M["red"].array()
        if a_r.any():
            self._glaze(img, a_r * self.g_ink, self.col["red"], 1.0)

        # III. La bandada y la posta del rojo
        fm = {"ink": M["ink"], "white": M["white_over"], "grey": M["grey"], "red_over": M["red_over"],
              "white_over": M["white_over"]}
        M["white_over"].clear()
        self.flock.draw(fm, t)
        self._over(img, M["grey"].array(), lin("#7a7163"), 0.97)
        self._over(img, M["white_over"].array() * (0.8 + 0.2 * self.g_white), self.col["white"], 0.95)
        self._over(img, M["ink"].array(), self.col["ink"], 0.95)
        self._over(img, M["red_over"].array(), self.col["red"], 1.0)
        if t >= self.flock.t_eye + 0.05:
            e, r = self.flock.duty_eye()
            M["white_over"].clear()
            M["white_over"].dot(e[0] - r * 0.4, e[1] - r * 0.45, max(0.6, r * 0.38), 1.0)
            self._over(img, M["white_over"].array(), self.col["white"], 0.9)

        # Nota de comportamiento (voz de guía de campo)
        self._note(img, t)

        # Abolladura del papel en el clic
        dt = t - T["click"]
        if 0 <= dt < 0.45:
            amp = np.exp(-dt / 0.14) * np.cos(dt * 2 * np.pi / 0.3)
            k = np.float32(5.5 * amp)
            shade = 1.0 + k * (self.dent_gx * -0.55 + self.dent_gy * -0.85)
            img *= np.clip(shade, 0.93, 1.07)[..., None]
            d = np.float32(220.0 * u * amp)
            img = cv2.remap(img, (self.gx + self.dent_gx * d).astype(np.float32),
                            (self.gy + self.dent_gy * d).astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        return img

    def _note(self, img, t):
        t0, t1 = T["note"]
        if t < t0:
            return
        L = self.L
        m = self.masks["note"]
        m.clear()
        sc, it = self.fonts["sc"], self.fonts["it"]
        cap = L["note_cap"]
        l1a, l1b = "Comportamiento: ", "cuando una ve, todas vuelan;"
        l2 = "duermen por turnos."
        wa = Word(l1a, sc, cap * 0.92, 0, 0, 0.08, "left").width
        wb = Word(l1b, it, cap * 1.1, 0, 0, 0.0, "left").width
        x = L["note_cx"] - (wa + wb) / 2
        y1, y2 = L["note_y"]
        for font, text, c, xx, yy, trk in ((sc, l1a, cap * 0.92, x, y1, 0.08), (it, l1b, cap * 1.1, x + wa, y1, 0.0)):
            w = Word(text, font, c, xx, yy, trk, "left")
            for Lt in w.letters:
                Lt.draw(m.ctx)
        w2 = Word(l2, it, cap * 1.1, L["note_cx"], y2, 0.0, "center")
        for Lt in w2.letters:
            Lt.draw(m.ctx)
        m.set_alpha(1.0)
        m.ctx.fill()
        a = m.array()
        # se escribe de izquierda a derecha, línea por línea, con el borde de la pluma suave
        u = np.clip((t - t0) / (t1 - t0), 0, 1)
        xL = L["note_cx"] - (wa + wb) / 2
        span = wa + wb
        H, W = a.shape
        xs = self.gx[0]
        f1 = np.clip(u * 1.6, 0, 1)
        f2 = np.clip(u * 1.6 - 0.6, 0, 1)
        r1 = smoothstep(xL + span * f1 + 8, xL + span * f1 - 8, xs)
        r2 = smoothstep(xL + span * f2 + 8, xL + span * f2 - 8, xs)
        yy = np.arange(H, dtype=np.float32)[:, None]
        mid = (y1 + y2) / 2 - L["note_cap"] * 0.4
        reveal = np.where(yy < mid, r1[None, :], r2[None, :])
        self._over(img, a * reveal * self.g_ink, self.col["sepia_ink"], 0.95)

    def frame_srgb8(self, t):
        f = self.frame(t)
        return (linear_to_srgb(f) * 255 + 0.5).astype(np.uint8)
