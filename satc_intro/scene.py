"""La lámina completa y su línea de tiempo (5 s, 30 fps).

Acto I   (0,00–0,55) La vigía y la torre: la torre se traza con regla en 0,4 s; el queltehue deja de
                     picotear, despierta y abre el ojo: el primer rojo.
Acto II  (0,55–2,62) El murmullo y el clic: las motas del kraft se despegan y forman las palabras
                     como puntillado, que se densifica hasta leerse; el queltehue grita y ALERTA sale
                     de su pico abierto letra por letra; en el clic (1,65 s) todo se imprime en
                     registro y el papel se hunde. El título queda quieto ~1 s.
Acto III (2,62–5,00) La suelta y la posta: onda desde el ojo; cada letra se abre como alas y es un
                     queltehue que baja al potrero; la «A» vuela con su plancha roja, que al posarse
                     se contrae hasta su ojo: la vigía de turno; las demás duermen; el queltehue
                     cierra el suyo; nota final y ≥ 1,1 s de reposo.
"""
import cv2
import numpy as np

from . import landscape
from .color import lin, linear_to_srgb, PALETTE
from .flock import Flock
from .ink import Mask
from .murmur import Murmur, blue_noise_in_mask
from .noise import fbm, smoothstep
from .paper import make_kraft
from .plate import Plate
from .queltehue import Queltehue
from .title import Title
from .tower import Tower
from .typography import Font, Word

FPS = 30
DURATION = 5.0

T = dict(tower=(0.06, 0.46), legend2=0.40, alert=0.44, eye=0.52, lift=(0.55, 0.80), shout=(1.02, 1.28),
         click=1.65, release=2.62, sleep=3.60, note=(3.55, 3.80))


def layout(W, H):
    """Maquetación. 16:9 (escritorio) o 4:5 (móvil)."""
    if W / H > 1.2:
        u = W / 1920.0
        v = H / 1080.0
        s = lambda x, y: (x * u, y * v)
        return dict(
            u=u, frame=26 * u, margin=60 * u, head_base=66 * v, head_rule=84 * v, head_cap=13.5 * u,
            foot_rule=958 * v, foot_l1=990 * v, foot_l2=1022 * v, foot_cap=13.5 * u,
            plate_box=(150 * u, 470 * v, 1770 * u, 950 * v), horizon=690 * v, tower_base=s(1590, 640),
            tower_h=310 * v, meadow_top=772 * v, bird_feet=s(430, 905), bird_size=480 * v,
            num1=s(186, 900), num2=s(1616, 322), huellas=s(840, 982), no_confundir=s(1232, 982),
            map_x=1824 * u, map_top=136 * v, map_bottom=430 * v,
            title_cx=1075 * u, title_w=560 * u, title_top=150 * v, title_gap=14 * v, title_small_cap=40 * v,
            land_box=(700 * u, 800 * v, 1700 * u, 938 * v), land_h=(26 * v, 42 * v),
            duty_pt=s(1010, 950), duty_h=132 * v,
            note_cx=1075 * u, note_y=(300 * v, 352 * v), note_cap=26 * u,
        )
    if H / W > 1.5:
        # 9:16 (teléfono, pantalla completa)
        u = W / 1080.0
        v = H / 1920.0
        s = lambda x, y: (x * u, y * v)
        return dict(
            portrait=True, inset_scale=0.78,
            u=u, frame=22 * u, margin=48 * u, head_base=64 * v, head_rule=82 * v, head_cap=12.5 * u,
            foot_rule=1790 * v, foot_l1=1822 * v, foot_l2=1852 * v, foot_cap=12.5 * u,
            plate_box=(60 * u, 1180 * v, 1020 * u, 1775 * v), horizon=1390 * v, tower_base=s(870, 1382),
            tower_h=330 * v, meadow_top=1480 * v, bird_feet=s(285, 1748), bird_size=450 * v,
            hill_w=320 * u, hill_h=105 * u,
            num1=s(66, 1738), num2=s(896, 1032), huellas=s(560, 1818), no_confundir=s(800, 1818),
            map_x=1000 * u, map_top=130 * v, map_bottom=400 * v,
            title_cx=490 * u, title_w=740 * u, title_top=300 * v, title_gap=16 * v, title_small_cap=52 * v,
            land_box=(560 * u, 1510 * v, 1000 * u, 1765 * v), land_h=(30 * v, 50 * v),
            duty_pt=s(700, 1770), duty_h=150 * v,
            note_cx=520 * u, note_y=(640 * v, 694 * v), note_cap=28 * u,
        )
    u = W / 1080.0
    v = H / 1350.0
    s = lambda x, y: (x * u, y * v)
    return dict(
        portrait=True, inset_scale=0.78,
        u=u, frame=22 * u, margin=48 * u, head_base=60 * v, head_rule=76 * v, head_cap=12.5 * u,
        foot_rule=1238 * v, foot_l1=1268 * v, foot_l2=1296 * v, foot_cap=12.5 * u,
        plate_box=(60 * u, 700 * v, 1020 * u, 1225 * v), horizon=915 * v, tower_base=s(900, 905),
        tower_h=285 * v, meadow_top=990 * v, bird_feet=s(270, 1190), bird_size=400 * v,
        hill_w=300 * u, hill_h=95 * u,
        num1=s(66, 1180), num2=s(924, 612), huellas=s(560, 1264), no_confundir=s(800, 1264),
        map_x=1000 * u, map_top=120 * v, map_bottom=380 * v,
        title_cx=470 * u, title_w=640 * u, title_top=122 * v, title_gap=14 * v, title_small_cap=44 * v,
        land_box=(420 * u, 1020 * v, 1000 * u, 1215 * v), land_h=(26 * v, 42 * v),
        duty_pt=s(640, 1228), duty_h=124 * v,
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
                        red=lin(PALETTE["alerta"]), red_dark=lin("#a8311c"), grey=lin("#7a7163"),
                        speck=lin("#2e2218"), hole=lin("#dfc9a2"))

        # --- papel y paisaje -------------------------------------------------------------------------
        paper, ph, cand = make_kraft(W, H, seed=seed, lift_n=int(4200 * (W * H) / (1920 * 1080)))
        self.paper, self.ph = paper, ph
        layers, geo = landscape.build(W, H, ph, seed=seed + 4, lay=dict(
            plate_box=L["plate_box"], horizon=L["horizon"], tower_base=L["tower_base"], meadow_top=L["meadow_top"],
            hill_w=L.get("hill_w"), hill_h=L.get("hill_h")))
        self.geo = geo
        canvas = paper.copy()
        for lyr in layers:
            lyr.apply(canvas, 10.0)

        # --- queltehue (reserva: pintado sobre papel limpio) ---------------------------------------
        self.q = Queltehue(L["bird_feet"], L["bird_size"], seed=seed + 1, t_alert=T["alert"], t_eye=T["eye"],
                           t_shout=T["shout"][0], t_shout_end=T["shout"][1], t_sleep=T["sleep"])
        qx0, qy0, qx1, qy1 = self.q.bbox

        def avoid(x, y):
            return qx0 + 40 * u < x < qx1 - 20 * u and qy0 < y < qy1 - 30 * u

        m = Mask(W, H)
        for st in landscape.ink_strokes(geo, W, H, seed=seed + 9, avoid=avoid):
            st.draw(m, 10.0)
        self.gx, self.gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        r2 = np.random.default_rng(seed + 50)
        self.rough_dx = (fbm((H, W), 8, r2, octaves=3) - 0.5) * 1.6
        self.rough_dy = (fbm((H, W), 8, r2, octaves=3) - 0.5) * 1.6
        self._over(canvas, self._rough(m.array()), self.col["sepia_ink"], 0.88)
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
        self.title = Title(L, self.fonts["anton"], self.q.eye_screen, rng, t_click=T["click"],
                           beak_fn=self.q.beak_screen, t_ring=T["shout"][0])
        blk = self.title.block
        # puntillado: puntos con distancia mínima dentro de cada palabra en tinta
        targets, fills = [], []
        for w in self.title.words:
            if w.color != "ink":
                continue
            m.clear()
            for Lt in w.letters:
                Lt.draw(m.ctx)
            m.set_alpha(1.0)
            m.ctx.fill()
            area = (m.array() > 0.6)
            n_w = int(np.clip(area.sum() / (24.0 * u * u), 220, 640))
            # el contorno define la letra: ~68 % de los puntos en el borde, el resto adentro
            er = cv2.erode(area.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=max(1, int(round(u)))) > 0
            edge = area & ~er
            n_e = int(n_w * 0.68)
            pe = blue_noise_in_mask(edge, n_e, rng, max(2.4 * u, np.sqrt(edge.sum() / max(1, n_e)) * 0.8))
            pi = blue_noise_in_mask(er, n_w - len(pe), rng, max(3.2 * u, np.sqrt(er.sum() / max(1, n_w - n_e)) * 0.75))
            targets.append(np.vstack([pe, pi]) if len(pi) else pe)
            # puntos de relleno: la tinta se asienta dentro de los trazos antes del clic
            fills.append(blue_noise_in_mask(area, int(area.sum() / (11.0 * u * u)), rng, 2.7 * u))
        targets = np.vstack(targets)
        # motas de origen: visibles desde el cuadro 0, fuera del título y del queltehue, más en el cielo
        x0b, y0b, x1b, y1b = blk
        ok = []
        for (x, y, r, d) in cand:
            if qx0 < x < qx1 and qy0 < y < qy1:
                continue
            if y > L["plate_box"][3] - 10 * u or y < L["head_rule"] + 10 * u:
                continue
            if y > L["meadow_top"] and rng.random() < 0.75:
                continue
            ok.append((x, y, r, d))
        ok = np.array(ok)
        rng.shuffle(ok)
        sources = ok[:len(targets)]
        targets = targets[:len(sources)]
        self.murmur = Murmur(sources, targets, rng, center=np.array([(x0b + x1b) / 2, (y0b + y1b) / 2]), u=u,
                             t_lift=T["lift"], t_print=T["click"] - 2.0 / FPS, fill=np.vstack(fills))
        eye_rel = self.q.eye_screen(T["release"])
        self.flock = Flock(self.title, L, rng, W, H, t_release=T["release"], eye=eye_rel, spread=0.45)
        # colores efectivos de las aves: los mismos pigmentos que la vigía, sobre el kraft
        kraft = self.paper.reshape(-1, 3).mean(axis=0)
        self.bird_colors = dict(back=kraft * lin("#7d7465") ** 0.95, head=kraft * lin("#8f8a80") ** 0.95,
                                white=self.col["white"], ink=self.col["ink"], red=self.col["red"])
        self.flock.set_textures(ph, self.rough_dx * 1.6, self.rough_dy * 1.6,
                                dict(back=lin("#7d7465"), head=lin("#8f8a80")))

        # --- grano fijo al papel, distinto en cada plancha --------------------------------------------------
        fine = fbm((H, W), 1.6, r2, octaves=2)
        self.g_ink = (0.55 + 0.45 * smoothstep(0.18, 0.42, 0.65 * fine + 0.35 * ph)).astype(np.float32)
        self.g_ink = np.clip(self.g_ink * (0.93 + 0.1 * fbm((H, W), 40, r2, octaves=2)), 0, 1)
        coarse = fbm((H, W), 2.6, r2, octaves=2)
        self.g_red = (0.62 + 0.38 * smoothstep(0.22, 0.5, 0.7 * coarse + 0.3 * ph)).astype(np.float32)
        self.g_red = np.clip(self.g_red * (0.88 + 0.16 * fbm((H, W), 70, r2, octaves=2)), 0, 1)
        self.g_white = (0.35 + 0.65 * smoothstep(0.16, 0.5, 0.5 * fbm((H, W), 1.4, r2, octaves=2) + 0.5 * ph))
        self.g_white = self.g_white.astype(np.float32)

        # --- golpe de prensa: hundimiento amplio + marca de plancha que queda -------------------------------
        bx0, by0, bx1, by1 = [int(v) for v in blk]
        pad = int(26 * u)
        rect = np.zeros((H, W), np.float32)
        cv2.rectangle(rect, (bx0 - pad, by0 - pad), (bx1 + pad, by1 + pad), 1.0, -1)
        broad = cv2.GaussianBlur(rect, (0, 0), 55 * u)
        mark = cv2.GaussianBlur(rect, (0, 0), 3.0 * u)
        bgx = cv2.Sobel(broad, cv2.CV_32F, 1, 0, ksize=5)
        bgy = cv2.Sobel(broad, cv2.CV_32F, 0, 1, ksize=5)
        mgx = cv2.Sobel(mark, cv2.CV_32F, 1, 0, ksize=5)
        mgy = cv2.Sobel(mark, cv2.CV_32F, 0, 1, ksize=5)
        lx, ly = -0.55, -0.83
        sb = bgx * lx + bgy * ly
        sm = mgx * lx + mgy * ly
        self.dent_shade = (sb / (np.abs(sb).max() + 1e-6)).astype(np.float32)
        self.mark_shade = (sm / (np.abs(sm).max() + 1e-6)).astype(np.float32)
        gm = np.hypot(bgx, bgy).max() + 1e-6
        self.dent_dx = (bgx / gm).astype(np.float32)
        self.dent_dy = (bgy / gm).astype(np.float32)

        self.masks = {k: Mask(W, H) for k in ("ink", "white", "ink_plate", "red", "grey", "red_over",
                                              "white_over", "speck", "shadow", "holes", "tower", "note",
                                              "relay", "glint")}

    # --- utilidades de composición -------------------------------------------------------------------
    @staticmethod
    def _over(canvas, a, col, alpha=1.0):
        aa = np.clip(a * alpha, 0, 1)[..., None]
        canvas *= (1 - aa)
        canvas += col[None, None, :] * aa

    @staticmethod
    def _glaze(canvas, a, col, k):
        canvas *= np.exp(a[..., None] * k * np.log(np.clip(col, 0.01, 1))[None, None, :])

    def _rough(self, a):
        return cv2.remap(a, self.gx + self.rough_dx, self.gy + self.rough_dy, cv2.INTER_LINEAR)

    # --- cuadro -------------------------------------------------------------------------------------------
    def frame(self, t):
        L = self.L
        t2 = np.floor(t * FPS / 2) * 2 / FPS      # capas dibujadas: «en dos»
        img = self.base.copy()
        M = self.masks
        for m in M.values():
            m.clear()

        # I. La torre, trazada con regla (y la línea 2 de la leyenda)
        self.tower.draw(M["tower"], t2)
        self.plate.draw_legend2(M["tower"], t, T["legend2"])
        self._over(img, M["tower"].array(), self.col["ink"], 0.95)

        # La vigía: cabeza, cuello, cresta, pico y ojo
        q = self.q
        hr, ha = q.head_sprite(t, self.paper, self.col["ink"], self.col["white"], self.col["red"])
        x0, y0, x1, y1 = q.bbox
        reg = img[y0:y1, x0:x1]
        reg *= (1 - ha[..., None])
        reg += hr * ha[..., None]

        # II. El murmullo: motas del kraft, huecos claros, puntillado
        self.murmur.draw(M, t)
        self._over(img, M["holes"].array(), self.col["hole"], 0.8)
        self._over(img, cv2.GaussianBlur(M["shadow"].array(), (0, 0), 1.2), lin("#6d5a40"), 0.8)
        self._over(img, M["speck"].array(), self.col["speck"], 0.95)

        # El título: base blanca (solo bajo el rojo, en registro) -> plancha roja -> plancha negra
        tm = {"white": M["white"], "ink": M["ink_plate"], "red": M["red"]}
        self.title.draw(tm, t)
        self.flock.draw_letters(tm, t)
        a_w = M["white"].array()
        if a_w.any():
            self._over(img, a_w * self.g_white, self.col["white"], 0.9)
        a_r = M["red"].array()
        if a_r.any():
            self._glaze(img, a_r * self.g_red, self.col["red"], 1.0)
        a_i = M["ink_plate"].array()
        if a_i.any():
            self._glaze(img, a_i * self.g_ink, self.col["ink"], 1.35)

        # III. La bandada (cada ave compuesta por separado, de atrás hacia adelante) y la posta del rojo
        self.flock.render(img, t, self.bird_colors, self.g_red)

        # Nota de comportamiento (voz de guía de campo)
        self._note(img, t)

        # Golpe de prensa en el clic: 4 cuadros de hundimiento; queda la marca de plancha
        f = int(np.floor((t - T["click"]) * FPS + 1e-6))
        if f >= 0:
            amp = {0: 1.0, 1: 0.72, 2: 0.42, 3: 0.18}.get(f, 0.0)
            mark = 0.035 if f > 3 else 0.035 + 0.045 * amp
            shade = 1.0 + np.float32(0.08 * amp) * self.dent_shade + np.float32(mark) * self.mark_shade
            img *= shade[..., None]
            if amp > 0:
                d = np.float32(7.0 * L["u"] * amp)
                img = cv2.remap(img, (self.gx - self.dent_dx * d).astype(np.float32),
                                (self.gy - self.dent_dy * d).astype(np.float32), cv2.INTER_LINEAR,
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
        u = np.clip((t - t0) / (t1 - t0), 0, 1)
        xL = x
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
