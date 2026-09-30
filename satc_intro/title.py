"""El título como impresión riso en tres niveles de confianza: rumor → indicio → dato.

* rumor: motas del kraft que se levantan (ver murmur.py);
* indicio: SISTEMA DE / TEMPRANA / COMUNITARIO flotan como pasadas descuadradas (plancha blanca y
  plancha de tinta, cada una rígida: se desplaza y gira apenas) que tiemblan «en dos»;
* dato: ALERTA sale del ojo del queltehue en un arco rojo; al llegar, TODO encaja en registro en el
  mismo cuadro, con 2 cuadros de anticipación y 3 de sobreimpulso.
"""
import numpy as np

from .typography import Word

FPS = 30.0


def _rigid(ctx, cx, cy, dx, dy, rot, sx=1.0, sy=None):
    sy = sx if sy is None else sy
    ctx.translate(cx + dx, cy + dy)
    ctx.rotate(rot)
    ctx.scale(sx, sy)
    ctx.translate(-cx, -cy)


class Title:
    def __init__(self, lay, font, eye_fn, rng, t_ghost=1.45, t_click=2.10, t_alerta=1.80):
        self.lay = lay
        self.font = font
        self.eye_fn = eye_fn          # t -> posición del ojo en pantalla
        self.t_ghost, self.t_click, self.t_alerta = t_ghost, t_click, t_alerta
        cx = lay["title_cx"]
        width = lay["title_w"]
        y = lay["title_top"]
        gap = lay["title_gap"]
        words = []
        # Bloque justificado: cada palabra se escala para ocupar el mismo ancho.
        specs = [("SISTEMA DE", "ink", 0.34), ("ALERTA", "red", 0.0), ("TEMPRANA", "ink", 0.0),
                 ("COMUNITARIO", "ink", 0.0)]
        caps = []
        for text, color, trk in specs:
            probe = Word(text, font, 100.0, 0, 0, trk, "left")
            caps.append(100.0 * width / probe.width)
        caps[0] = min(caps[0], lay.get("title_small_cap", 999))
        for (text, color, trk), cap in zip(specs, caps):
            if text == "SISTEMA DE":
                # la línea chica se ajusta con espaciado para calzar el ancho
                probe = Word(text, font, cap, 0, 0, 0.0, "left")
                n = len(text) - 1
                trk = (width - probe.width) / max(1, n) / (cap * font.upm / font.cap)
            y += cap
            w = Word(text, font, cap, cx, y, trk, "center")
            w.color = color
            words.append(w)
            y += gap
        self.words = words
        self.alerta = words[1]
        self.block = (min(w.box[0] for w in words), min(w.box[1] for w in words),
                      max(w.box[2] for w in words), max(w.box[3] for w in words))
        # Descuadre de cada plancha (rígida) por palabra
        self.off = {}
        for i, w in enumerate(words):
            for plate in ("white", "ink"):
                ang = rng.uniform(0, 2 * np.pi)
                mag = rng.uniform(6, 12) * lay["u"]
                self.off[(i, plate)] = (np.cos(ang) * mag, np.sin(ang) * mag, np.deg2rad(rng.uniform(-1.3, 1.3)))
        self.tremble_seed = rng.integers(1 << 30)
        # Letras en orden (para la suelta del acto III)
        self.letters = []
        for i, w in enumerate(words):
            for L in w.letters:
                L.word_index = i
                L.color = w.color
                self.letters.append(L)
        self.pop_time = {}  # id(letter) -> tiempo en que se vuelve pájaro

    # --- estados ------------------------------------------------------------------------------------
    def ghost_alpha(self, i, t):
        """Opacidad del indicio para la palabra i (llega a medida que llegan las motas)."""
        t0 = self.t_ghost + 0.07 * [0, 0, 1, 2][i]
        u = np.clip((t - t0) / 0.35, 0, 1)
        return 0.78 * u * u * (3 - 2 * u)

    def tremble(self, i, plate, t):
        """Temblor «en dos»: cambia cada 2 cuadros, sin interpolar."""
        step = int(np.floor(t * FPS / 2))
        r = np.random.default_rng((self.tremble_seed + step * 131 + i * 17 + (plate == "ink") * 7) % (1 << 31))
        return r.normal(0, 1.4, 2) * self.lay["u"], np.deg2rad(r.normal(0, 0.25))

    def click_scale(self, t):
        f = int(np.round((t - self.t_click) * FPS))
        return {-2: 0.985, -1: 0.975, 0: 1.045, 1: 0.985, 2: 1.008}.get(f, 1.0)

    def alerta_flight(self, t):
        """ALERTA sale del ojo: posición del centro, escala y avance (0..1)."""
        t0, t1 = self.t_alerta, self.t_click
        u = np.clip((t - t0) / (t1 - t0), 0, 1)
        e = u * u * (3 - 2 * u)
        e = 1 - (1 - u) ** 2.6
        p0 = self.eye_fn(t0)
        p3 = self.alerta.center
        p1 = p0 + np.array([40, -230]) * self.lay["u"]
        p2 = p3 + np.array([-260, 40]) * self.lay["u"]
        b = ((1 - e) ** 3) * p0 + 3 * e * (1 - e) ** 2 * p1 + 3 * e * e * (1 - e) * p2 + e ** 3 * p3
        s = 0.04 + 0.96 * e ** 1.4
        return b, s, u

    # --- dibujo --------------------------------------------------------------------------------------
    def _draw_word(self, ctx, w, skip=None, squash=None):
        for L in w.letters:
            if skip is not None and skip(L):
                continue
            if squash is not None:
                sq = squash(L)
                if sq is not None:
                    sx, sy = sq
                    ctx.save()
                    cx, cy = L.center
                    ctx.translate(cx, cy)
                    ctx.scale(sx, sy)
                    ctx.translate(-cx, -cy)
                    L.draw(ctx)
                    ctx.restore()
                    continue
            L.draw(ctx)

    def draw(self, masks, t):
        """Dibuja las planchas en masks['white'], masks['ink'], masks['red']."""
        if t < self.t_ghost - 0.01:
            return
        tc = self.t_click
        fclick = int(np.round((t - tc) * FPS))
        clicked = fclick >= 0

        def gone(L):
            tp = self.pop_time.get(id(L))
            return tp is not None and t >= tp + 2.0 / FPS

        def squash(L):
            tp = self.pop_time.get(id(L))
            if tp is None or t < tp:
                return None
            k = int(np.floor((t - tp) * FPS))
            return {0: (1.10, 0.88), 1: (1.18, 0.78)}.get(k, None)

        sc = self.click_scale(t)
        bcx, bcy = (self.block[0] + self.block[2]) / 2, (self.block[1] + self.block[3]) / 2
        for i, w in enumerate(self.words):
            is_alerta = w is self.alerta
            for plate in ("white", "red" if is_alerta else "ink"):
                ctx = masks[plate].ctx
                ctx.save()
                if clicked:
                    alpha = 1.0
                    _rigid(ctx, bcx, bcy, 0, 0, 0, sc)
                elif is_alerta:
                    if t < self.t_alerta:
                        ctx.restore()
                        continue
                    if plate == "white":  # la base blanca llega recién con el clic
                        ctx.restore()
                        continue
                    pos, s, u = self.alerta_flight(t)
                    alpha = 1.0
                    # estela: copias arrastradas (smear), no desenfoque
                    for lag, a in ((2.0, 0.22), (1.0, 0.42)):
                        p2, s2, _ = self.alerta_flight(t - lag / FPS / 1.5)
                        ctx.save()
                        cx, cy = w.center
                        ctx.translate(p2[0], p2[1])
                        ctx.scale(s2, s2)
                        ctx.translate(-cx, -cy)
                        self._draw_word(ctx, w)
                        masks[plate].set_alpha(a)
                        ctx.fill()
                        ctx.restore()
                    cx, cy = w.center
                    ctx.translate(pos[0], pos[1])
                    ctx.scale(s, s)
                    ctx.translate(-cx, -cy)
                else:
                    alpha = self.ghost_alpha(i, t)
                    if alpha <= 0:
                        ctx.restore()
                        continue
                    dx, dy, rot = self.off[(i, "ink" if plate != "white" else "white")]
                    boost = 1.35 if fclick in (-2, -1) else 1.0   # anticipación: se separan más
                    (tx, ty), tr = self.tremble(i, plate, t)
                    _rigid(ctx, w.center[0], w.center[1], dx * boost + tx, dy * boost + ty, rot * boost + tr,
                           sc)
                self._draw_word(ctx, w, skip=gone, squash=squash)
                masks[plate].set_alpha(alpha)
                ctx.fill()
                ctx.restore()
