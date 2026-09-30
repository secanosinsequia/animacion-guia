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
    def __init__(self, lay, font, eye_fn, rng, t_click=2.00, flight=0.40, beak_fn=None, t_ring=None):
        self.lay = lay
        self.font = font
        self.eye_fn = eye_fn          # t -> posición del ojo en pantalla
        self.beak_fn = beak_fn or eye_fn   # t -> punta del pico: de ahí sale el grito
        self.t_ring = t_ring               # anillo limpio que sale de la pupila al gritar
        self.t_click = t_click
        self.flight = flight           # cada letra de ALERTA vuela 12 cuadros
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
        # Desregistro riso: plancha negra y plancha roja en sentidos opuestos (±3–4 px)
        u = lay["u"]
        ang = rng.uniform(0, 2 * np.pi)
        self.mis = np.array([np.cos(ang), np.sin(ang)]) * 3.6 * u
        self.letters = []
        for i, w in enumerate(words):
            for L in w.letters:
                L.word_index = i
                L.color = w.color
                self.letters.append(L)
        self.pop_time = {}  # id(letter) -> tiempo en que se vuelve pájaro
        # ALERTA sale del ojo letra por letra (1 cuadro de desfase); la última llega en el clic
        nA = len(self.alerta.letters)
        self.depart = [t_click - flight - (nA - 1 - k) / FPS for k in range(nA)]

    # --- estados ------------------------------------------------------------------------------------
    def click_scale(self, t):
        f = int(np.round((t - self.t_click) * FPS))
        return {0: 1.045, 1: 0.985, 2: 1.008}.get(f, 1.0)

    def plate_offset(self, t, plate):
        """Desregistro que salta cada cuadro en los 2 cuadros previos al clic; luego, registro exacto."""
        f = int(np.floor((t - self.t_click) * FPS + 1e-6))
        if f >= 0:
            return np.zeros(2)
        sign = 1 if plate == "ink" else -1
        jump = {-2: np.array([1.0, 0.35]), -1: np.array([-0.55, 1.0])}.get(f, np.array([1.0, 0.35]))
        R = np.array([[self.mis[0], -self.mis[1]], [self.mis[1], self.mis[0]]]) / max(1e-6, np.hypot(*self.mis))
        return sign * (R @ jump) * np.hypot(*self.mis)

    def letter_flight(self, k, t):
        """Posición del centro, escala y avance de la letra k de ALERTA (sale del pico abierto)."""
        L = self.alerta.letters[k]
        t0 = self.depart[k]
        u = np.clip((t - t0) / self.flight, 0, 1)
        e = 1 - (1 - u) ** 2.4
        p0 = self.beak_fn(t0)
        p3 = L.center
        uu = self.lay["u"]
        p1 = p0 + np.array([150, -12]) * uu                 # sale hacia adelante, por las líneas de voz
        p2 = p3 + np.array([-150 + 20 * k, 70]) * uu
        b = ((1 - e) ** 3) * p0 + 3 * e * (1 - e) ** 2 * p1 + 3 * e * e * (1 - e) * p2 + e ** 3 * p3
        s = 0.10 + 0.90 * e ** 1.2
        return b, s, u

    # --- dibujo --------------------------------------------------------------------------------------
    def draw(self, masks, t):
        """Dibuja las planchas en masks['ink'] (negra), masks['red'] (roja) y masks['white'] (base de la roja)."""
        tc = self.t_click
        f_rel = int(np.floor((t - tc) * FPS + 1e-6))
        sc = self.click_scale(t)
        bcx, bcy = (self.block[0] + self.block[2]) / 2, (self.block[1] + self.block[3]) / 2

        def state(L):
            tp = self.pop_time.get(id(L))
            if tp is None or t < tp:
                return "print", None
            k = int(np.floor((t - tp) * FPS + 1e-6))
            if k <= 1:
                return "squash", k
            return "gone", k

        # plancha negra: las palabras en tinta existen solo como puntos (murmullo) hasta 2 cuadros antes del clic
        if f_rel >= -2:
            off = self.plate_offset(t, "ink")
            ctx = masks["ink"].ctx
            for w in self.words:
                if w.color != "ink":
                    continue
                for L in w.letters:
                    st, k = state(L)
                    if st == "gone":
                        continue
                    ctx.save()
                    ctx.translate(off[0], off[1])
                    _rigid(ctx, bcx, bcy, 0, 0, 0, sc)
                    if st == "squash":
                        bx = (L.box[0] + L.box[2]) / 2
                        by = L.box[3]
                        ctx.translate(bx, by)
                        ctx.rotate(np.deg2rad(3.5 if k == 0 else -3.5))   # se estremece, sin achatarse
                        ctx.translate(-bx, -by)
                    L.draw(ctx)
                    ctx.restore()
            masks["ink"].set_alpha(1.0)
            ctx.fill()

        # plancha roja (con su base blanca en registro exacto): ALERTA sale del ojo
        off = self.plate_offset(t, "red")
        for k, L in enumerate(self.alerta.letters):
            if t < self.depart[k]:
                continue
            st, kk = state(L)
            if st == "gone":
                continue
            pos, s, u = self.letter_flight(k, t)
            for plate in ("white", "red"):
                ctx = masks[plate].ctx
                ctx.save()
                if u >= 1:
                    ctx.translate(off[0], off[1])
                    if f_rel >= 0:
                        _rigid(ctx, bcx, bcy, 0, 0, 0, sc)
                    if st == "squash":
                        bx = (L.box[0] + L.box[2]) / 2
                        by = L.box[3]
                        ctx.translate(bx, by)
                        ctx.rotate(np.deg2rad(3.5 if kk == 0 else -3.5))  # se estremece, sin achatarse
                        ctx.translate(-bx, -by)
                else:
                    cx, cy = L.center
                    ctx.translate(pos[0], pos[1])
                    ctx.scale(s, s)
                    ctx.translate(-cx, -cy)
                L.draw(ctx)
                masks[plate].set_alpha(1.0)
                ctx.fill()
                ctx.restore()
            # un único «smear» dibujado por letra, recién cuando ya se separó del ojo (sin línea al ojo)
            if 0 < u < 1:
                pp, sp, _ = self.letter_flight(k, t - 1.0 / FPS)
                v = pos - pp
                sp_len = np.hypot(*v)
                away = np.hypot(*(pos - self.eye_fn(self.depart[k])))
                if sp_len > 6 * self.lay["u"] and away > 70 * self.lay["u"]:
                    ang = np.arctan2(v[1], v[0])
                    mid = pos - v * 0.35
                    stretch = min(1.8, 1.0 + sp_len / max(8.0, (L.box[3] - L.box[1]) * s))
                    ctx = masks["red"].ctx
                    ctx.save()
                    cx, cy = L.center
                    ctx.translate(mid[0], mid[1])
                    ctx.rotate(ang)
                    ctx.scale(stretch, 0.8)
                    ctx.rotate(-ang)
                    ctx.scale(s, s)
                    ctx.translate(-cx, -cy)
                    L.draw(ctx)
                    masks["red"].set_alpha(0.3)
                    ctx.fill()
                    ctx.restore()
        # anillo limpio que sale de la pupila al gritar (sin gotas)
        if self.t_ring is not None:
            d = (t - self.t_ring) * FPS
            if 0 <= d < 6:
                uu = self.lay["u"]
                e = self.eye_fn(self.t_ring)
                r = (9 + 7.5 * d) * uu
                c = masks["red"].ctx
                c.set_line_width(max(0.8, (2.8 - 0.4 * d) * uu))
                c.new_path()
                c.arc(e[0], e[1], r, 0, 2 * np.pi)
                masks["red"].set_alpha(0.9 * (1 - d / 6))
                c.stroke()
