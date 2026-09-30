"""Aparato de lámina de guía de campo: marco, cabecera, leyenda, huellas, «No confundir con»,
mapa de distribución y numerales de figura."""
import numpy as np

from .ink import Stroke
from .typography import Word

# Contorno muy simplificado de Chile continental (lon, lat), de norte a sur por la costa y de
# vuelta por la cordillera. Suficiente para un mini-mapa de distribución.
CHILE_COAST = [(-70.3, -18.3), (-70.2, -20.2), (-70.15, -22.1), (-70.6, -23.6), (-70.6, -26.3),
               (-71.3, -29.9), (-71.6, -33.0), (-71.8, -34.3), (-72.5, -35.6), (-73.2, -36.8),
               (-73.6, -37.6), (-73.4, -39.0), (-73.8, -40.3), (-73.9, -41.7), (-74.2, -42.6),
               (-74.0, -43.4), (-74.4, -44.6), (-75.1, -46.2), (-75.5, -47.8), (-75.3, -49.6),
               (-74.8, -51.7), (-74.3, -53.0), (-72.4, -54.2), (-70.0, -55.0), (-67.4, -55.8),
               (-67.0, -54.9), (-68.6, -52.4)]
CHILE_ANDES = [(-68.6, -52.4), (-71.9, -52.0), (-72.3, -50.6), (-73.4, -49.2), (-72.4, -48.3),
               (-71.9, -46.6), (-71.7, -44.8), (-71.8, -42.8), (-71.7, -41.0), (-71.2, -39.3),
               (-71.0, -37.5), (-70.4, -35.5), (-70.0, -33.0), (-70.3, -30.0), (-69.6, -28.0),
               (-68.4, -26.0), (-67.4, -24.0), (-67.7, -22.2), (-68.4, -21.0), (-68.9, -19.6),
               (-69.5, -18.0), (-70.3, -18.3)]


class Plate:
    def __init__(self, W, H, lay, fonts, rng):
        self.W, self.H = W, H
        self.lay = lay
        self.f = fonts
        self.rng = rng
        self.u = lay["u"]

    # --- estático ----------------------------------------------------------------------------------
    def draw_static(self, mask):
        u = self.u
        c = mask.ctx
        W, H = self.W, self.H
        L = self.lay
        # Marco doble
        c.set_line_cap(0)
        for inset, w in ((L["frame"], 1.6 * u), (L["frame"] + 6 * u, 0.7 * u)):
            mask.set_alpha(0.9)
            c.set_line_width(w)
            c.rectangle(inset, inset, W - 2 * inset, H - 2 * inset)
            c.stroke()
        # Filetes de cabecera y pie
        mask.set_alpha(0.85)
        c.set_line_width(0.8 * u)
        for y in (L["head_rule"], L["foot_rule"]):
            c.move_to(L["margin"], y)
            c.line_to(W - L["margin"], y)
            c.stroke()
        # Cabecera
        sc, it, rm = self.f["sc"], self.f["it"], self.f["rm"]
        cap = L["head_cap"]
        yb = L["head_base"]
        w1 = Word("LA BANDADA VIGÍA  ·  ", sc, cap, 0, 0, 0.12, "left").width
        w2 = Word("Vigilans communitas", it, cap * 1.12, 0, 0, 0.0, "left").width
        if L.get("portrait"):
            x = L["margin"]
        else:
            self._text(mask, sc, "GUÍA DE CAMPO DEL TERRITORIO", cap, L["margin"], yb, tracking=0.16)
            x = W / 2 - (w1 + w2) / 2
        self._text(mask, sc, "LA BANDADA VIGÍA  ·  ", cap, x, yb, tracking=0.12)
        self._text(mask, it, "Vigilans communitas", cap * 1.12, x + w1, yb)
        self._text(mask, sc, "LÁM. I", cap, W - L["margin"], yb, tracking=0.16, anchor="right")
        # Leyenda (línea 1) y crédito
        fcap = L["foot_cap"]
        x0 = L["margin"]
        wa = self._text(mask, rm, "1  Queltehue, ", fcap, x0, L["foot_l1"]).width
        wb = self._text(mask, it, "Vanellus chilensis", fcap * 1.05, x0 + wa, L["foot_l1"]).width
        self._text(mask, rm, ", la vigía. Voz: «quel-te-hue».", fcap, x0 + wa + wb, L["foot_l1"])
        credit = "Red Comunitaria de Alerta Energética · 2026"
        if L.get("portrait"):
            # En vertical, el crédito corre a lo largo del margen derecho (línea de pie de imprenta)
            c = mask.ctx
            c.save()
            cw = Word(credit, rm, fcap * 0.85, 0, 0, 0.04, "left").width
            c.translate(W - L["frame"] - 14 * self.u, L["foot_rule"] - 20 * self.u)
            c.rotate(-np.pi / 2)
            ww = Word(credit, rm, fcap * 0.85, 0, 0, 0.04, "left")
            for Lt in ww.letters:
                Lt.draw(c)
            mask.set_alpha(0.85)
            c.fill()
            c.restore()
        else:
            self._text(mask, rm, credit, fcap * 0.92, W - L["margin"], L["foot_l2"], anchor="right")
        # Numeral 1 junto al queltehue
        n1 = L["num1"]
        self._text(mask, rm, "1", fcap * 1.3, n1[0], n1[1])
        self._huellas(mask)
        self._no_confundir(mask)
        self._distribucion(mask)

    def _text(self, mask, font, text, cap, x, y, tracking=0.0, anchor="left", alpha=0.92):
        if anchor == "right":
            w = Word(text, font, cap, 0, 0, tracking, "left").width
            x = x - w
        w = Word(text, font, cap, x, y, tracking, "left")
        for Lt in w.letters:
            Lt.draw(mask.ctx)
        mask.set_alpha(alpha)
        mask.ctx.fill()
        return w

    def _huellas(self, mask):
        u = self.u * self.lay.get("inset_scale", 1.0)
        x, y = self.lay["huellas"]
        rng = self.rng
        sc, rm, it = self.f["sc"], self.f["rm"], self.f["it"]
        fcap = self.lay["foot_cap"]
        self._text(mask, sc, "HUELLAS", fcap * 0.95, x, y, tracking=0.14)
        # Huella de queltehue: tres dedos adelante, uno mínimo atrás
        cx, cy = x + 14 * u, y + 36 * u
        for a in (-38, 0, 38):
            r = np.deg2rad(a - 90)
            p1 = (cx + 15 * u * np.cos(r), cy + 15 * u * np.sin(r))
            Stroke([(cx, cy), p1], 2.4 * u, rng, pool=0.5, taper=(0.1, 0.5), smooth=False).draw(mask, 10)
        Stroke([(cx, cy), (cx - 1 * u, cy + 5 * u)], 2.0 * u, rng, pool=0.4, smooth=False).draw(mask, 10)
        self._text(mask, it, "queltehue", self.lay["foot_cap"] * 0.8, x + 32 * u, y + 30 * u)
        self._scale_bar(mask, x + 34 * u, y + 44 * u, 20 * u, "5 cm")
        # Planta de la torre: base y tres tensores a 120°, con regla (grosor constante)
        tx, ty = x + 150 * u, y + 34 * u
        c = mask.ctx
        c.set_line_cap(0)
        mask.set_alpha(0.92)
        R = 16 * u
        for k in range(3):
            a = np.deg2rad(90 + 120 * k)
            ax, ay = tx + R * np.cos(a), ty + R * np.sin(a)
            c.set_line_width(0.7 * u)
            c.move_to(tx, ty)
            c.line_to(ax, ay)
            c.stroke()
            c.set_line_width(1.2 * u)
            c.rectangle(ax - 1.8 * u, ay - 1.8 * u, 3.6 * u, 3.6 * u)
            c.stroke()
        tri = [(tx + 3.2 * u * np.cos(np.deg2rad(90 + 120 * k)), ty + 3.2 * u * np.sin(np.deg2rad(90 + 120 * k)))
               for k in range(3)]
        mask.fill_poly(tri, 0.92)
        self._text(mask, it, "torre", self.lay["foot_cap"] * 0.8, tx + 24 * u, y + 30 * u)
        self._scale_bar(mask, tx + 24 * u, y + 44 * u, 30 * u, "50 m")

    def _scale_bar(self, mask, x, y, L, label):
        u = self.u * self.lay.get("inset_scale", 1.0)
        c = mask.ctx
        c.set_line_cap(0)
        mask.set_alpha(0.9)
        c.set_line_width(0.8 * u)
        c.move_to(x, y)
        c.line_to(x + L, y)
        c.stroke()
        for xx in (x, x + L):
            c.move_to(xx, y - 3 * u)
            c.line_to(xx, y + 3 * u)
            c.stroke()
        c.rectangle(x, y - 1.5 * u, L / 2, 3 * u)
        c.fill()
        self._text(mask, self.f["rm"], label, self.lay["foot_cap"] * 0.7, x + L + 4 * u, y + 3 * u)

    def _no_confundir(self, mask):
        u = self.u * self.lay.get("inset_scale", 1.0)
        x, y = self.lay["no_confundir"]
        sc, it, rm = self.f["sc"], self.f["it"], self.f["rm"]
        fcap = self.lay["foot_cap"]
        self._text(mask, sc, "NO CONFUNDIR CON", fcap * 0.95, x, y, tracking=0.14)
        # Antena de telefonía: celosía ahusada, platos y paneles
        c = mask.ctx
        c.set_line_cap(1)
        mask.set_alpha(0.9)
        bx, by = x + 16 * u, y + 62 * u
        top = by - 48 * u
        c.set_line_width(1.0 * u)
        for s in (-1, 1):
            c.move_to(bx + s * 7 * u, by)
            c.line_to(bx + s * 2.5 * u, top)
            c.stroke()
        n = 7
        for i in range(n):
            f0, f1 = i / n, (i + 1) / n
            w0 = 7 - 4.5 * f0
            w1 = 7 - 4.5 * f1
            s0 = -1 if i % 2 == 0 else 1
            c.set_line_width(0.6 * u)
            c.move_to(bx + s0 * w0 * u, by - 48 * u * f0)
            c.line_to(bx - s0 * w1 * u, by - 48 * u * f1)
            c.stroke()
        # platos
        for (dx, dy, r) in ((-7, -36, 4.2), (7, -30, 3.6)):
            c.save()
            c.translate(bx + dx * u, by + dy * u)
            c.scale(0.55, 1.0)
            c.arc(0, 0, r * u, 0, 2 * np.pi)
            c.restore()
            c.set_line_width(1.0 * u)
            c.stroke()
        # paneles
        for dx in (-4.5, 0, 4.5):
            c.rectangle(bx + dx * u - 1.2 * u, top - 1 * u, 2.4 * u, 7 * u)
            c.fill()
        self._text(mask, it, "antena de telefonía", fcap * 0.9, x + 38 * u, y + 30 * u)
        self._text(mask, it, "(platos y paneles)", fcap * 0.9, x + 38 * u, y + 52 * u)

    def _distribucion(self, mask):
        u = self.u
        x0, y0, y1 = self.lay["map_x"], self.lay["map_top"], self.lay["map_bottom"]
        sc, it = self.f["sc"], self.f["it"]
        fcap = self.lay["foot_cap"]
        lat0, lat1 = -17.5, -56.0
        k = (y1 - y0) / (lat0 - lat1)
        lon_c = -71.0

        def P(lon, lat):
            return (x0 + (lon - lon_c) * k * np.cos(np.deg2rad(-lat)), y0 + (lat0 - lat) * k)

        poly = [P(*p) for p in CHILE_COAST + CHILE_ANDES]
        mask.set_alpha(0.9)
        c = mask.ctx
        c.set_line_width(0.8 * u)
        c.move_to(*poly[0])
        for p in poly[1:]:
            c.line_to(*p)
        c.close_path()
        c.stroke()
        # Rango «de Arica a Chiloé»: relleno de tinta con clip al contorno
        c.save()
        c.move_to(*poly[0])
        for p in poly[1:]:
            c.line_to(*p)
        c.close_path()
        c.clip()
        ya, yc = P(-70, -18.4)[1], P(-70, -42.6)[1]
        c.rectangle(x0 - 60 * u, ya, 120 * u, yc - ya)
        c.clip()
        c.set_line_width(0.7 * u)
        mask.set_alpha(0.85)
        step = 2.6 * u
        yy = ya - 120 * u
        while yy < yc + 120 * u:
            c.move_to(x0 - 60 * u, yy)
            c.line_to(x0 + 60 * u, yy - 120 * u)
            yy += step
        c.stroke()
        c.restore()
        # Marcas
        for lat, name in ((-18.48, "Arica"), (-42.6, "Chiloé")):
            px, py = P(-70.0, lat)
            c.set_line_width(0.7 * u)
            mask.set_alpha(0.9)
            c.move_to(x0 - 18 * u, py)
            c.line_to(x0 - 10 * u, py)
            c.stroke()
            self._text(mask, it, name, fcap * 0.85, x0 - 20 * u, py + 4 * u, anchor="right")
        self._text(mask, sc, "DISTRIBUCIÓN", fcap * 0.8, x0 + 10 * u, y0 - 12 * u, tracking=0.12, anchor="right")

    # --- dinámico ----------------------------------------------------------------------------------
    def draw_legend2(self, mask, t, t0):
        """Línea 2 de la leyenda y numeral 2: aparecen con la torre."""
        a = float(np.clip((t - t0) / 0.18, 0, 1))
        if a <= 0:
            return
        L = self.lay
        rm, it = self.f["rm"], self.f["it"]
        fcap = L["foot_cap"]
        x0 = L["margin"]
        wa = self._text(mask, rm, "2  Torre de medición de viento, o ", fcap, x0, L["foot_l2"], alpha=0.92 * a).width
        self._text(mask, it, "mástil anemométrico.", fcap * 1.05, x0 + wa, L["foot_l2"], alpha=0.92 * a)
        n2 = L["num2"]
        self._text(mask, rm, "2", fcap * 1.3, n2[0], n2[1], alpha=0.92 * a)
