"""Tipografía: maquetado con HarfBuzz (kerning real), contornos con fontTools, dibujo con Cairo.

Cada palabra conoce la caja y el centro de cada letra, para poder animarlas una a una (el golpe
riso del título y, al final, la transformación de cada letra en un pájaro).
"""
import os

import numpy as np
import uharfbuzz as hb
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")


class _OpsPen(BasePen):
    """Registra el contorno como operaciones cúbicas reproducibles en Cairo."""

    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.ops = []

    def _moveTo(self, p):
        self.ops.append(("M", p))

    def _lineTo(self, p):
        self.ops.append(("L", p))

    def _curveToOne(self, p1, p2, p3):
        self.ops.append(("C", p1, p2, p3))

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        c1 = (p0[0] + 2 / 3 * (p1[0] - p0[0]), p0[1] + 2 / 3 * (p1[1] - p0[1]))
        c2 = (p2[0] + 2 / 3 * (p1[0] - p2[0]), p2[1] + 2 / 3 * (p1[1] - p2[1]))
        self.ops.append(("C", c1, c2, p2))

    def _closePath(self):
        self.ops.append(("Z",))

    def _endPath(self):
        self.ops.append(("Z",))


class Font:
    def __init__(self, filename):
        path = os.path.join(FONT_DIR, filename)
        self.tt = TTFont(path)
        self.gs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()
        self.upm = self.tt["head"].unitsPerEm
        self.cap = getattr(self.tt["OS/2"], "sCapHeight", 0) or self.upm * 0.7
        blob = hb.Blob.from_file_path(path)
        self.hbfont = hb.Font(hb.Face(blob))
        self._ops = {}

    def ops(self, name):
        if name not in self._ops:
            pen = _OpsPen(self.gs)
            self.gs[name].draw(pen)
            self._ops[name] = pen.ops
        return self._ops[name]

    def shape(self, text, px, tracking=0.0):
        """Devuelve [(nombre, x, y)] en px (línea base en y=0) y el ancho total."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hbfont, buf, {"kern": True, "liga": True})
        k = px / self.upm
        x = 0.0
        out = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            name = self.order[info.codepoint]
            out.append((name, x + pos.x_offset * k, -pos.y_offset * k))
            x += pos.x_advance * k + tracking * px
        return out, x - tracking * px

    def draw(self, ctx, name, x, y, k):
        """Traza el glifo en el contexto Cairo (sin rellenar). k: px por unidad de fuente."""
        for op in self.ops(name):
            if op[0] == "M":
                ctx.move_to(x + op[1][0] * k, y - op[1][1] * k)
            elif op[0] == "L":
                ctx.line_to(x + op[1][0] * k, y - op[1][1] * k)
            elif op[0] == "C":
                (a, b), (c, d), (e, f) = op[1], op[2], op[3]
                ctx.curve_to(x + a * k, y - b * k, x + c * k, y - d * k, x + e * k, y - f * k)
            else:
                ctx.close_path()

    def bounds(self, name):
        g = self.gs[name]
        from fontTools.pens.boundsPen import BoundsPen
        bp = BoundsPen(self.gs)
        g.draw(bp)
        return bp.bounds  # (xmin, ymin, xmax, ymax) en unidades de fuente o None


class Letter:
    def __init__(self, char, name, x, y, k, font):
        self.char, self.name, self.x, self.y, self.k, self.font = char, name, x, y, k, font
        b = font.bounds(name)
        if b is None:
            self.box = None
            self.center = np.array([x, y])
        else:
            x0, y0, x1, y1 = b
            self.box = (x + x0 * k, y - y1 * k, x + x1 * k, y - y0 * k)
            self.center = np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    def draw(self, ctx):
        self.font.draw(ctx, self.name, self.x, self.y, self.k)


class Word:
    """Palabra maquetada en pantalla. `anchor`: 'center' o 'left'; y = línea base."""

    def __init__(self, text, font, cap_px, x, y, tracking=0.0, anchor="center"):
        self.text, self.font = text, font
        px = cap_px * font.upm / font.cap  # tamaño de em para lograr esa altura de mayúscula
        glyphs, width = font.shape(text, px, tracking)
        x0 = x - width / 2 if anchor == "center" else x
        k = px / font.upm
        self.letters = []
        chars = list(text)
        for i, (name, gx, gy) in enumerate(glyphs):
            ch = chars[i] if i < len(chars) else ""
            L = Letter(ch, name, x0 + gx, y + gy, k, font)
            if L.box is not None:
                self.letters.append(L)
        boxes = np.array([L.box for L in self.letters])
        self.box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
        self.center = np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])
        self.width = width
        self.cap_px = cap_px


def draw_text(mask, font, text, cap_px, x, y, alpha=1.0, tracking=0.0, anchor="left"):
    """Texto simple (rótulos de la lámina) en una máscara."""
    w = Word(text, font, cap_px, x, y, tracking, anchor)
    ctx = mask.ctx
    for L in w.letters:
        L.draw(ctx)
    mask.set_alpha(alpha)
    ctx.fill()
    return w
