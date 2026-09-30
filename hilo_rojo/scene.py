"""«Hilván»: línea de tiempo (13 s, 24 fps animados en dos = 12 imágenes únicas por segundo).

Conocer   0,0–1,0   póster (solo se mueve el humo de la chimenea)
Vigilar   1,0–6,4   la aguja hilvana las huellas; el hilo se tensa y la tela se frunce; contraluz:
                    una lámpara recorre el revés y aparece un solo hilo; vuelve la luz de frente
Alertar   6,4–8,6   la lana roja va de casa en casa, paralela al hilván, fijada con puntaditas
Responder 8,6–9,9   se saca el hilván (quedan los agujeros); el frunce se relaja
Red       9,9–13,0  la cámara se aleja por pasos: la lana roja es el cordel del que cuelgan las
                    arpilleras de otros territorios
"""
import cv2
import numpy as np

from satc_intro.color import lin, linear_to_srgb
from satc_intro.geometry import catmull_rom, resample
from satc_intro.noise import smooth_noise, smoothstep
from . import territory
from .figures import needle_sprite, hershey_strokes
from .pieces import Piece
from .thread import Sprite, Stitch, Yarn, composite, composite_T, knot, running_stitch

FPS = 24
UFPS = 12
DURATION = 13.0

T = dict(hilvan=(1.0, 3.3), taut=(3.3, 4.0), dim=4.0, scan=(4.17, 5.33), full=(5.33, 6.17), back=6.17,
         wool=(6.42, 8.58), pull=(8.58, 9.42), dolly=(9.92, 11.42))


def q(t):
    """Tiempo cuantizado a la imagen única (stop-motion en dos)."""
    return np.floor(t * UFPS + 1e-6) / UFPS


class Scene:
    def __init__(self, W=1920, H=1080, seed=11):
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        self.rng = rng
        base, Tb, info = territory.build(W, H, seed=seed)
        self.base, self.Tb, self.info = base, Tb, info
        u = self.u = info["u"]
        X = lambda f: f * W
        Y = lambda f: f * H
        P = lambda fx, fy: np.array([fx * W, fy * H])
        blk = "#141516"
        # ---------------------------------------------------------------- huellas (hilván por delante)
        tx, ty = info["tower_base"]
        self.H1 = np.array([tx, ty])
        g = []   # (t_aparece, sprite)
        t0, t1 = 1.0, 1.62
        mast = [(ty, ty - 64 * u), (ty - 68 * u, ty - 132 * u), (ty - 136 * u, ty - 200 * u)]
        seq = [((tx, a), (tx + rng.normal(0, 0.5), b), 2.6 * u) for a, b in mast]
        seq += [((tx, ty - 124 * u), (tx - 96 * u, ty + 20 * u), 1.7 * u),
                ((tx, ty - 124 * u), (tx + 100 * u, ty + 12 * u), 1.7 * u),
                ((tx, ty - 194 * u), (tx + 74 * u, ty + 18 * u), 1.6 * u)]
        seq += [((tx - 15 * u, ty - 62 * u), (tx + 15 * u, ty - 62 * u), 1.8 * u),
                ((tx - 15 * u, ty - 128 * u), (tx + 15 * u, ty - 128 * u), 1.8 * u),
                ((tx - 12 * u, ty - 190 * u), (tx + 12 * u, ty - 190 * u), 1.8 * u)]
        for i, (a, b, w) in enumerate(seq):
            g.append((t0 + (t1 - t0) * i / len(seq), Stitch.render(a, b, w, blk, rng, kind="synthetic")))
        self.tower_top = np.array([tx, ty - 200 * u])
        # estacas con cinta naranja
        t0, t1 = 1.62, 2.12
        stakes = [P(0.585 + 0.026 * k, 0.800 + 0.006 * k) for k in range(4)]
        for i, sp_ in enumerate(stakes):
            tt = t0 + (t1 - t0) * i / 4
            g.append((tt, Stitch.render(sp_, sp_ - (0, 22 * u), 2.6 * u, blk, rng, kind="synthetic")))
            g.append((tt, Stitch.render(sp_ - (0, 22 * u), sp_ + (9 * u, -18 * u), 2.2 * u, "#f07a1a", rng)))
            g.append((tt, Stitch.render(sp_ - (0, 15 * u), sp_ + (9 * u, -18 * u), 2.2 * u, "#f07a1a", rng)))
        self.H2 = stakes[1]
        # aviso en el cerco
        f0, f1 = info["fence"]
        nc = f0 + (f1 - f0) * 0.58 - (0, 34 * u)
        nw, nh = 30 * u, 22 * u
        self.notice = Piece([nc + (-nw, -nh), nc + (nw, -nh), nc + (nw, nh), nc + (-nw, nh)], "#f1ede2", rng,
                            kind="plain", stitch=None, margin=8, fabric_scale=u)
        self.t_notice = 2.12
        t0, t1 = 2.2, 2.62
        tacks = [nc + (-nw + 4 * u, -nh + 4 * u), nc + (nw - 4 * u, -nh + 4 * u), nc + (nw - 4 * u, nh - 4 * u),
                 nc + (-nw + 4 * u, nh - 4 * u)]
        for i, c in enumerate(tacks):
            g.append((t0 + 0.08 * i, Stitch.render(c - (3 * u, 3 * u), c + (3 * u, 3 * u), 2.2 * u, blk, rng,
                                                   kind="synthetic")))
        for r in range(3):
            y_ = nc[1] - nh * 0.45 + r * nh * 0.42
            xx0, xx1 = nc[0] - nw * 0.62, nc[0] + nw * (0.62 if r < 2 else 0.1)
            k = 0
            x_ = xx0
            while x_ < xx1:
                g.append((t0 + 0.34 + 0.02 * r, Stitch.render((x_, y_), (x_ + 4.5 * u, y_), 1.2 * u, blk, rng)))
                x_ += 6.5 * u
        self.H3 = nc
        # toma de agua en el río
        t0, t1 = 2.72, 3.25
        bank = P(0.292, 0.622)
        into = P(0.262, 0.652)
        mid = (bank + into) / 2
        g.append((t0, Stitch.render(bank, mid, 3.4 * u, blk, rng, kind="synthetic")))
        g.append((t0 + 0.12, Stitch.render(mid, into, 3.4 * u, blk, rng, kind="synthetic")))
        g.append((t0 + 0.24, knot(bank + (4 * u, -2 * u), 5.0 * u, blk, rng)))
        self.H4 = bank
        self.hilvan = sorted(g, key=lambda x: x[0])
        # dónde está la aguja en cada tramo
        self.needle_track = [(1.0, 1.62, [self.H1, self.tower_top]), (1.62, 2.12, [stakes[0], stakes[-1]]),
                             (2.12, 2.64, [tacks[0], tacks[2]]), (2.72, 3.3, [bank, into])]

        # ---------------------------------------------------------------- el hilo por el revés
        self.back_path = [self.H1 + (0, 2 * u), self.H2, self.H3, self.H4]
        self.back_mask = self._back_mask()
        # ---------------------------------------------------------------- el frunce entre dos casas
        self._pucker_prepare(self.H2, self.H3)
        # ---------------------------------------------------------------- la lana roja
        hs = info["houses"]
        A, B, C, D = [h["door"] for h in hs]
        off = np.array([0, -16 * u])
        way = [A + (6 * u, -6 * u), A + (60 * u, -70 * u), self.H4 + off + (18 * u, -6 * u),
               (self.H4 + self.H3) / 2 + off, self.H3 + off + (-10 * u, 0), C + (-30 * u, 30 * u), C + (0, 8 * u),
               (C + B) / 2 + (70 * u, 0), B + (10 * u, -6 * u), (B + self.H2) / 2 + (0, 20 * u), self.H2 + off,
               (self.H2 + D) / 2 + (0, -30 * u), D + (-6 * u, -8 * u), D + (60 * u, -120 * u),
               (D + self.H1) / 2 + (70 * u, 0), self.H1 + (40 * u, 10 * u), self.tower_top + (60 * u, -10 * u),
               np.array([tx - 30 * u, 0.06 * H]), np.array([tx - 50 * u, -30 * u])]
        path = catmull_rom(np.array(way), 18)
        path += np.stack([smooth_noise((1, len(path)), 40, rng)[0] - 0.5,
                          smooth_noise((1, len(path)), 40, rng)[0] - 0.5], 1) * 8 * u
        self.wool = Yarn(path, 8.5 * u, "#c3241c", rng, couch_every=19 * u, couch_color="#8e160f", fuzz=1.0)
        self.house_lit = []
        # ---------------------------------------------------------------- humo de la chimenea
        self.chimney = info["chimney"]
        # ---------------------------------------------------------------- agujeros que deja el hilván
        self.holes = []
        for tt, sp in self.hilvan:
            if sp.sh is not None:
                cx = sp.x0 + sp.a.shape[1] / 2
                cy = sp.y0 + sp.a.shape[0] / 2
        hole_pts = []
        for a, b, w in seq:
            hole_pts += [a, b]
        for sp_ in stakes:
            hole_pts += [tuple(sp_), tuple(sp_ - (0, 22 * u))]
        hole_pts += [tuple(c) for c in tacks] + [tuple(bank), tuple(mid), tuple(into)]
        self.hole_sprites = [self._hole(p) for p in hole_pts]
        # luz: parpadeo de exposición por imagen
        self.flick = np.random.default_rng(seed + 99).normal(0, 0.006, 400)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = xx / W - 0.5, yy / H - 0.5
        self.vign = (1 - 0.28 * (nx * nx * 1.1 + ny * ny * 1.3) ** 1.1).astype(np.float32)
        self.rake = (1 + 0.06 * (-(nx * 0.8 + ny * 0.6))).astype(np.float32)
        self.yy, self.xx = yy, xx
        self.lamp = np.array([1.0, 0.92, 0.78], np.float32)

    # ------------------------------------------------------------------------------------------------
    def _hole(self, p):
        u = self.u
        x0, y0 = int(p[0] - 6 * u), int(p[1] - 6 * u)
        n = int(12 * u) + 1
        yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n].astype(np.float32) + 0.5
        d = np.hypot(xx - p[0], yy - p[1])
        a = np.clip(1.6 * u - d + 0.5, 0, 1).astype(np.float32) * 0.85
        rgb = np.zeros((n, n, 3), np.float32) + lin("#2a1e14") * a[..., None]
        return Sprite(x0, y0, rgb, a, None)

    def _back_mask(self):
        """Sombra del hilo del revés: un núcleo nítido y un halo (la tela la suaviza)."""
        W, H, u = self.W, self.H, self.u
        m = np.zeros((H, W), np.float32)
        pts = resample(np.array(self.back_path), 2.0)
        pts = pts + np.stack([np.sin(np.arange(len(pts)) * 0.05) * 0.8, np.cos(np.arange(len(pts)) * 0.043) * 0.8], 1)
        cv2.polylines(m, [np.round(pts * 8).astype(np.int32)], False, 1.0, max(1, int(5.5 * u)), cv2.LINE_AA, shift=3)
        core = cv2.GaussianBlur(m, (0, 0), 1.3 * u)
        halo = cv2.GaussianBlur(m, (0, 0), 7.0 * u)
        return np.clip(core * 1.25 + halo * 0.9, 0, 1)

    def _pucker_prepare(self, a, b):
        u = self.u
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        d = b - a
        L = float(np.hypot(*d))
        t = d / L
        n = np.array([-t[1], t[0]])
        pad = 130 * u
        x0 = int(max(0, min(a[0], b[0]) - pad))
        y0 = int(max(0, min(a[1], b[1]) - pad))
        x1 = int(min(self.W, max(a[0], b[0]) + pad))
        y1 = int(min(self.H, max(a[1], b[1]) + pad))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        qx, qy = xx - a[0], yy - a[1]
        s = qx * t[0] + qy * t[1]
        e = qx * n[0] + qy * n[1]
        g = np.exp(-(e / (58 * u)) ** 2) * smoothstep(-20 * u, 60 * u, s) * smoothstep(L + 20 * u, L - 60 * u, s)
        lam = 21 * u
        self.pk = dict(box=(x0, y0, x1, y1), s=s, e=e, g=g.astype(np.float32), t=t, n=n, L=L, lam=lam,
                       xx=xx, yy=yy, a=a)

    def _pucker(self, img, Tm, A):
        """Frunce real: pliegues perpendiculares al hilo, estampado comprimido, sombra rasante."""
        if A <= 0.001:
            return
        pk = self.pk
        x0, y0, x1, y1 = pk["box"]
        s, g, lam, L = pk["s"], pk["g"], pk["lam"], pk["L"]
        t, n = pk["t"], pk["n"]
        ph = 2 * np.pi * s / lam
        # de dónde viene cada punto: compresión hacia el centro + tela metida en los valles
        ds = A * g * (0.30 * (s - L / 2) + 0.42 * lam / (2 * np.pi) * np.sin(ph))
        de = A * g * 0.06 * pk["e"]
        mx = (pk["xx"] + ds * t[0] + de * n[0]).astype(np.float32)
        my = (pk["yy"] + ds * t[1] + de * n[1]).astype(np.float32)
        mx -= x0
        my -= y0
        img[y0:y1, x0:x1] = cv2.remap(np.ascontiguousarray(img[y0:y1, x0:x1]), mx, my, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        Tm[y0:y1, x0:x1] = cv2.remap(np.ascontiguousarray(Tm[y0:y1, x0:x1]), mx, my, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        # luz rasante sobre los pliegues: la pendiente a lo largo del hilo
        Ld = -0.64 * t[0] - 0.56 * t[1]
        slope = np.cos(ph)
        shade_ = 1 + A * g * (0.30 * Ld * slope - 0.12 * (1 - np.sin(ph)) * 0.5)
        img[y0:y1, x0:x1] *= np.clip(shade_, 0.6, 1.35)[..., None]
        Tm[y0:y1, x0:x1] *= (1 - 0.45 * A * g * (0.5 + 0.5 * np.sin(ph)))[..., None]

    # ------------------------------------------------------------------------------------------------
    def _smoke(self, canvas, tq):
        u = self.u
        k = int(round(tq * UFPS))
        r = np.random.default_rng(1000 + k)
        c = self.chimney
        pts = [c]
        for i in range(1, 7):
            pts.append(c + (np.sin(i * 1.3 + k * 0.7) * 10 * u * i ** 0.6 + r.normal(0, 2 * u), -i * 16 * u))
        Yarn(catmull_rom(np.array(pts), 8), 4.2 * u, "#f3efe7", r, fuzz=1.6).draw(canvas, 1e9, shadow=0.25)

    def _needle_at(self, tq):
        u = self.u
        for (a, b, pts) in self.needle_track:
            if a <= tq < b:
                f = (tq - a) / (b - a)
                p0, p1 = np.asarray(pts[0]), np.asarray(pts[1])
                p = p0 + (p1 - p0) * f
                d = p1 - p0
                d = d / (np.linalg.norm(d) + 1e-9)
                k = int(round(tq * UFPS))
                inside = (k % 2 == 0)
                back = p - d * 70 * u
                tip = p + d * (18 if inside else 40) * u
                return needle_sprite(back, tip, 3.4 * u, hide_from=0.74 if inside else None), back, "#141516"
        a, b = T["wool"]
        if a <= tq < b:
            L = self.wool.length * (tq - a) / (b - a)
            p, d = self.wool.head(L)
            k = int(round(tq * UFPS))
            inside = (k % 2 == 0)
            back = p - d * 22 * u
            tip = p + d * (60 if not inside else 36) * u
            return needle_sprite(back, tip, 5.2 * u, hide_from=0.62 if inside else None), back, "#c3241c"
        return None, None, None

    # ------------------------------------------------------------------------------------------------
    def light(self, tq):
        """(luz de frente, luz de atrás, centro de la lámpara o None)."""
        if tq < T["dim"]:
            return 1.0, 0.0, None
        s0, s1 = T["scan"]
        if tq < s0:
            return 0.55, 0.25, np.asarray(self.back_path[0])
        if tq < s1:
            f = (tq - s0) / (s1 - s0)
            pts = np.array(self.back_path)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            arc = np.concatenate([[0], np.cumsum(seg)])
            Lh = f * arc[-1]
            k = int(np.clip(np.searchsorted(arc, Lh), 1, len(pts) - 1))
            c = pts[k - 1] + (pts[k] - pts[k - 1]) * (Lh - arc[k - 1]) / max(1e-6, seg[k - 1])
            return 0.14, 1.0, c
        f0, f1 = T["full"]
        if tq < f1:
            return 0.12, 1.0, "full"
        if tq < T["back"] + 1 / UFPS:
            return 0.6, 0.35, "full"
        return 1.0, 0.0, None

    def pucker_amount(self, tq):
        a, b = T["taut"]
        if tq < a:
            return 0.0
        if tq < b:
            return float(smoothstep(a, b, tq))
        p0, p1 = T["pull"]
        if tq < p0:
            return 1.0
        return float(1.0 - 0.72 * smoothstep(p0, p1, tq))

    def hilvan_visible(self, tq):
        """Cuántas puntadas del hilván quedan (al sacarlo desaparecen en orden inverso)."""
        p0, p1 = T["pull"]
        n = len(self.hilvan)
        if tq < p0:
            return n, False
        f = np.clip((tq - p0) / (p1 - p0), 0, 1)
        return int(round(n * (1 - f))), True

    # ------------------------------------------------------------------------------------------------
    def arpillera(self, t):
        """La arpillera (en su propio encuadre), sin el mundo alrededor."""
        tq = q(t)
        img = self.base.copy()
        Tm = self.Tb.copy()
        u = self.u
        self._smoke(img, tq)
        # hilván
        nvis, pulling = self.hilvan_visible(tq)
        if tq >= self.t_notice and (not pulling or nvis > 0):
            self.notice.bake(img, Tm) if not pulling or nvis > len(self.hilvan) * 0.35 else None
        shown = [sp for (tt, sp) in self.hilvan if tt <= tq][:nvis]
        for sp in shown:
            composite(img, sp, shadow=0.55)
            composite_T(Tm, sp)
        if pulling:
            for hs_ in self.hole_sprites:
                composite(img, hs_, shadow=0)
            # el cabo suelto del hilván, tirado hacia afuera
            if nvis > 0:
                last = shown[-1]
                p = np.array([last.x0 + last.a.shape[1] / 2, last.y0 + last.a.shape[0] / 2])
                end = p + np.array([-420, 260]) * u
                mid = (p + end) / 2 + np.array([40, 80]) * u
                tail = Yarn(catmull_rom(np.array([p, mid, end]), 12), 2.4 * u, "#141516", self.rng, fuzz=0,
                            kind="floss")
                tail.draw(img, 1e9, shadow=0.5)
        # frunce
        self._pucker(img, Tm, self.pucker_amount(tq))
        # lana roja
        a, b = T["wool"]
        if tq >= a:
            L = self.wool.length * np.clip((tq - a) / (b - a), 0, 1)
            self.wool.draw(img, L)
        # aguja
        nd, back, thread_col = self._needle_at(tq)
        if nd is not None:
            composite(img, nd, shadow=0.45)
            composite_T(Tm, nd)
        # luz
        F, B, lamp_c = self.light(tq)
        k = int(round(tq * UFPS))
        out = img * (F * (1 + self.flick[k % len(self.flick)])) * self.rake[..., None]
        if B > 0:
            Te = Tm * (1 - 0.93 * self.back_mask)[..., None]
            if isinstance(lamp_c, str):
                field, gain = np.float32(1.0), 9.5
            else:
                R = 0.16 * max(self.W, self.H)
                field = (0.008 + np.exp(-((self.xx - lamp_c[0]) ** 2 + (self.yy - lamp_c[1]) ** 2)
                                        / (2 * R * R)))[..., None]
                gain = 20.0
            x = Te * self.lamp * gain * B * field
            lum = x[..., 0] * 0.3 + x[..., 1] * 0.55 + x[..., 2] * 0.15
            lum2 = 1 - np.exp(-lum)
            glow = x * (lum2 / np.maximum(lum, 1e-5))[..., None]      # conserva el color (vitral)
            glow = glow * 0.92 + lum2[..., None] * 0.08
            bloom = cv2.GaussianBlur(glow, (0, 0), 10 * u) * 0.28
            out = out + glow + bloom
        out *= self.vign[..., None]
        return np.clip(out, 0, 1)

    def frame_srgb8(self, t):
        return (linear_to_srgb(self.arpillera(t)) * 255 + 0.5).astype(np.uint8)
