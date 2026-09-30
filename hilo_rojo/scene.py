"""«Hilván»: línea de tiempo (14 s, 24 fps animados en dos = 12 imágenes únicas por segundo).

Conocer   0,0–1,0    póster: la arpillera cuelga del cordel (solo se mueve el humo de la chimenea)
Vigilar   1,0–4,0    la aguja hilvana las huellas (torre, estacas, aviso, toma de agua); tira del hilo,
                     se tensa y la tela se frunce entre dos casas
Alertar   4,0–6,4    se apaga la sala; una lámpara recorre el revés deteniéndose en cada huella y
                     aparece un solo hilo; luego toda la tela es linterna; vuelve la luz
Responder 6,4–9,9    la lana roja va de casa en casa, paralela al hilván, fijada con puntaditas; se
                     saca el hilván (quedan los agujeros y un frunce leve, como memoria)
Red       10,25–14,0 la cámara se aleja por pasos: la lana sube al cordel del que cuelgan las
                     arpilleras de otros territorios; un pulso recorre el cordel desde el nudo y
                     mece a las vecinas (la alerta viaja por la red); la tira con el título
"""
import cv2
import numpy as np

from satc_intro.color import lin, linear_to_srgb
from satc_intro.geometry import catmull_rom, resample
from satc_intro.noise import smooth_noise, smoothstep
from . import territory
from .figures import needle_sprite
from .pieces import Piece
from .thread import Sprite, Stitch, Yarn, composite, composite_T, knot
from .world import World

FPS = 24
UFPS = 12
DURATION = 14.0

T = dict(hilvan=(1.0, 3.3), taut=(3.3, 4.0), dim=4.0, scan=(4.17, 5.33), full=(5.33, 6.17), back=6.17,
         wool=(6.42, 8.67), pull=(8.75, 9.92), dolly=(10.25, 11.75), pulse=12.0)
BLK = "#141516"


def q(t):
    """Tiempo cuantizado a la imagen única (stop-motion en dos)."""
    return np.floor(t * UFPS + 1e-6) / UFPS


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


class Scene:
    def __init__(self, W=1920, H=1080, seed=11):
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        self.rng = rng
        base, Tb, info = territory.build(W, H, seed=seed)
        self.base, self.Tb, self.info = base, Tb, info
        u = self.u = info["u"]
        P = lambda fx, fy: np.array([fx * W, fy * H])
        g = []   # (t_aparece, sprite, p0, p1): cada puntada del hilván, con sus dos agujeros

        def add(tt, a, b, w, col=BLK, kind="synthetic"):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            g.append((tt, Stitch.render(a, b, w, col, rng, kind=kind), a, b))

        # ---------------------------------------------------------------- huellas (hilván por delante)
        tx, ty = info["tower_base"]
        self.H1 = np.array([tx, ty])
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
            add(t0 + (t1 - t0) * i / len(seq), a, b, w)
        self.tower_top = np.array([tx, ty - 200 * u])
        # estacas con cinta naranja
        t0, t1 = 1.62, 2.12
        stakes = info["stakes"]
        for i, sp_ in enumerate(stakes):
            tt = t0 + (t1 - t0) * i / 4
            add(tt, sp_, sp_ - (0, 22 * u), 2.6 * u)
            add(tt, sp_ - (0, 22 * u), sp_ + (9 * u, -18 * u), 2.2 * u, "#f07a1a", "floss")
            add(tt, sp_ - (0, 15 * u), sp_ + (9 * u, -18 * u), 2.2 * u, "#f07a1a", "floss")
        self.H2 = stakes[1]
        # aviso en el cerco
        f0, f1 = info["fence"]
        nc = f0 + (f1 - f0) * 0.58 - (0, 34 * u)
        nw, nh = 30 * u, 22 * u
        self.notice = Piece([nc + (-nw, -nh), nc + (nw, -nh), nc + (nw, nh), nc + (-nw, nh)], "#f1ede2", rng,
                            kind="plain", stitch=None, margin=8, fabric_scale=u)
        self.t_notice = 2.12
        self.notice.t_place = self.t_notice
        t0 = 2.2
        tacks = [nc + (-nw + 4 * u, -nh + 4 * u), nc + (nw - 4 * u, -nh + 4 * u), nc + (nw - 4 * u, nh - 4 * u),
                 nc + (-nw + 4 * u, nh - 4 * u)]
        for i, c in enumerate(tacks):
            add(t0 + 0.08 * i, c - (3 * u, 3 * u), c + (3 * u, 3 * u), 2.2 * u)
        for r in range(3):
            y_ = nc[1] - nh * 0.45 + r * nh * 0.42
            xx0, xx1 = nc[0] - nw * 0.62, nc[0] + nw * (0.62 if r < 2 else 0.1)
            x_ = xx0
            while x_ < xx1:
                add(t0 + 0.34 + 0.02 * r, (x_, y_), (x_ + 4.5 * u, y_), 1.2 * u, BLK, "floss")
                x_ += 6.5 * u
        self.H3 = nc
        # toma de agua en el río
        t0 = 2.72
        bank, into = info["intake"]
        mid = (bank + into) / 2
        add(t0, bank, mid, 3.4 * u)
        add(t0 + 0.12, mid, into, 3.4 * u)
        kp = bank + (4 * u, -2 * u)
        g.append((t0 + 0.24, knot(kp, 5.0 * u, BLK, rng), kp, kp))
        self.H4 = bank
        self.knot_pt = kp
        self.hilvan = sorted(g, key=lambda x: x[0])
        # huella de cada puntada (para saber cuándo la aguja salta por el revés a otra)
        self.huella_of = []
        for (tt, _, _, _) in self.hilvan:
            self.huella_of.append(0 if tt < 1.62 else 1 if tt < 2.12 else 2 if tt < 2.7 else 3)

        # ---------------------------------------------------------------- el hilo por el revés
        self.back_path = [self.H1 + (0, 2 * u), self.H2, self.H3, self.H4]
        self.back_mask = self._back_mask()
        # ---------------------------------------------------------------- el frunce entre dos casas
        self._pucker_prepare(self.H2, self.H3)
        # ---------------------------------------------------------------- la lana roja
        hs = info["houses"]
        A, B, C, D = [h["door"] for h in hs]
        off = np.array([0, -16 * u])
        if not info["portrait"]:
            way = [A + (6 * u, -6 * u), A + (60 * u, -70 * u), self.H4 + off + (18 * u, -6 * u),
                   (self.H4 + self.H3) / 2 + off, self.H3 + off + (-10 * u, 0), C + (-30 * u, 30 * u), C + (0, 8 * u),
                   (C + B) / 2 + (70 * u, 0), B + (10 * u, -6 * u), (B + self.H2) / 2 + (0, 20 * u), self.H2 + off,
                   (self.H2 + D) / 2 + (0, -30 * u), D + (-6 * u, -8 * u), D + (60 * u, -120 * u),
                   (D + self.H1) / 2 + (70 * u, 0), self.H1 + (40 * u, 10 * u), self.tower_top + (60 * u, -10 * u),
                   np.array([tx - 30 * u, 0.06 * H]), np.array([tx - 50 * u, -30 * u])]
        else:
            # en vertical la lana sube: de la casa de abajo, zigzagueando de casa en casa, hasta la torre
            way = [B + (6 * u, -6 * u), (B + D) / 2 + (0, -40 * u), D + (-6 * u, -8 * u), D + (-40 * u, -90 * u),
                   self.H2 + off + (30 * u, 0), (self.H2 + self.H4) / 2 + off + (0, 26 * u), self.H4 + off + (22 * u, -4 * u),
                   A + (10 * u, -6 * u), A + (70 * u, -60 * u), (A + self.H3) / 2 + (0, -10 * u), self.H3 + off + (-10 * u, 0),
                   C + (-10 * u, 8 * u), C + (40 * u, -80 * u), (C + self.H1) / 2 + (60 * u, 0), self.H1 + (40 * u, 10 * u),
                   self.tower_top + (60 * u, -10 * u), np.array([tx - 30 * u, 0.04 * H]), np.array([tx - 50 * u, -30 * u])]
        path = catmull_rom(np.array(way), 18)
        path += np.stack([smooth_noise((1, len(path)), 40, rng)[0] - 0.5,
                          smooth_noise((1, len(path)), 40, rng)[0] - 0.5], 1) * 8 * u
        self.wool = Yarn(path, 8.5 * u, "#c3241c", rng, couch_every=19 * u, couch_color="#8e160f", fuzz=1.0)
        # ---------------------------------------------------------------- humo de la chimenea
        self.chimney = info["chimney"]
        # ---------------------------------------------------------------- agujeros que deja el hilván
        pts = []
        for (tt, sp, a, b) in self.hilvan:
            for p in (a, b):
                if all(np.hypot(*(p - q_)) > 2.5 * u for q_ in pts):
                    pts.append(p)
        self.hole_sprites = [self._hole(p, np.random.default_rng(int(p[0] * 13 + p[1]))) for p in pts]
        # luz: parpadeo de exposición por imagen
        self.flick = np.random.default_rng(seed + 99).normal(0, 0.006, 400)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.yy, self.xx = yy, xx
        self.lamp = np.array([1.0, 0.93, 0.82], np.float32)
        # ---------------------------------------------------------------- el mundo (cordel, vecinas, título)
        self.layout = layout_for(W, H, self)
        self.world = World(W, H, self.layout)

    # ------------------------------------------------------------------------------------------------
    def _hole(self, p, rng):
        """Agujero que deja una puntada al sacarla: hueco oscuro, borde de fibras empujadas."""
        u = self.u
        R = 2.0 * u
        x0, y0 = int(p[0] - 6 * u), int(p[1] - 6 * u)
        n = int(12 * u) + 2
        yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n].astype(np.float32) + 0.5
        dx, dy = xx - p[0], yy - p[1]
        d = np.hypot(dx, dy)
        core = np.clip(R * 0.62 - np.hypot(dx + 0.3 * u, dy + 0.3 * u) + 0.5, 0, 1)
        rim = np.clip(1 - np.abs(d - R * 1.05) / (0.7 * u), 0, 1)
        lit = np.clip((dx * 0.64 + dy * 0.56) / (d + 1e-3), 0, 1)       # borde de abajo-derecha iluminado
        a = np.clip(core * 0.8 + rim * (0.18 + 0.3 * lit), 0, 1).astype(np.float32)
        rgb = (lin("#1e150e")[None, None, :] * core[..., None] * 0.8
               + lin("#efe4cf")[None, None, :] * (rim * (0.18 + 0.3 * lit))[..., None])
        # una fibra suelta
        ang = rng.uniform(0, 2 * np.pi)
        fib = np.zeros((n, n), np.float32)
        a0 = np.array([p[0] - x0, p[1] - y0]) + np.array([np.cos(ang), np.sin(ang)]) * R
        a1 = a0 + np.array([np.cos(ang + 0.5), np.sin(ang + 0.5)]) * 3.2 * u
        cv2.line(fib, (int(a0[0] * 4), int(a0[1] * 4)), (int(a1[0] * 4), int(a1[1] * 4)), 0.5, 1, cv2.LINE_AA, shift=2)
        rgb = rgb * (1 - fib[..., None]) + lin("#3a3430")[None, None, :] * fib[..., None]
        a = np.maximum(a, fib)
        return Sprite(x0, y0, rgb.astype(np.float32), a, None)

    def _back_mask(self):
        """Sombra del hilo del revés: un núcleo nítido (el hilo toca la tela) y un halo leve."""
        W, H, u = self.W, self.H, self.u
        m = np.zeros((H, W), np.float32)
        pts = resample(np.array(self.back_path), 2.0)
        pts = pts + np.stack([np.sin(np.arange(len(pts)) * 0.05) * 0.8, np.cos(np.arange(len(pts)) * 0.043) * 0.8], 1)
        cv2.polylines(m, [np.round(pts * 8).astype(np.int32)], False, 1.0, max(1, int(5.4 * u)), cv2.LINE_AA, shift=3)
        core = cv2.GaussianBlur(m, (0, 0), 0.9 * u)
        halo = cv2.GaussianBlur(m, (0, 0), 6.0 * u)
        return np.clip(core * 1.5 + halo * 0.55, 0, 1)

    def _pucker_prepare(self, a, b):
        u = self.u
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        d = b - a
        L = float(np.hypot(*d))
        t = d / L
        n = np.array([-t[1], t[0]])
        pad = 150 * u
        x0 = int(max(0, min(a[0], b[0]) - pad))
        y0 = int(max(0, min(a[1], b[1]) - pad))
        x1 = int(min(self.W, max(a[0], b[0]) + pad))
        y1 = int(min(self.H, max(a[1], b[1]) + pad))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        qx, qy = xx - a[0], yy - a[1]
        s = qx * t[0] + qy * t[1]
        e = qx * n[0] + qy * n[1]
        rngp = np.random.default_rng(3)
        # los pliegues no son perfectos: el ancho de la banda y la fase varían a lo largo
        wob = (smooth_noise(s.shape, 40 * u, rngp) - 0.5)
        g = (np.exp(-(e / (68 * u * (1 + 0.25 * wob))) ** 2) * smoothstep(-20 * u, 60 * u, s)
             * smoothstep(L + 20 * u, L - 60 * u, s))
        lam = 22 * u
        self.pk = dict(box=(x0, y0, x1, y1), s=s, e=e, g=g.astype(np.float32), t=t, n=n, L=L, lam=lam,
                       xx=xx, yy=yy, a=a, ph0=(wob * 1.4).astype(np.float32))

    def _pucker(self, img, Tm, A):
        """Frunce real: pliegues perpendiculares al hilo, estampado comprimido, sombra rasante."""
        if A <= 0.001:
            return
        pk = self.pk
        x0, y0, x1, y1 = pk["box"]
        s, g, lam, L = pk["s"], pk["g"], pk["lam"], pk["L"]
        t, n = pk["t"], pk["n"]
        ph = 2 * np.pi * s / lam + pk["ph0"] * (1 - np.abs(pk["e"]) / (140 * self.u)).clip(0, 1)
        ds = A * g * (0.40 * (s - L / 2) + 0.55 * lam / (2 * np.pi) * np.sin(ph))
        de = A * g * 0.07 * pk["e"]
        mx = (pk["xx"] + ds * t[0] + de * n[0]).astype(np.float32) - x0
        my = (pk["yy"] + ds * t[1] + de * n[1]).astype(np.float32) - y0
        img[y0:y1, x0:x1] = cv2.remap(np.ascontiguousarray(img[y0:y1, x0:x1]), mx, my, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        Tm[y0:y1, x0:x1] = cv2.remap(np.ascontiguousarray(Tm[y0:y1, x0:x1]), mx, my, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        Ld = -0.64 * t[0] - 0.56 * t[1]
        slope = np.cos(ph)
        shade_ = 1 + A * g * (0.62 * Ld * slope - 0.12 * (1 - np.sin(ph)))
        img[y0:y1, x0:x1] *= np.clip(shade_, 0.55, 1.4)[..., None]
        Tm[y0:y1, x0:x1] *= (1 - 0.5 * A * g * (0.5 + 0.5 * np.sin(ph)))[..., None]

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

    def _stitch_state(self, tq):
        last = nxt = None
        il = inx = None
        for i, item in enumerate(self.hilvan):
            if item[0] <= tq + 1e-6:
                last, il = item, i
            else:
                nxt, inx = item, i
                break
        return last, nxt, il, inx

    def _thread(self, a, b, w, color, rng, sag=0.12, kind="floss"):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        dist = float(np.hypot(*(b - a)))
        if dist < 2:
            return []
        n = _unit(b - a)
        perp = np.array([-n[1], n[0]])
        mid = (a + b) / 2 + np.array([0, sag * dist]) + perp * rng.normal(0, 0.03) * dist
        return Yarn(catmull_rom(np.array([a, mid, b]), 14), w, color, rng, fuzz=0, kind=kind).chunks

    def _needle_layers(self, tq):
        """La aguja y su hilo en este instante: lista de sprites (el hilo primero)."""
        u = self.u
        k = int(round(tq * UFPS))
        rj = np.random.default_rng(900 + k)
        out = []
        h0, h1 = T["hilvan"]
        if h0 <= tq < h1:
            last, nxt, il, inx = self._stitch_state(tq)
            th = np.deg2rad(52 + rj.normal(0, 4))
            v = np.array([np.cos(th), np.sin(th)])            # de la punta hacia el ojo
            Ln, wn = 230 * u, 7.4 * u
            inside = (k % 2 == 0) and nxt is not None and (nxt[0] - tq) < 0.3
            jump = (last is None) or (nxt is not None and self.huella_of[inx] != self.huella_of[il])
            if inside:
                entry = nxt[2]
                Lp = Ln * 0.86
                tip = entry - v * 0.2 * Lp
                eye = entry + v * 0.8 * Lp
                nd = needle_sprite(eye, tip, wn, hide_from=0.8, lift=0.6)
            else:
                ref = last[3] if last is not None else nxt[2]
                dd = _unit((nxt[2] - ref) if nxt is not None else (1.0, -0.3))
                tip = ref + dd * 18 * u + np.array([0, -5 * u])
                eye = tip + v * Ln
                nd = needle_sprite(eye, tip, wn, lift=1.3)
            eye_pt = eye + (tip - eye) * 0.07
            if jump and inside:
                anchor = nxt[2] + _unit(v) * 10 * u
            elif last is not None:
                anchor = last[3]
            else:
                anchor = None
            if anchor is None:
                out += self._thread(eye_pt, eye_pt + np.array([-12 * u, 70 * u]), 2.5 * u, BLK, rj, sag=0.0)
            else:
                out += self._thread(anchor, eye_pt, 2.5 * u, BLK, rj, sag=0.10)
            out.append(nd)
            return out
        a, b = T["taut"]
        if a <= tq < b - 0.04:
            # tira del hilo desde el nudo: el hilo queda tenso y la tela se frunce
            f = float(smoothstep(a, b - 0.1, tq))
            pull = _unit((-1.0, 0.22))
            eye = self.knot_pt + pull * (70 * u + 900 * u * f ** 1.6)
            tip = eye + _unit(pull + np.array([0.1, -0.35])) * 230 * u
            out += self._thread(self.knot_pt, eye, 2.5 * u, BLK, rj, sag=0.0)
            out.append(needle_sprite(eye, tip, 7.4 * u, lift=1.6))
            return out
        a, b = T["wool"]
        if a <= tq < b:
            L = self.wool.length * (tq - a) / (b - a)
            p, d = self.wool.head(L)
            inside = (k % 2 == 0)
            th = np.deg2rad(58 + rj.normal(0, 4))
            v = np.array([np.cos(th), np.sin(th)])
            Ln, wn = 200 * u, 7.2 * u
            if inside:
                Lp = Ln * 0.86
                tip = p - v * 0.2 * Lp
                eye = p + v * 0.8 * Lp
                nd = needle_sprite(eye, tip, wn, hide_from=0.8, lift=0.6)
            else:
                tip = p + d * 22 * u + np.array([0, -5 * u])
                eye = tip + v * Ln
                nd = needle_sprite(eye, tip, wn, lift=1.3)
            eye_pt = eye + (tip - eye) * 0.07
            out += self._thread(p, eye_pt, 8.0 * u, "#c3241c", rj, sag=0.16, kind="wool")
            out.append(nd)
            return out
        return out

    # ------------------------------------------------------------------------------------------------
    def light(self, tq):
        """(luz de frente, luz de atrás, lámpara): lámpara = dict(c, R) o 'full'."""
        d0 = T["dim"]
        s0, s1 = T["scan"]
        f0, f1 = T["full"]
        bk = T["back"]
        big = max(self.W, self.H)
        if tq < d0:
            return 1.0, 0.0, None
        if tq < s0:                                   # se apaga la sala
            i = int(round((tq - d0) * UFPS))
            return (0.42, 0.12)[min(i, 1)], (0.25, 0.6)[min(i, 1)], dict(c=self.back_path[0], R=0.13 * big)
        if tq < s1:                                   # la lámpara se detiene en cada huella
            i = int(round((tq - s0) * UFPS))
            n = int(round((s1 - s0) * UFPS))
            stops = np.array(self.back_path, np.float64)
            # 4 paradas de 2 imágenes y 3 tránsitos de 2 imágenes (14 en total)
            sched = []
            for j in range(4):
                sched += [j, j]
                if j < 3:
                    sched += [j + 0.34, j + 0.67]
            x = sched[min(i, len(sched) - 1)] if n >= len(sched) else sched[int(i * len(sched) / n)]
            j = int(np.floor(x))
            fr = x - j
            c = stops[j] if fr == 0 else stops[j] + (stops[min(j + 1, 3)] - stops[j]) * smoothstep(0, 1, fr)
            return 0.045, 1.0, dict(c=c, R=0.12 * big)
        if tq < f1:                                   # toda la tela es linterna
            i = int(round((tq - f0) * UFPS))
            if i < 2:
                return 0.045, 1.0, dict(c=self.back_path[-1] * 0.5 + np.array([self.W, self.H]) * 0.25,
                                        R=(0.30, 0.55)[i] * big)
            return 0.045, 1.0, "full"
        if tq < bk + 3 / UFPS:                        # vuelve la luz de la sala
            i = int(round((tq - bk) * UFPS))
            return (0.3, 0.62, 0.9)[min(i, 2)], (0.6, 0.25, 0.08)[min(i, 2)], "full"
        return 1.0, 0.0, None

    def pucker_amount(self, tq):
        a, b = T["taut"]
        if tq < a:
            return 0.0
        if tq < b:
            return float(smoothstep(a, b - 0.1, tq))
        p0, p1 = T["pull"]
        if tq < p0:
            return 1.0
        return float(1.0 - 0.70 * smoothstep(p0, p1, tq))

    def hilvan_visible(self, tq):
        """Cuántas puntadas del hilván quedan (al sacarlo desaparecen en orden inverso)."""
        p0, p1 = T["pull"]
        n = len(self.hilvan)
        if tq < p0:
            return n, False
        f = np.clip((tq - p0) / (p1 - p0 - 0.15), 0, 1)
        return int(round(n * (1 - f))), True

    # ------------------------------------------------------------------------------------------------
    def arpillera(self, t):
        """La arpillera (en su propio encuadre) ya iluminada, y la luz de la sala."""
        tq = q(t)
        img = self.base.copy()
        Tm = self.Tb.copy()
        u = self.u
        self._smoke(img, tq)
        # hilván
        nvis, pulling = self.hilvan_visible(tq)
        if tq >= self.t_notice and (not pulling or nvis > len(self.hilvan) * 0.35):
            if tq < self.t_notice + 1 / UFPS - 1e-6:
                self.notice.draw(img, tq)                 # un cuadro levantado (stop-motion)
            else:
                self.notice.bake(img, Tm)
        shown = [it for it in self.hilvan if it[0] <= tq + 1e-6][:nvis]
        if pulling:
            for hs_ in self.hole_sprites:
                composite(img, hs_, shadow=0)
        for (_, sp, _, _) in shown:
            composite(img, sp, shadow=0.55)
            composite_T(Tm, sp)
        if pulling and 0 < nvis < len(self.hilvan):
            # el cabo del hilván, tirado hacia afuera
            p = shown[-1][3]
            rj = np.random.default_rng(int(tq * 100))
            end = p + np.array([-460, 300]) * u
            for sp in self._thread(p, end, 2.5 * u, BLK, rj, sag=0.02):
                composite(img, sp, 0.5)
        # frunce
        self._pucker(img, Tm, self.pucker_amount(tq))
        # lana roja
        a, b = T["wool"]
        if tq >= a:
            L = self.wool.length * np.clip((tq - a) / (b - a), 0, 1)
            self.wool.draw(img, L)
        # aguja
        for sp in self._needle_layers(tq):
            composite(img, sp, shadow=0.45)
            composite_T(Tm, sp)
        # luz
        F, B, lamp = self.light(tq)
        out = img * F
        if B > 0:
            Te = Tm * (1 - 0.95 * self.back_mask)[..., None]
            if isinstance(lamp, str):
                field, gain = np.float32(1.0), 6.5
            else:
                c, R = lamp["c"], lamp["R"]
                field = (0.004 + np.exp(-((self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2) / (2 * R * R)))[..., None]
                gain = 9.0
            x = Te * self.lamp * gain * B * field
            lum = x[..., 0] * 0.3 + x[..., 1] * 0.55 + x[..., 2] * 0.15
            lum2 = lum / (1 + lum)                                    # Reinhard: conserva el vitral
            glow = x * (lum2 / np.maximum(lum, 1e-5))[..., None]
            bloom = cv2.GaussianBlur(glow, (0, 0), 9 * u) * 0.22
            out = out + glow + bloom
        return np.clip(out, 0, 1), (F, B, lamp)

    # ------------------------------------------------------------------------------------------------
    def camera(self, tq):
        """Acercamiento y centro del plano del cordel: quieta hasta el alejamiento, que va por pasos."""
        L = self.layout
        a, b = T["dolly"]
        e = float(smoothstep(a, b, tq))
        z = float(np.exp(np.log(L["z0"]) * (1 - e)))
        c0 = np.asarray(L["c0"], np.float64)
        c1 = np.array([self.W / 2, self.H / 2])
        f = (L["z0"] - z) / (L["z0"] - 1) if L["z0"] != 1 else 1.0
        return z, c0 + (c1 - c0) * f

    def frame(self, t):
        tq = q(t)
        img, (F, B, lamp) = self.arpillera(t)
        z, c = self.camera(tq)
        k = int(round(tq * UFPS))
        tau = tq - T["pulse"]
        if self.world.pulse_active(tau):
            return self.world.render_pulse(img, z, c, tau, flick=float(self.flick[k % len(self.flick)]))
        return self.world.render(img, z, c, F=F, B=B, lamp=lamp, flick=float(self.flick[k % len(self.flick)]),
                                 wool_tied=tq >= T["wool"][1] - 1e-6)

    def frame_srgb8(self, t):
        return (linear_to_srgb(self.frame(t)) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------
# encuadres
# ------------------------------------------------------------------------------------------------------
TITLE = [("Sistema de Alerta", "scripts"), ("Temprana Comunitario", "scripts")]
SUB = "RED COMUNITARIA DE ALERTA ENERGÉTICA"


def layout_for(W, H, sc):
    """Dónde cuelga cada arpillera en el encuadre final (mundo = último cuadro). Las medidas están
    pensadas para 1920×1080 (o 1080×1920) y se escalan al tamaño pedido."""
    from .territory import build_desert, build_towers, exit_points
    ex = exit_points(sc.wool.pts, sc.u)
    land = W >= H
    f = W / (1920 if land else 1080)
    F = lambda v: v * f
    nb_w, nb_h = int(round(1280 * f)), int(round(960 * f))
    des, des_ex = build_desert(nb_w, nb_h)
    tow, tow_ex = build_towers(int(round(1200 * f)), nb_h)
    nb_u = min(nb_w, nb_h) / 1080
    if land:
        main_x = (F(480), F(1440))
        z0 = 1.8
        title = dict(rect=tuple(F(v) for v in (560, 800, 1360, 995)), font="scriptc", lines=[
            (F(882), F(46), [("Sistema", "floss"), ("de", "floss"), ("Alerta", "wool")]),
            (F(956), F(46), [("Temprana", "floss"), ("Comunitario", "floss")])])
        line = (F(980), F(165), 0.00003 / f)
        nbx = [(F(-170), F(430)), (F(1500), F(2060))]
    else:
        main_x = (F(205), F(875))
        z0 = 1.52
        title = dict(rect=tuple(F(v) for v in (130, 1655, 950, 1845)), font="scriptc", lines=[
            (F(1737), F(44), [("Sistema", "floss"), ("de", "floss"), ("Alerta", "wool")]),
            (F(1807), F(44), [("Temprana", "floss"), ("Comunitario", "floss")])])
        line = (F(540), F(325), 0.00003 / f)
        nbx = [(F(-560), F(110)), (F(970), F(1600))]
    L = dict(line=line, main=dict(size=(W, H), x=main_x),
             neighbors=[dict(img=des, x=nbx[0], exit=des_ex, wool_w=8.0 * nb_u),
                        dict(img=tow, x=nbx[1], exit=tow_ex, wool_w=8.0 * nb_u)],
             title=title, wool_exit=ex, wool_w=8.5 * sc.u, z0=z0)
    # centro inicial: la tela arriba, con un margen donde se ven el cordel y los perritos
    xm, ym, k = L["line"]
    y_top = ym + 3 * f
    cy = y_top + (H / 2 - F(64)) / z0
    L["c0"] = ((main_x[0] + main_x[1]) / 2, cy)
    return L
