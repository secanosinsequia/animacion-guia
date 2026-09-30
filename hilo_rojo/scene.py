"""«Hilván»: línea de tiempo (15 s, 24 fps animados en dos = 12 imágenes únicas por segundo).

Conocer   0,0–1,0    póster: la arpillera cuelga de un cordel de cáñamo; una aguja enhebrada con hilo
                     negro cuelga desde fuera del cuadro, meciéndose (algo está por coserse)
Vigilar   1,0–4,4    la aguja hilvana las señales tempranas: estacas de topógrafo, un AVISO prendido
                     con alfiler de gancho, una toma de agua con su caseta y, al final, la torre de
                     alta tensión, solo como contorno de puntadas largas; tira del hilo: la tela se
                     frunce entre dos casas, que se acercan
Alertar   4,4–7,5    se apaga la sala; una lámpara recorre el revés deteniéndose en cada huella y
                     aparece un solo hilo; luego toda la tela es linterna (se lee el saco harinero)
Responder 7,5–11,0   la lana roja nace donde empieza el hilo y lo recorre por delante; de ahí salen
                     ramales a cada casa (se encienden las ventanas, la gente levanta los brazos);
                     la lana sube y se anuda al cordel; una vecina tira del hilván y lo junta en un
                     ovillo: salen las puntadas (la torre al último); quedan los agujeros
Red       11,2–15,0  la cámara se aleja por pasos; desde el nudo, el rojo recorre el cordel de
                     arpillera en arpillera y las vecinas se mecen; la tira con el título
"""
import cv2
import numpy as np

from satc_intro.color import lin, linear_to_srgb
from satc_intro.geometry import catmull_rom, resample
from satc_intro.noise import smooth_noise, smoothstep
from . import territory
from .figures import needle_sprite, doll, doll_layer
from .signals import build_signals
from .thread import Sprite, Yarn, composite, composite_T, knot
from .world import World

FPS = 24
UFPS = 12
DURATION = 15.0

T = dict(stakes=(1.0, 1.72), notice=(1.76, 2.36), intake=(2.4, 2.96), pylon=(3.0, 3.66), taut=(3.7, 4.36),
         dim=(4.4, 4.64), scan=(4.64, 5.64), full=(5.64, 7.0), back=(7.0, 7.5), wool=(7.5, 8.55),
         spurs=(8.55, 9.3), rise=(9.3, 9.62), pull=(9.72, 11.0), dolly=(11.2, 12.7), front=(12.8, 13.6))
BLK = "#141516"
WOOL = "#c3241c"


def q(t):
    """Tiempo cuantizado a la imagen única (stop-motion en dos)."""
    return np.floor(t * UFPS + 1e-6) / UFPS


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def _wobbly(pts, rng, amp, step):
    """Camino a mano: la lana nunca va derecha."""
    path = catmull_rom(np.array(pts, np.float64), 14)
    path = resample(path, step)
    n = len(path)
    off = np.stack([smooth_noise((1, n), 30, rng)[0] - 0.5, smooth_noise((1, n), 30, rng)[0] - 0.5], 1)
    return path + off * amp


class Scene:
    def __init__(self, W=1920, H=1080, seed=11):
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        self.rng = rng
        base, Tb, info = territory.build(W, H, seed=seed)
        self.base, self.Tb, self.info = base, Tb, info
        u = self.u = info["u"]
        self.portrait = info["portrait"]
        # ---------------------------------------------------------------- las huellas (hilván)
        sig = build_signals(info, u, rng, T)
        self.hilvan = sig["stitches"]                 # (t, sprite, p0, p1, huella)
        self.extra = sig["extra"]                     # (t, sprite, huella)
        self.sig_pieces = sig["pieces"]               # (t, Piece, huella)
        Hh = sig["H"]
        self.H1, self.H2, self.H3, self.H4 = Hh["H1"], Hh["H2"], Hh["H3"], Hh["H4"]
        self.pylon_top = sig["pylon_top"]
        self.first_stitch = self.hilvan[0][2]
        self.last_stitch = next(it for it in reversed(self.hilvan) if it[4] == 3)[3]
        # ---------------------------------------------------------------- el hilo por el revés
        self.back_path = [self.H2, self.H3, self.H4, self.H1]
        self.back_mask = self._back_mask()
        # a contraluz, cada agujero del hilván es una estrellita de luz
        self.pinholes = self._pinholes()
        # ---------------------------------------------------------------- el frunce entre dos casas
        self._pucker_prepare([(self.H2, self.H3), (self.H3, self.H4)])
        # ---------------------------------------------------------------- la lana roja
        self._wool_prepare(rng)
        # ---------------------------------------------------------------- vecinas y vecinos
        self._dolls_prepare()
        # ---------------------------------------------------------------- humo de la chimenea
        self.chimney = info["chimney"]
        # ---------------------------------------------------------------- agujeros que quedan
        pts = []
        for it in self.hilvan:
            for p in (it[2], it[3]):
                if all(np.hypot(*(p - q_)) > 3.2 * u for q_ in pts):
                    pts.append(p)
        self.hole_sprites = [self._hole(p, np.random.default_rng(int(p[0] * 13 + p[1]))) for p in pts]
        # luz y grano
        self.flick = np.random.default_rng(seed + 99).normal(0, 0.009, 400)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.yy, self.xx = yy, xx
        self.lamp = np.array([1.0, 0.86, 0.64], np.float32)        # luz de tarde, cálida
        # ---------------------------------------------------------------- el mundo
        self.layout = layout_for(W, H, self)
        self.world = World(W, H, self.layout)

    # ------------------------------------------------------------------------------------------------
    def _hole(self, p, rng):
        """Agujero que deja una puntada al sacarla: hueco de 4–5 px, anillo de trama corrida y pelusa."""
        u = self.u
        R = 2.4 * u
        x0, y0 = int(p[0] - 8 * u), int(p[1] - 8 * u)
        n = int(16 * u) + 2
        yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n].astype(np.float32) + 0.5
        dx, dy = xx - p[0], yy - p[1]
        d = np.hypot(dx, dy)
        wob = 1 + 0.18 * np.sin(np.arctan2(dy, dx) * 3 + rng.uniform(0, 6))
        core = np.clip(R * 0.62 * wob - d + 0.5, 0, 1)
        rim = np.clip(1 - np.abs(d - R * 1.15 * wob) / (0.8 * u), 0, 1)
        lit = np.clip((dx * 0.64 + dy * 0.56) / (d + 1e-3), 0, 1)
        a = np.clip(core * 0.85 + rim * (0.22 + 0.32 * lit), 0, 1).astype(np.float32)
        rgb = (lin("#1e150e")[None, None, :] * core[..., None] * 0.85
               + lin("#efe4cf")[None, None, :] * (rim * (0.22 + 0.32 * lit))[..., None])
        fib = np.zeros((n, n), np.float32)
        for _ in range(2):
            ang = rng.uniform(0, 2 * np.pi)
            a0 = np.array([p[0] - x0, p[1] - y0]) + np.array([np.cos(ang), np.sin(ang)]) * R
            a1 = a0 + np.array([np.cos(ang + rng.normal(0, 0.6)), np.sin(ang + rng.normal(0, 0.6))]) * 3.5 * u
            cv2.line(fib, (int(a0[0] * 4), int(a0[1] * 4)), (int(a1[0] * 4), int(a1[1] * 4)), 0.55, 1, cv2.LINE_AA,
                     shift=2)
        rgb = rgb * (1 - fib[..., None]) + lin("#3a3430")[None, None, :] * fib[..., None]
        a = np.maximum(a, fib)
        return Sprite(x0, y0, rgb.astype(np.float32), a, None)

    def _back_mask(self):
        """Sombra del hilo del revés: núcleo nítido de 5–6 px (el hilo toca la tela) y un halo corto."""
        W, H, u = self.W, self.H, self.u
        m = np.zeros((H * 2, W * 2), np.float32)
        pts = resample(np.array(self.back_path), 2.0)
        pts = pts + np.stack([np.sin(np.arange(len(pts)) * 0.05) * 0.8, np.cos(np.arange(len(pts)) * 0.043) * 0.8], 1)
        cv2.polylines(m, [np.round(pts * 16).astype(np.int32)], False, 1.0, max(1, int(11.0 * u)), cv2.LINE_AA, shift=3)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
        core = cv2.GaussianBlur(m, (0, 0), 0.5 * u)
        halo = cv2.GaussianBlur(m, (0, 0), 2.6 * u)
        return np.clip(core * 1.35 + halo * 0.45, 0, 1)

    def _pinholes(self):
        u = self.u
        m = np.zeros((self.H, self.W), np.float32)
        for it in self.hilvan:
            if np.hypot(*(it[3] - it[2])) < 7 * u:          # las letras del aviso no: serían una mancha
                continue
            for p in (it[2], it[3]):
                cv2.circle(m, (int(p[0] * 4), int(p[1] * 4)), max(1, int(1.5 * u * 4)), 1.0, -1, cv2.LINE_AA, shift=2)
        return np.clip(cv2.GaussianBlur(m, (0, 0), 0.9 * u) * 1.4, 0, 1)

    # ------------------------------------------------------------------------------------------------
    def _pucker_prepare(self, segs):
        """Frunce a lo largo del hilo (tramos segs): pliegues perpendiculares al hilo, de crestas redondas y
        valles en V, y un tirón que acerca lo que queda en los extremos. Se precalcula para A = 1."""
        u, W, H = self.u, self.W, self.H
        x0, y0, x1, y1 = 0, 0, W, H                      # toda la tela: el tirón llega lejos
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        DX = np.zeros_like(xx)
        DY = np.zeros_like(xx)
        S = np.zeros_like(xx)
        TM = np.zeros_like(xx)
        G = np.zeros_like(xx)
        rngp = np.random.default_rng(3)
        lam = 23 * u
        for (a, b) in segs:
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            d = b - a
            L = float(np.hypot(*d))
            t = d / L
            n = np.array([-t[1], t[0]])
            qx, qy = xx - a[0], yy - a[1]
            s_ = qx * t[0] + qy * t[1]
            e = qx * n[0] + qy * n[1]
            wob = (smooth_noise(s_.shape, 40 * u, rngp) - 0.5)
            g = (np.exp(-(e / (78 * u * (1 + 0.25 * wob))) ** 2) * smoothstep(-26 * u, 34 * u, s_)
                 * smoothstep(L + 26 * u, L - 34 * u, s_))
            beyond = np.maximum(np.maximum(-s_, s_ - L), 0)
            tug = np.exp(-(e / (230 * u)) ** 2) * np.exp(-beyond / (360 * u)) * np.clip((s_ - L / 2) / (L / 2), -1, 1)
            # pliegues desparejos: la separación y la fase cambian a lo largo (y entre un lado y el otro)
            wob2 = (smooth_noise(s_.shape, 90 * u, rngp) - 0.5)
            ph = 2 * np.pi * s_ / (lam * (1 + 0.45 * wob2)) + wob * 3.2 + e / (60 * u) * 0.9
            gather = 0.18 * L
            ds = g * 0.18 * (np.clip(s_, 0, L) - L / 2) + gather / 2 * tug * (1 - g) + g * 0.45 * lam / (2 * np.pi) * np.sin(ph)
            de = g * 0.05 * e
            DX += ds * t[0] + de * n[0]
            DY += ds * t[1] + de * n[1]
            hgt = np.abs(np.sin(ph / 2)) ** 0.5
            dh = np.gradient(hgt, axis=1) * t[0] + np.gradient(hgt, axis=0) * t[1]
            Ld = -0.64 * t[0] - 0.56 * t[1]
            S += g * (2.8 * lam / (2 * np.pi) * dh * Ld * 0.5 - 0.2 * (1 - hgt))
            TM += g * hgt
            G = np.maximum(G, g)
        # nada se mueve en la orilla de la arpillera (el borde de festón queda firme)
        edge = 90 * u
        win = (smoothstep(0, edge, xx) * smoothstep(0, edge, W - xx) * smoothstep(0, edge, yy)
               * smoothstep(0, edge, H - yy)).astype(np.float32)
        self.pk = dict(box=(x0, y0, x1, y1), xx=xx, yy=yy, DX=DX * win, DY=DY * win, S=S * win,
                       TM=np.clip(TM, 0, 1) * win)

    def _pucker(self, img, Tm, A):
        """Aplica el frunce con intensidad A (0..1)."""
        if A <= 0.001:
            return
        pk = self.pk
        x0, y0, x1, y1 = pk["box"]
        mx = (pk["xx"] + A * pk["DX"] - x0).astype(np.float32)
        my = (pk["yy"] + A * pk["DY"] - y0).astype(np.float32)
        img[y0:y1, x0:x1] = cv2.remap(np.ascontiguousarray(img[y0:y1, x0:x1]), mx, my, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        Tm[y0:y1, x0:x1] = cv2.remap(np.ascontiguousarray(Tm[y0:y1, x0:x1]), mx, my, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        img[y0:y1, x0:x1] *= np.clip(1 + A * pk["S"], 0.55, 1.45)[..., None]
        Tm[y0:y1, x0:x1] *= (1 - 0.55 * A * pk["TM"])[..., None]

    # ------------------------------------------------------------------------------------------------
    def _wool_prepare(self, rng):
        """La lana roja: nace donde empieza el hilo (la primera estaca), lo recorre por delante y sube al
        cordel; de ella salen ramales a cada casa."""
        u, W, H = self.u, self.W, self.H
        hs = self.info["houses"]
        doors = [np.asarray(h["door"]) for h in hs]
        self.doors = doors
        H1, H2, H3, H4 = self.H1, self.H2, self.H3, self.H4
        start = self.first_stitch + np.array([-8 * u, 16 * u])
        off = lambda a, b, k: (a + b) / 2 + _unit(np.array([-(b - a)[1], (b - a)[0]])) * k * u
        way = [start, H2 + (0, 18 * u), off(H2, H3, 20), H3 + (0, 44 * u), off(H3, H4, 14), H4 + (10 * u, -18 * u)]
        if not self.portrait:
            # la lana esquiva el techo de la casa amarilla camino a la torre
            c = hs[2]["center"]
            way += [np.array([(H4[0] + c[0]) / 2, H4[1] - 70 * u]), np.array([c[0], hs[2]["top"] - 95 * u])]
        else:
            way += [(H4 + H1) / 2 + np.array([-40 * u, 0])]
        way += [H1 + (-26 * u, 8 * u)]
        rise = [H1 + (-60 * u, -110 * u), np.array([H1[0] - 80 * u, 0.12 * H]), np.array([H1[0] - 96 * u, -30 * u])]
        path = _wobbly(way + rise, rng, 7 * u, 3.0 * u)
        self.wool = Yarn(path, 8.5 * u, WOOL, rng, couch_every=19 * u, couch_color="#8e160f", fuzz=1.0)
        # largo hasta la torre (fin del primer tramo)
        d_ = np.linalg.norm(self.wool.pts - (H1 + (-26 * u, 8 * u)), axis=1)
        self.L_h1 = float(self.wool.arc[int(np.argmin(d_))])
        # ramales: del punto del tramo principal más cercano a cada puerta
        order = [0, 2, 1, 3]                            # A, C, B, D (van acelerando)
        kmax = int(np.searchsorted(self.wool.arc, self.L_h1))
        self.spurs = []
        for i in order:
            d0 = doors[i]
            k_ = int(np.argmin(np.linalg.norm(self.wool.pts[:kmax] - d0, axis=1)))
            p0 = self.wool.pts[k_]
            mid = (p0 + d0) / 2 + _unit(np.array([-(d0 - p0)[1], (d0 - p0)[0]])) * 18 * u
            sp = _wobbly([p0, mid, d0 + (0, -4 * u)], rng, 5 * u, 3.0 * u)
            self.spurs.append(dict(house=i, yarn=Yarn(sp, 7.0 * u, WOOL, rng, couch_every=17 * u,
                                                      couch_color="#8e160f", fuzz=1.0)))
        a, b = T["spurs"]
        durs = np.array([1.0, 0.8, 0.64, 0.5])
        edges = a + np.concatenate([[0], np.cumsum(durs)]) / durs.sum() * (b - a)
        for sp, t0, t1 in zip(self.spurs, edges[:-1], edges[1:]):
            sp["t"] = (t0, t1)

    def wool_state(self, tq):
        """(largo del tramo principal, [largo de cada ramal], tramo activo (yarn, largo) o None)."""
        a, b = T["wool"]
        if tq < a:
            return 0.0, [0.0] * len(self.spurs), None
        f = float(np.clip((tq - a) / (b - a), 0, 1))
        Lm = self.L_h1 * f
        active = (self.wool, Lm) if f < 1 else None
        Ls = []
        for sp in self.spurs:
            t0, t1 = sp["t"]
            fs = float(np.clip((tq - t0) / (t1 - t0), 0, 1))
            Ls.append(sp["yarn"].length * fs)
            if t0 <= tq < t1:
                active = (sp["yarn"], sp["yarn"].length * fs)
        r0, r1 = T["rise"]
        if tq >= r0:
            fr = float(np.clip((tq - r0) / (r1 - r0), 0, 1))
            Lm = self.L_h1 + (self.wool.length - self.L_h1) * fr
            active = (self.wool, Lm) if fr < 1 else None
        return Lm, Ls, active

    def house_lit(self, tq):
        return [tq >= sp["t"][1] - 1e-6 for sp in sorted(self.spurs, key=lambda s: s["house"])]

    # ------------------------------------------------------------------------------------------------
    def _dolls_prepare(self):
        """Cada muñeca en sus poses (quieta, brazos arriba) y la vecina que tira del hilván."""
        self.dolls = []
        houses = [h["center"] for h in self.info["houses"]]
        for i, sp in enumerate(self.info["dolls"]):
            mk = lambda arms, lean=0.0, hands=None: doll_layer(doll(
                sp["fx"], sp["base"], sp["hgt"], np.random.default_rng(700 + i), 0, dress=sp["dress"],
                dress_kind=sp["dress_kind"], dress2=sp["dress2"], skin=sp["skin"], hair=sp["hair"], arms=arms,
                u=sp["u"], hands=hands), lean=lean)
            near = int(np.argmin([np.hypot(sp["fx"] - c[0], sp["base"] - c[1]) for c in houses]))
            d = dict(spec=sp, house=near, rest=mk(sp["arms"]), up=mk("up"))
            self.dolls.append(d)
        # la vecina que saca el hilván: la más cercana a la primera estaca
        k = int(np.argmin([np.hypot(d["spec"]["fx"] - self.first_stitch[0], d["spec"]["base"] - self.first_stitch[1])
                           for d in self.dolls]))
        self.puller = k
        sp = self.dolls[k]["spec"]
        h = sp["hgt"]
        side = 1.0 if self.first_stitch[0] >= sp["fx"] else -1.0
        neck = sp["base"] - h * 0.68
        hands = (np.array([sp["fx"] + side * h * 0.30, neck + h * 0.06]), np.array([sp["fx"] + side * h * 0.34, neck + h * 0.14]))
        self.pull_hand = (hands[0] + hands[1]) / 2
        self.dolls[k]["pull"] = [doll_layer(doll(sp["fx"], sp["base"], h, np.random.default_rng(700 + k), 0,
                                                 dress=sp["dress"], dress_kind=sp["dress_kind"], dress2=sp["dress2"],
                                                 skin=sp["skin"], hair=sp["hair"], u=sp["u"], hands=hands),
                                            lean=-side * ln) for ln in (0.05, 0.10, 0.15)]
        self.ball_pt = np.array([sp["fx"] - side * h * 0.22, sp["base"] - 6 * self.u])

    def _draw_dolls(self, img, tq):
        lit_t = {sp["house"]: sp["t"][1] for sp in self.spurs}
        p0, p1 = T["pull"]
        for i, d in enumerate(self.dolls):
            lay = d["rest"]
            tl = lit_t.get(d["house"])
            if tl is not None and tl <= tq < tl + 0.9:
                lay = d["up"]
            if i == self.puller and p0 <= tq < p1 + 0.25:
                k = int(np.clip((tq - p0) * UFPS, 0, 2))
                lay = d["pull"][k] if tq < p1 else d["pull"][0]
            composite(img, lay, shadow=0)

    # ------------------------------------------------------------------------------------------------
    def _smoke(self, canvas, tq):
        u = self.u
        k = int(round(tq * UFPS))
        r = np.random.default_rng(1000 + k)
        c = self.chimney
        pts = [c]
        for i in range(1, 7):
            pts.append(c + (np.sin(i * 1.3 + k * 0.9) * 12 * u * i ** 0.6 + r.normal(0, 3 * u), -i * 16 * u))
        Yarn(catmull_rom(np.array(pts), 8), 4.2 * u, "#f3efe7", r, fuzz=1.6).draw(canvas, 1e9, shadow=0.25)

    def _thread(self, a, b, w, color, rng, sag=0.12, kind="floss"):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        dist = float(np.hypot(*(b - a)))
        if dist < 2:
            return []
        nrm = _unit(b - a)
        perp = np.array([-nrm[1], nrm[0]])
        mid = (a + b) / 2 + np.array([0, sag * dist]) + perp * rng.normal(0, 0.03) * dist
        return Yarn(catmull_rom(np.array([a, mid, b]), 14), w, color, rng, fuzz=0, kind=kind).chunks

    def dangle_point(self, tq):
        """Antes de coser, la aguja cuelga de su hilo desde fuera del cuadro y se mece."""
        if tq >= T["stakes"][0]:
            return None
        u = self.u
        k = int(round(tq * UFPS))
        sway = np.sin(k / UFPS * 2 * np.pi / 1.3) * 7 * u
        dx = 140 * u if not self.portrait else 330 * u
        return self.first_stitch + np.array([dx + sway, -250 * u])

    def _needle_layers(self, tq):
        """La aguja y su hilo en este instante: lista de sprites (el hilo primero)."""
        u = self.u
        k = int(round(tq * UFPS))
        rj = np.random.default_rng(900 + k)
        out = []
        Ln, wn = 236 * u, 7.6 * u
        eye_of = lambda eye, tip: eye + (tip - eye) * 0.07
        dp = self.dangle_point(tq)
        if dp is not None:
            eye = dp
            tip = dp + np.array([np.sin(k * 0.7) * 6 * u, Ln])
            out.append(needle_sprite(eye, tip, wn, lift=3.0))
            return out
        h0, h1 = T["stakes"][0], T["pylon"][1]
        if h0 <= tq < h1:
            last = nxt = None
            for it in self.hilvan:
                if it[0] <= tq + 1e-6:
                    last = it
                else:
                    nxt = it
                    break
            th = np.deg2rad(52 + rj.normal(0, 4))
            v = np.array([np.cos(th), np.sin(th)])
            inside = (k % 2 == 0) and nxt is not None and (nxt[0] - tq) < 0.3
            if inside:
                entry = nxt[2]
                Lp = Ln * 0.86
                tip, eye = entry - v * 0.2 * Lp, entry + v * 0.8 * Lp
                nd = needle_sprite(eye, tip, wn, hide_from=0.8, lift=0.6)
            else:
                ref = last[3] if last is not None else nxt[2]
                dd = _unit((nxt[2] - ref) if nxt is not None else (1.0, -0.3))
                tip = ref + dd * 18 * u + np.array([0, -5 * u])
                eye = tip + v * Ln
                nd = needle_sprite(eye, tip, wn, lift=1.4)
            e_pt = eye_of(eye, tip)
            jump = last is None or (nxt is not None and nxt[4] != last[4])
            anchor = (nxt[2] + v * 10 * u) if (jump and inside) else (last[3] if last is not None else None)
            if anchor is None:
                out += self._thread(e_pt, e_pt + np.array([-12 * u, 70 * u]), 2.5 * u, BLK, rj, sag=0.0)
            else:
                out += self._thread(anchor, e_pt, 2.5 * u, BLK, rj, sag=0.10)
            out.append(nd)
            return out
        a, b = T["taut"]
        if a <= tq < b - 0.2:
            # tira del hilo desde la torre: queda tenso y la tela se frunce
            f = float(smoothstep(a, b - 0.3, tq))
            pull = _unit((1.0, -0.42)) if not self.portrait else _unit((1.0, -0.3))
            eye = self.last_stitch + pull * (60 * u + 720 * u * f ** 1.4)
            tip = eye + _unit(pull + np.array([0.05, -0.3])) * Ln
            out += self._thread(self.last_stitch, eye, 2.5 * u, BLK, rj, sag=0.0)
            out.append(needle_sprite(eye, tip, wn, lift=1.6))
            return out
        Lm, Ls, active = self.wool_state(tq)
        if active is not None:
            yarn, L = active
            p, d = yarn.head(L)
            inside = (k % 2 == 0)
            th = np.deg2rad(58 + rj.normal(0, 4))
            v = np.array([np.cos(th), np.sin(th)])
            Lw, ww = 200 * u, 8.0 * u
            if inside:
                Lp = Lw * 0.86
                tip, eye = p - v * 0.2 * Lp, p + v * 0.8 * Lp
                nd = needle_sprite(eye, tip, ww, hide_from=0.8, lift=0.6)
            else:
                tip = p + d * 22 * u + np.array([0, -5 * u])
                eye = tip + v * Lw
                nd = needle_sprite(eye, tip, ww, lift=1.3)
            out += self._thread(p, eye_of(eye, tip), 8.0 * u, WOOL, rj, sag=0.16, kind="wool")
            out.append(nd)
        return out

    # ------------------------------------------------------------------------------------------------
    def light(self, tq):
        """(luz de frente, luz de atrás, lámpara): lámpara = dict(c, R) o 'full'."""
        d0, d1 = T["dim"]
        s0, s1 = T["scan"]
        f0, f1 = T["full"]
        b0, b1 = T["back"]
        big = max(self.W, self.H)
        if tq < d0:
            return 1.0, 0.0, None
        if tq < d1:                                    # se apaga la sala en tres imágenes
            i = min(2, int(round((tq - d0) * UFPS)))
            return (0.62, 0.3, 0.1)[i], (0.12, 0.35, 0.7)[i], dict(c=self.back_path[0], R=0.1 * big)
        if tq < s1:                                    # la lámpara se detiene en cada huella
            i = int(round((tq - s0) * UFPS))
            sched = [0, 0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3, 3, 3]
            x = sched[min(i, len(sched) - 1)]
            j = int(np.floor(x))
            fr = x - j
            stops = np.array(self.back_path, np.float64)
            c = stops[j] if fr == 0 else stops[j] + (stops[min(j + 1, 3)] - stops[j]) * fr
            return 0.025, 1.0, dict(c=c, R=0.085 * big)
        if tq < f1:                                    # toda la tela es linterna
            i = int(round((tq - f0) * UFPS))
            if i < 2:
                return 0.03, 1.0, dict(c=self.back_path[-1] * 0.4 + np.array([self.W, self.H]) * 0.3,
                                       R=(0.26, 0.5)[i] * big)
            return 0.035, 1.0, "full"
        if tq < b1:                                    # vuelve la luz del día, de a poco
            i = min(5, int(round((tq - b0) * UFPS)))
            return (0.12, 0.25, 0.42, 0.6, 0.8, 0.94)[i], (0.8, 0.6, 0.4, 0.22, 0.1, 0.03)[i], "full"
        return 1.0, 0.0, None

    def pull_state(self, tq):
        """Cuántas puntadas del hilván ya salieron (en el orden en que se cosieron; la torre al último)."""
        p0, p1 = T["pull"]
        n = len(self.hilvan)
        if tq < p0:
            return 0
        n_sig = sum(1 for it in self.hilvan if it[4] != 3)
        f = (tq - p0) / (p1 - p0)
        # las señales salen rápido (40 % del tiempo); la torre, puntada a puntada (≥ 8 imágenes)
        if f < 0.4:
            return int(round(n_sig * f / 0.4))
        return int(round(n_sig + (n - n_sig) * np.clip((f - 0.4) / 0.55, 0, 1)))

    def pucker_amount(self, tq):
        a, b = T["taut"]
        if tq < a:
            return 0.0
        if tq < b:
            return float(smoothstep(a + 0.08, b - 0.3, tq))
        gone = self.pull_state(tq)
        # el frunce se afloja cuando salen las puntadas del tramo fruncido (aviso y toma)
        idx = [i for i, it in enumerate(self.hilvan) if it[4] in (1, 2)]
        if not idx or gone <= idx[0]:
            return 1.0
        f = np.clip((gone - idx[0]) / max(1, idx[-1] - idx[0]), 0, 1)
        return float(1.0 - 0.72 * f)

    # ------------------------------------------------------------------------------------------------
    def arpillera(self, t):
        """La arpillera (en su propio encuadre) ya iluminada, y la luz de la sala."""
        tq = q(t)
        img = self.base.copy()
        Tm = self.Tb.copy()
        u = self.u
        self._smoke(img, tq)
        gone = self.pull_state(tq)
        alive = [it for i, it in enumerate(self.hilvan) if it[0] <= tq + 1e-6 and i >= gone]
        live_h = set(it[4] for it in self.hilvan[gone:]) if gone < len(self.hilvan) else set()
        if gone > 0:
            for hs_ in self.hole_sprites[: max(1, int(len(self.hole_sprites) * gone / len(self.hilvan)))]:
                composite(img, hs_, shadow=0)
        # retazos del megaproyecto (se posan en stop-motion; se van cuando sale su hilván)
        for (tt, pc, h) in self.sig_pieces:
            if tq + 1e-6 >= tt and (gone == 0 or h in live_h):
                if tq < tt + 1 / UFPS - 1e-6:
                    pc.draw(img, tq)
                else:
                    pc.bake(img, Tm)
        for (tt, sp, h) in self.extra:
            if tq + 1e-6 >= tt and (gone == 0 or h in live_h):
                composite(img, sp, shadow=0.5)
                composite_T(Tm, sp)
        for it in alive:
            composite(img, it[1], shadow=0.55)
            composite_T(Tm, it[1])
        # el cabo del hilo tras la torre (cortado)
        if T["taut"][1] - 0.2 <= tq and gone < len(self.hilvan):
            rj = np.random.default_rng(5)
            for sp in self._thread(self.last_stitch, self.last_stitch + np.array([30, 46]) * u, 2.5 * u, BLK, rj,
                                   sag=0.05):
                composite(img, sp, 0.5)
        # frunce
        self._pucker(img, Tm, self.pucker_amount(tq))
        # lana roja: el tramo principal y los ramales a las casas
        Lm, Ls, _ = self.wool_state(tq)
        if Lm > 0:
            self.wool.draw(img, Lm)
        for sp, L in zip(self.spurs, Ls):
            if L > 0:
                sp["yarn"].draw(img, L)
        # ventanas encendidas
        for i, lit in enumerate(self.house_lit(tq)):
            if lit:
                for sp in self.info["houses"][i]["lit"]:
                    composite(img, sp, shadow=0.3)
        # la gente
        self._draw_dolls(img, tq)
        # la vecina saca el hilván: el hilo suelto va de sus manos a la puntada que sigue; ovillo a sus pies
        p0, p1 = T["pull"]
        if p0 <= tq < p1 + 0.5:
            rj = np.random.default_rng(int(tq * 100))
            if gone < len(self.hilvan):
                nxt = self.hilvan[gone][2]
                for sp in self._thread(self.pull_hand, nxt, 2.4 * u, BLK, rj, sag=0.08):
                    composite(img, sp, 0.5)
            r_ball = (5 + 12 * min(1.0, gone / len(self.hilvan))) * u
            ang = np.linspace(0, 2 * np.pi * 5, 90)
            rr = r_ball * (0.55 + 0.45 * np.abs(np.sin(ang * 0.37)))
            ball = np.stack([self.ball_pt[0] + rr * np.cos(ang + np.sin(ang * 0.21)),
                             self.ball_pt[1] - r_ball * 0.7 + rr * 0.7 * np.sin(ang)], 1)
            for sp in Yarn(ball, 2.4 * u, BLK, np.random.default_rng(4), fuzz=0, kind="floss").chunks:
                composite(img, sp, 0.45)
            for sp in self._thread(self.pull_hand, self.ball_pt + (0, -r_ball), 2.4 * u, BLK, rj, sag=0.25):
                composite(img, sp, 0.4)
        # aguja
        for sp in self._needle_layers(tq):
            composite(img, sp, shadow=0.45)
            composite_T(Tm, sp)
        # luz
        F, B, lamp = self.light(tq)
        out = img * F
        if B > 0:
            Te = Tm * (1 - 0.96 * self.back_mask)[..., None]
            Te = np.maximum(Te, (self.pinholes * 0.75)[..., None] * (1 - 0.5 * self.back_mask)[..., None])
            if isinstance(lamp, str):
                field, gain = np.float32(1.0), 11.0
            else:
                c, R = lamp["c"], lamp["R"]
                field = (0.006 + np.exp(-((self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2) / (2 * R * R)))[..., None]
                gain = 22.0
            x = Te * self.lamp * gain * B * field
            lum = x[..., 0] * 0.3 + x[..., 1] * 0.55 + x[..., 2] * 0.15
            lum2 = lum / (1 + lum)                                    # Reinhard: conserva el vitral
            glow = x * (lum2 / np.maximum(lum, 1e-5))[..., None]
            bloom = cv2.GaussianBlur(glow, (0, 0), 7 * u) * 0.2
            out = out + glow + bloom
        return np.clip(out, 0, 1), (F, B, lamp)

    # ------------------------------------------------------------------------------------------------
    def frame(self, t):
        tq = q(t)
        img, (F, B, lamp) = self.arpillera(t)
        k = int(round(tq * UFPS))
        return self.world.render_at(img, tq, T, F=F, B=B, lamp=lamp, flick=float(self.flick[k % len(self.flick)]),
                                    wool_tied=tq >= T["rise"][1] - 1e-6, dangle=self.dangle_point(tq), k=k)

    def frame_srgb8(self, t):
        return (linear_to_srgb(self.frame(t)) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------
# encuadres
# ------------------------------------------------------------------------------------------------------
def layout_for(W, H, sc):
    """Dónde cuelga cada arpillera en el encuadre final (mundo = último cuadro). Medidas pensadas para
    1920×1080 (o 1080×1920) y escaladas al tamaño pedido."""
    from .territory import build_desert, build_towers, exit_points
    ex = exit_points(sc.wool.pts, sc.u)
    land = W >= H
    f = W / (1920 if land else 1080)
    F = lambda v: v * f
    if land:
        nb = [(int(round(1280 * f)), int(round(960 * f))), (int(round(1200 * f)), int(round(960 * f)))]
    else:
        nb = [(int(round(900 * f)), int(round(1200 * f))), (int(round(880 * f)), int(round(1180 * f)))]
    des, des_ex = build_desert(*nb[0])
    tow, tow_ex = build_towers(*nb[1])
    nb_u = [min(s) / 1080 for s in nb]
    if land:
        main_x = (F(440), F(1480))
        z0 = 1.66
        title = dict(rect=tuple(F(v) for v in (540, 832, 1380, 1024)), font="timesi", lines=[
            (F(905), F(40), [("Sistema", "floss"), ("de", "floss"), ("Alerta", "wool"), ("Temprana", "floss")]),
            (F(966), F(40), [("Comunitario", "floss")]),
            (F(1006), F(15), [("Red Comunitaria de Alerta Energética", "small")])])
        line = (F(960), F(150), 0.000028 / f)
        nbx = [(F(-190), F(380)), (F(1540), F(2110))]
    else:
        main_x = (F(270), F(810))
        z0 = 1.9
        title = dict(rect=tuple(F(v) for v in (110, 1455, 970, 1660)), font="timesi", lines=[
            (F(1528), F(38), [("Sistema", "floss"), ("de", "floss"), ("Alerta", "wool")]),
            (F(1590), F(38), [("Temprana", "floss"), ("Comunitario", "floss")]),
            (F(1636), F(15), [("Red Comunitaria de Alerta Energética", "small")])])
        line = (F(540), F(378), 0.000026 / f)
        nbx = [(F(-240), F(222)), (F(858), F(1310))]
    L = dict(line=line, main=dict(size=(W, H), x=main_x),
             neighbors=[dict(img=des, x=nbx[0], exit=des_ex, wool_w=8.0 * nb_u[0]),
                        dict(img=tow, x=nbx[1], exit=tow_ex, wool_w=8.0 * nb_u[1])],
             title=title, wool_exit=ex, wool_w=8.5 * sc.u, z0=z0)
    # centro inicial: la tela arriba, con un margen donde se ven el cordel y los perritos
    xm, ym, k = L["line"]
    y_top = ym + 3 * f
    cy = y_top + (H / 2 - F(66)) / z0
    L["c0"] = ((main_x[0] + main_x[1]) / 2, cy)
    return L
