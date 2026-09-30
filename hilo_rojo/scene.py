"""«Hilván»: línea de tiempo (16,95 s, 24 fps animados en dos = 12 imágenes únicas por segundo).

Conocer   0,0–1,2    póster: la arpillera cuelga de un cordel de cáñamo; una aguja grande, enhebrada con
                     hilo negro, cuelga desde fuera del cuadro y se mece; baja a coser (algo llega)
Vigilar   1,2–4,6    la cámara se acerca: la aguja hilvana las señales tempranas —estacas de topógrafo,
                     un AVISO prendido con alfiler de gancho, una cañería con su llave de paso que sale
                     del río—; la cámara se aleja y la aguja hilvana en el cerro la torre de alta tensión,
                     solo como contorno (lo que vendría); tira del hilo y la tela se frunce
Alertar   4,65–7,9   baja la luz de la sala; una luz cálida pasa por detrás, de izquierda a derecha, y
                     aparece por el revés un solo hilo que une todas las señales; toda la tela es
                     linterna (se lee el saco: HARINA); se abre la cortina y vuelve el día
Responder 7,95–12,75 una vecina tiende la lana roja: marca el hilo por delante y la lleva de casa en
                     casa (se enciende cada ventana y su gente apunta al cerro); pausa; la cámara se
                     acerca: la vecina tira del hilván y la torre se descose puntada a puntada hacia su
                     ovillo (quedan los agujeros); la lana sube y se anuda al cordel
Red       12,25–16,95 la cámara se aleja por pasos: el cordel sostiene otras arpilleras; desde el nudo, el
                     rojo corre por el cordel, baja a cada vecina y enciende su ventana; ellas se mecen.
                     La tira del título, en la pared
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
DURATION = 16.95

T = dict(hook=(0.92, 1.17), stakes=(1.17, 1.8), notice=(1.85, 2.4), intake=(2.45, 2.98), pylon=(3.3, 3.95),
         taut=(4.0, 4.6), dim=(4.65, 4.9), sweep=(4.9, 6.25), full=(6.25, 7.45), curtain=(7.45, 7.87),
         wool=(7.95, 9.85), pull=(10.55, 11.95), rise=(12.25, 12.75), dolly=(12.25, 13.75), front=(13.85, 15.15))
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


def _crossings(pts):
    """Cuántas veces se cruza consigo misma una polilínea (tramos no vecinos)."""
    pts = [np.asarray(p, np.float64) for p in pts]

    def cross(a, b, c, d):
        o = lambda p, q, r: np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
        return o(a, b, c) != o(a, b, d) and o(c, d, a) != o(c, d, b)

    n = 0
    for i in range(len(pts) - 1):
        for j in range(i + 2, len(pts) - 1):
            if cross(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                n += 1
    return n


class Scene:
    def __init__(self, W=1920, H=1080, seed=11, ss=1.5):
        # la tela se cose (se calcula) a más resolución que el cuadro: así la cámara puede acercarse
        self.OW, self.OH = W, H
        W, H = int(round(W * ss)), int(round(H * ss))
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
        self.notice_c = sig["notice_c"]
        self.pylon_top = sig["pylon_top"]
        self.first_stitch = self.hilvan[0][2]
        self.tower_idx = [i for i, it in enumerate(self.hilvan) if it[4] == 3]
        self.last_stitch = self.hilvan[self.tower_idx[-1]][3]
        # ---------------------------------------------------------------- el hilo por el revés
        self.back_path = [self.H2, self.H3, self.H4, self.H1]
        self.back_mask = self._back_mask(rng)
        self.pinholes = self._pinholes()
        # ---------------------------------------------------------------- el frunce
        self._pucker_prepare([(self.H2, self.H3), (self.H3, self.H4)], [self.H2, self.H3, self.H4])
        # ---------------------------------------------------------------- vecinas y vecinos, y la lana roja
        self._dolls_prepare()
        self._wool_prepare(rng)
        # ---------------------------------------------------------------- humo y agujeros
        self.chimney = info["chimney"]
        self.tower_holes = [self._hole(p, np.random.default_rng(int(p[0] * 13 + p[1])), faint=True)
                            for p in self._tower_hole_pts()]
        # luz y grano
        self.flick = np.random.default_rng(seed + 99).normal(0, 0.009, 400)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.yy, self.xx = yy, xx
        self.lamp = np.array([1.0, 0.88, 0.72], np.float32)        # luz de tarde, cálida
        # ---------------------------------------------------------------- el mundo
        self.layout = layout_for(self.OW, self.OH, self)
        self.world = World(self.OW, self.OH, self.layout)

    # ------------------------------------------------------------------------------------------------
    def _tower_hole_pts(self):
        pts = []
        for i in self.tower_idx:
            for p in self.hilvan[i][2:4]:
                if all(np.hypot(*(p - q_)) > 3.2 * self.u for q_ in pts):
                    pts.append(p)
        return pts

    def _hole(self, p, rng, faint=False):
        """Agujero que deja una puntada al sacarla. faint: apenas una marca (memoria, no mancha)."""
        u = self.u
        R = 2.2 * u
        x0, y0 = int(p[0] - 8 * u), int(p[1] - 8 * u)
        n = int(16 * u) + 2
        yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n].astype(np.float32) + 0.5
        dx, dy = xx - p[0], yy - p[1]
        d = np.hypot(dx, dy)
        wob = 1 + 0.18 * np.sin(np.arctan2(dy, dx) * 3 + rng.uniform(0, 6))
        core = np.clip(R * 0.6 * wob - d + 0.5, 0, 1)
        rim = np.clip(1 - np.abs(d - R * 1.15 * wob) / (0.8 * u), 0, 1)
        lit = np.clip((dx * 0.64 + dy * 0.56) / (d + 1e-3), 0, 1)
        k = 0.34 if faint else 1.0
        a = np.clip((core * 0.85 + rim * (0.22 + 0.32 * lit)) * k, 0, 1).astype(np.float32)
        rgb = (lin("#1e150e")[None, None, :] * core[..., None] * 0.85
               + lin("#efe4cf")[None, None, :] * (rim * (0.22 + 0.32 * lit))[..., None]) * k
        return Sprite(x0, y0, rgb.astype(np.float32), a, None)

    def _back_mask(self, rng):
        """El hilo del revés: una sola hebra gruesa (≈7 px) con holgura entre huella y huella, grosor que
        cambia al torcerse y un halo corto (más borroso donde se separa de la tela)."""
        W, H, u = self.W, self.H, self.u
        pts = []
        for a, b in zip(self.back_path[:-1], self.back_path[1:]):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            L = float(np.hypot(*(b - a)))
            nrm = np.array([-(b - a)[1], (b - a)[0]]) / (L + 1e-9)
            mid = (a + b) / 2 + nrm * L * 0.02 * rng.choice([-1, 1]) + np.array([0, L * 0.012])
            seg = catmull_rom(np.array([a, mid, b]), 30)
            pts.append(seg if not pts else seg[1:])
        pts = resample(np.vstack(pts), 1.5)
        n = len(pts)
        rad = 3.6 * u * (0.8 + 0.4 * smooth_noise((1, n), 25, rng)[0])
        far = smooth_noise((1, n), 60, rng)[0]
        core = np.zeros((H, W), np.float32)
        loose = np.zeros((H, W), np.float32)
        for (x, y), r, f in zip(pts, rad, far):
            cv2.circle(core, (int(x * 4), int(y * 4)), int(r * 4), float(1 - 0.5 * f), -1, cv2.LINE_AA, shift=2)
            cv2.circle(loose, (int(x * 4), int(y * 4)), int(r * 4), float(f), -1, cv2.LINE_AA, shift=2)
        core = cv2.GaussianBlur(np.clip(core, 0, 1), (0, 0), 0.6 * u)
        halo = cv2.GaussianBlur(np.clip(loose, 0, 1), (0, 0), 3.2 * u)
        return np.clip(core * 1.25 + halo * 0.6, 0, 1)

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
    def _pucker_prepare(self, segs, pins):
        """Frunce: pliegues perpendiculares al hilo que arrastran la tela (y su estampado), pliegues que
        irradian de cada huella, un tirón que acerca y ladea lo de los extremos y el ruedo que sube."""
        u, W, H = self.u, self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        DX = np.zeros_like(xx)
        DY = np.zeros_like(xx)
        S = np.zeros_like(xx)
        TM = np.zeros_like(xx)
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
            wob2 = (smooth_noise(s_.shape, 90 * u, rngp) - 0.5)
            g = (np.exp(-(e / (82 * u * (1 + 0.3 * wob))) ** 2) * smoothstep(-26 * u, 34 * u, s_)
                 * smoothstep(L + 26 * u, L - 34 * u, s_))
            beyond = np.maximum(np.maximum(-s_, s_ - L), 0)
            side = np.clip((s_ - L / 2) / (L / 2), -1, 1)
            tug = np.exp(-(e / (240 * u)) ** 2) * np.exp(-beyond / (380 * u)) * side
            ph = 2 * np.pi * s_ / (lam * (1 + 0.45 * wob2)) + wob * 3.2 + e / (60 * u) * 0.9
            gather = 0.2 * L
            # el estampado se arrastra de verdad: compresión en los pliegues y ondulación lateral
            ds = (g * 0.2 * (np.clip(s_, 0, L) - L / 2) + gather / 2 * tug * (1 - g)
                  + g * 1.05 * lam / (2 * np.pi) * np.sin(ph))
            de = g * (0.06 * e + 3.2 * u * np.cos(ph) * (0.6 + 0.4 * wob2))
            # los extremos se ladean hacia el hilo (las casas se inclinan)
            de = de + 1.5 * u * np.abs(tug) * np.clip(e / (120 * u), -1, 1) * side
            DX += ds * t[0] + de * n[0]
            DY += ds * t[1] + de * n[1]
            hgt = np.abs(np.sin(ph / 2)) ** 0.5
            dh = np.gradient(hgt, axis=1) * t[0] + np.gradient(hgt, axis=0) * t[1]
            Ld = -0.64 * t[0] - 0.56 * t[1]
            S += g * (3.2 * lam / (2 * np.pi) * dh * Ld * 0.5 - 0.22 * (1 - hgt))
            TM += g * hgt
        # pliegues que irradian de cada huella (el hilo tira de un punto)
        for p in pins:
            p = np.asarray(p, np.float64)
            dx, dy = xx - p[0], yy - p[1]
            r = np.hypot(dx, dy) + 1e-3
            th = np.arctan2(dy, dx)
            R = 90 * u
            gr = np.exp(-r / R) * smoothstep(3 * u, 14 * u, r)
            ph = 7 * th + 1.3 * np.sin(r / (22 * u))
            rad = -0.16 * r * gr                                  # la tela se recoge hacia el punto
            tan = 2.2 * u * gr * np.sin(ph)                       # y se pliega en abanico
            DX += rad * dx / r - tan * dy / r
            DY += rad * dy / r + tan * dx / r
            hgt = np.abs(np.sin(ph / 2)) ** 0.6
            dth = np.gradient(hgt, axis=1) * (-dy / r) + np.gradient(hgt, axis=0) * (dx / r)
            Lt = -0.64 * (-dy / r) - 0.56 * (dx / r)
            S += gr * (2.4 * dth * Lt * r * 0.06 - 0.16 * (1 - hgt))
            TM += gr * hgt * 0.8
        # la orilla también cede un poco (el ruedo sube donde se recoge la tela)
        edge = 30 * u
        win = (0.35 + 0.65 * smoothstep(0, edge, xx) * smoothstep(0, edge, W - xx) * smoothstep(0, edge, yy)
               * smoothstep(0, edge, H - yy)).astype(np.float32)
        self.pk = dict(xx=xx, yy=yy, DX=DX * win, DY=DY * win, S=S * win, TM=np.clip(TM, 0, 1) * win)

    def _pucker(self, img, Tm, A, alpha=None):
        """Aplica el frunce con intensidad A (0..1); también a la silueta de la tela (alpha)."""
        if A <= 0.001:
            return img, Tm, alpha
        pk = self.pk
        mx = (pk["xx"] + A * pk["DX"]).astype(np.float32)
        my = (pk["yy"] + A * pk["DY"]).astype(np.float32)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        Tm = cv2.remap(Tm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if alpha is not None:
            alpha = cv2.remap(alpha, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img *= np.clip(1 + A * pk["S"], 0.55, 1.45)[..., None]
        Tm *= (1 - 0.55 * A * pk["TM"])[..., None]
        return img, Tm, alpha

    # ------------------------------------------------------------------------------------------------
    def _dolls_prepare(self):
        """Cada muñeca en sus poses. La vigía (la más cercana a las estacas) tiende la lana y descose."""
        self.dolls = []
        houses = [h["center"] for h in self.info["houses"]]
        mk_spec = lambda sp, **kw: doll(sp["fx"], sp["base"], sp["hgt"], np.random.default_rng(1), 0,
                                        dress=sp["dress"], dress_kind=sp["dress_kind"], dress2=sp["dress2"],
                                        skin=sp["skin"], hair=sp["hair"], u=sp["u"], kind=sp["kind"], **kw)
        for i, sp in enumerate(self.info["dolls"]):
            near = int(np.argmin([np.hypot(sp["fx"] - c[0], sp["base"] - c[1]) for c in houses]))
            d = dict(spec=sp, house=near, rest=doll_layer(mk_spec(sp, arms=sp["arms"])),
                     point=doll_layer(mk_spec(sp, arms="point", target=self.H1 - (0, 120 * self.u))))
            self.dolls.append(d)
        k = int(np.argmin([np.hypot(d["spec"]["fx"] - self.first_stitch[0], d["spec"]["base"] - self.first_stitch[1])
                           for d in self.dolls]))
        self.vigia = k
        sp = self.dolls[k]["spec"]
        h = sp["hgt"] * (0.74 if sp["kind"] == "child" else 1.0)
        neck = sp["base"] - h * 0.68
        # sostiene la lana: mano estirada hacia las estacas
        side = 1.0 if self.first_stitch[0] >= sp["fx"] else -1.0
        self.hold_hand = np.array([sp["fx"] + side * h * 0.32, neck + h * 0.12])
        other = np.array([sp["fx"] - side * h * 0.27, neck + h * 0.26])
        hands = (other, self.hold_hand) if side > 0 else (self.hold_hand, other)
        self.dolls[k]["hold"] = doll_layer(mk_spec(sp, hands=hands))
        # tira del hilván: las dos manos hacia la torre, el cuerpo inclinado hacia atrás
        tside = 1.0 if self.H1[0] >= sp["fx"] else -1.0
        hands = (np.array([sp["fx"] + tside * h * 0.28, neck + h * 0.04]),
                 np.array([sp["fx"] + tside * h * 0.33, neck + h * 0.12]))
        self.pull_hand = (hands[0] + hands[1]) / 2
        self.dolls[k]["pull"] = [doll_layer(mk_spec(sp, hands=hands), lean=-tside * ln) for ln in (0.06, 0.12, 0.16)]
        self.ball_pt = np.array([sp["fx"] - tside * h * 0.24, sp["base"] - 6 * self.u])

    def _draw_dolls(self, img, tq, Tm=None):
        lit_t = self.house_times
        p0, p1 = T["pull"]
        w0, w1 = T["wool"]
        for i, d in enumerate(self.dolls):
            lay = d["rest"]
            if i == self.vigia:
                if w0 <= tq < p0:
                    lay = d["hold"]
                elif p0 <= tq < p1:
                    kk = int(round((tq - p0) * UFPS))
                    lay = d["pull"][(0, 1, 2, 1)[kk % 4]]      # tira, recoge, tira...
            else:
                tl = lit_t.get(d["house"])
                if tl is not None and tq >= tl:
                    lay = d["point"]
            composite(img, lay, shadow=0)
            if Tm is not None:
                composite_T(Tm, lay)

    # ------------------------------------------------------------------------------------------------
    def _wool_prepare(self, rng):
        """La lana roja: sale de la mano de la vigía (su casa ya está encendida: ella vio), marca las señales
        (estacas, aviso, cañería), da la vuelta al campo de casa en casa y sube por el borde derecho hacia el
        cordel. Una sola vuelta, sin cruzarse."""
        u, W, H = self.u, self.W, self.H
        hs = self.info["houses"]
        doors = [np.asarray(h["door"]) for h in hs]
        stakes = self.info["stakes"]
        own, order = 2, [0, 1, 3]                             # amarilla (la de la vigía); roja, azul, blanca
        # las estacas se marcan en el orden que no hace cruzarse a la lana consigo misma
        tail = [self.notice_c + (30 * u, 64 * u), self.notice_c + (-20 * u, 66 * u), self.H4 + (8 * u, 30 * u)]
        opts = [[stakes[-1] + (6 * u, 16 * u), stakes[0] + (-4 * u, 16 * u)],
                [stakes[0] + (-4 * u, 16 * u), stakes[-1] + (6 * u, 16 * u)]]
        best = min(opts, key=lambda o: _crossings([self.hold_hand] + o + tail))
        way = [self.hold_hand] + best
        way += [self.notice_c + (30 * u, 64 * u), self.notice_c + (-20 * u, 66 * u)]
        way += [self.H4 + (8 * u, 30 * u)]
        keys = [("stakes", 2), ("notice", 4), ("intake", 5)]
        for i in order:
            way += [doors[i] + (0, 26 * u)]
            keys.append((i, len(way) - 1))
        ex = 0.962 * W
        last = doors[order[-1]]
        rise = [np.array([(last[0] + ex) / 2, last[1] - 30 * u]), np.array([ex, last[1] - 160 * u]),
                np.array([ex + 4 * u, 0.3 * H]), np.array([ex, 0.08 * H]), np.array([ex - 4 * u, -30 * u])]
        path = _wobbly(way + rise, rng, 6 * u, 3.0 * u)
        self.wool = Yarn(path, 8.5 * u, WOOL, rng, couch_every=19 * u, couch_color="#8e160f", fuzz=1.0)
        arc_at = lambda p: float(self.wool.arc[int(np.argmin(np.linalg.norm(self.wool.pts - p, axis=1)))])
        Lk = {name: arc_at(way[j]) for name, j in keys}
        self.L_end_houses = Lk[order[-1]]
        a, b = T["wool"]
        # horario: marcar las señales (lento) y luego el relevo, cada vez más rápido
        times = [a, a + 0.34, a + 0.62, a + 0.9, a + 1.26, a + 1.6, b]
        names = ["start", "stakes", "notice", "intake"] + order
        lens = [0.0] + [Lk[n] for n in names[1:]]
        r0, r1 = T["rise"]
        self.wool_sched = list(zip(times, lens)) + [(r0, self.L_end_houses), (r1, self.wool.length)]
        self.house_times = {i: t_ for i, t_ in zip(order, times[4:])}
        self.house_times[own] = a

    def wool_len(self, tq):
        sc = self.wool_sched
        if tq < sc[0][0]:
            return 0.0
        for (t0, l0), (t1, l1) in zip(sc[:-1], sc[1:]):
            if tq < t1:
                return l0 + (l1 - l0) * (tq - t0) / max(1e-6, t1 - t0)
        return sc[-1][1]

    def wool_active(self, tq):
        a, b = T["wool"]
        r0, r1 = T["rise"]
        return (a <= tq < b) or (r0 <= tq < r1)

    def house_lit(self, tq):
        return {i: tq >= t_ - 1e-6 for i, t_ in self.house_times.items()}

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

    def _dimple(self, img, p, r):
        """La aguja hunde la tela al entrar: un hoyuelo con su luz y su sombra, y arruguitas alrededor."""
        x0, y0 = int(p[0] - 3 * r), int(p[1] - 3 * r)
        x1, y1 = int(p[0] + 3 * r), int(p[1] + 3 * r)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
        dx, dy = (xx - p[0]) / r, (yy - p[1]) / r
        rr = np.sqrt(dx * dx + dy * dy) + 1e-3
        h = -np.exp(-rr * rr) * (1 + 0.25 * np.sin(np.arctan2(dy, dx) * 6) * np.clip(rr - 0.6, 0, 1))
        gy, gx = np.gradient(h)
        lam = -(gx * -0.64 + gy * -0.56) * 2.4
        img[y0:y1, x0:x1] *= np.clip(1 + lam - 0.18 * np.exp(-rr * rr * 2), 0.6, 1.3)[..., None]

    def _boil(self, img, c, k):
        """Vida de stop-motion: lo que está cerca de la mano se corre medio píxel de cuadro a cuadro."""
        u = self.u
        R = 240 * u
        x0, y0 = int(max(0, c[0] - R)), int(max(0, c[1] - R))
        x1, y1 = int(min(self.W, c[0] + R)), int(min(self.H, c[1] + R))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return img
        rg = np.random.default_rng(5000 + k)
        sh = (y1 - y0, x1 - x0)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        fall = np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * (R * 0.5) ** 2)).astype(np.float32)
        dx = (smooth_noise(sh, 50 * u, rg) - 0.5) * 1.6 * u * fall
        dy = (smooth_noise(sh, 50 * u, rg) - 0.5) * 1.6 * u * fall
        reg = np.ascontiguousarray(img[y0:y1, x0:x1])
        img[y0:y1, x0:x1] = cv2.remap(reg, (xx - x0 + dx).astype(np.float32), (yy - y0 + dy).astype(np.float32),
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return img

    def dangle(self, tq):
        """La aguja antes de coser: cuelga de su hilo desde fuera del cuadro y se mece; en tres imágenes
        baja en diagonal a la primera puntada mientras el hilo se desliza con ella. Devuelve
        dict(eye, tip, top) en coordenadas de la tela (top: de dónde viene el hilo) o None."""
        h0, h1 = T["hook"]
        if tq >= h1 - 1e-6:
            return None
        u = self.u
        k = int(round(tq * UFPS))
        dx = 150 * u if not self.portrait else 250 * u
        base_eye = self.first_stitch + np.array([dx, -330 * u])
        if tq < h0 - 1e-6:
            sway = np.sin(k / UFPS * 2 * np.pi / 1.4) * 24 * u
            eye = base_eye + np.array([sway, 0])
            ang = np.sin(k / UFPS * 2 * np.pi / 1.4 + 0.9) * 0.07
            tip = eye + np.array([np.sin(ang), np.cos(ang)]) * 300 * u
            top = np.array([eye[0] + 6 * u, -0.5 * self.H])
            return dict(eye=eye, tip=tip, top=top)
        i = int(round((tq - h0) * UFPS))
        f = (0.28, 0.63, 0.88)[min(i, 2)]
        end_tip = self.first_stitch + np.array([-4 * u, 6 * u])
        end_eye = end_tip + _unit(np.array([0.6, -0.8])) * 300 * u
        eye = base_eye + (end_eye - base_eye) * f
        tip = (base_eye + np.array([0, 300 * u])) + (end_tip - (base_eye + np.array([0, 300 * u]))) * f
        top = np.array([base_eye[0] + 6 * u, -0.5 * self.H]) * (1 - f) + self.first_stitch * f
        return dict(eye=eye, tip=tip, top=top)

    def _needle_layers(self, tq, img):
        """La aguja y su hilo en este instante: lista de sprites (el hilo primero). Hunde la tela al entrar."""
        u = self.u
        k = int(round(tq * UFPS))
        rj = np.random.default_rng(900 + k)
        out = []
        Ln, wn = 250 * u, 8.4 * u
        eye_of = lambda eye, tip: eye + (tip - eye) * 0.07
        dg = self.dangle(tq)
        if dg is not None:
            out.append(needle_sprite(dg["eye"], dg["tip"], 11 * u, lift=0.9))
            return out, dg["tip"]
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
                self._dimple(img, entry, 7 * u)
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
                out += self._thread(e_pt, e_pt + np.array([-12 * u, 70 * u]), 3.0 * u, BLK, rj, sag=0.0)
            else:
                out += self._thread(anchor, e_pt, 3.0 * u, BLK, rj, sag=0.10)
            out.append(nd)
            return out, tip
        a, b = T["taut"]
        if a <= tq < b - 0.2:
            f = float(smoothstep(a, b - 0.3, tq))
            pull = _unit((1.0, -0.42)) if not self.portrait else _unit((1.0, -0.3))
            eye = self.last_stitch + pull * (60 * u + 720 * u * f ** 1.4)
            tip = eye + _unit(pull + np.array([0.05, -0.3])) * Ln
            out += self._thread(self.last_stitch, eye, 3.0 * u, BLK, rj, sag=0.0)
            out.append(needle_sprite(eye, tip, wn, lift=1.6))
            return out, self.last_stitch
        if self.wool_active(tq):
            p, d = self.wool.head(self.wool_len(tq))
            inside = (k % 2 == 0)
            th = np.deg2rad(58 + rj.normal(0, 4))
            v = np.array([np.cos(th), np.sin(th)])
            Lw, ww = 220 * u, 8.8 * u
            if inside:
                Lp = Lw * 0.86
                tip, eye = p - v * 0.2 * Lp, p + v * 0.8 * Lp
                nd = needle_sprite(eye, tip, ww, hide_from=0.8, lift=0.6)
                self._dimple(img, p, 8 * u)
            else:
                tip = p + d * 22 * u + np.array([0, -5 * u])
                eye = tip + v * Lw
                nd = needle_sprite(eye, tip, ww, lift=1.3)
            out += self._thread(p, eye_of(eye, tip), 8.0 * u, WOOL, rj, sag=0.16, kind="wool")
            out.append(nd)
            return out, p
        return out, None

    # ------------------------------------------------------------------------------------------------
    def light(self, tq):
        """(luz de frente, luz de atrás, lámpara). Luz de frente: número o dict(curtain=0..1, fmin).
        Lámpara: dict(band=x) (franja vertical que cruza por detrás), dict(c, R) o 'full'."""
        d0, d1 = T["dim"]
        s0, s1 = T["sweep"]
        f0, f1 = T["full"]
        c0, c1 = T["curtain"]
        FMIN = 0.1
        if tq < d0:
            return 1.0, 0.0, None
        if tq < d1:                                    # baja la luz de la sala en tres imágenes
            i = min(2, int(round((tq - d0) * UFPS)))
            return (0.55, 0.3, 0.16)[i], 0.0, None
        if tq < s1:                                    # una luz cálida cruza por detrás, de izq. a der.
            f = (tq - s0) / (s1 - s0)
            return FMIN, 1.0, dict(band=-0.12 + 1.24 * f)
        if tq < f1:                                    # toda la tela es linterna
            i = int(round((tq - f0) * UFPS))
            return FMIN, (0.75, 1.0)[min(i, 1)], "full"
        if tq < c1:                                    # se abre la cortina: vuelve el día desde la izquierda
            f = (tq - c0) / (c1 - c0)
            return dict(curtain=float(f), fmin=FMIN), float(1 - smoothstep(0.2, 1.0, f)), "full"
        return 1.0, 0.0, None

    def pull_state(self, tq):
        """Cuántas puntadas de la torre ya salieron (de la última cosida a la primera)."""
        p0, p1 = T["pull"]
        n = len(self.tower_idx)
        if tq < p0:
            return 0
        f = np.clip((tq - p0 - 0.15) / (p1 - p0 - 0.3), 0, 1)
        return int(round(n * f))

    def pucker_amount(self, tq):
        a, b = T["taut"]
        if tq < a:
            return 0.0
        if tq < b:
            return float(smoothstep(a + 0.08, b - 0.3, tq))
        gone = self.pull_state(tq)
        return float(1.0 - 0.74 * gone / max(1, len(self.tower_idx)))

    # ------------------------------------------------------------------------------------------------
    def arpillera(self, t):
        """La arpillera en su propio encuadre: imagen con luz de frente plena, el resplandor de atrás
        (o None), la silueta (con el frunce) y la luz de la sala."""
        tq = q(t)
        img = self.base.copy()
        Tm = self.Tb.copy()
        u = self.u
        k = int(round(tq * UFPS))
        self._smoke(img, tq)
        gone = self.pull_state(tq)
        removed = set(self.tower_idx[::-1][:gone])
        if gone:
            for hs_ in self.tower_holes:
                composite(img, hs_, shadow=0)
        for (tt, pc, h) in self.sig_pieces:
            if tq + 1e-6 >= tt:
                if tq < tt + 1 / UFPS - 1e-6:
                    pc.draw(img, tq)
                else:
                    pc.bake(img, Tm)
        for (tt, sp, h) in self.extra:
            if tq + 1e-6 >= tt:
                composite(img, sp, shadow=0.5)
                composite_T(Tm, sp)
        for i, it in enumerate(self.hilvan):
            if it[0] <= tq + 1e-6 and i not in removed:
                composite(img, it[1], shadow=0.55)
                composite_T(Tm, it[1])
        # el cabo del hilo tras la torre (cortado), mientras la torre siga cosida
        if T["taut"][1] - 0.2 <= tq and gone == 0:
            rj = np.random.default_rng(5)
            for sp in self._thread(self.last_stitch, self.last_stitch + np.array([30, 46]) * u, 3.0 * u, BLK, rj,
                                   sag=0.05):
                composite(img, sp, 0.5)
        # frunce (también deforma la silueta: el ruedo sube)
        alpha = self.world.ea_main.copy() if hasattr(self, "world") else None
        img, Tm, alpha = self._pucker(img, Tm, self.pucker_amount(tq), alpha)
        # lana roja
        L = self.wool_len(tq)
        if L > 0:
            self.wool.draw(img, L)
        # ventanas encendidas
        for i, lit in self.house_lit(tq).items():
            if lit:
                for sp in self.info["houses"][i]["lit"]:
                    composite(img, sp, shadow=0.3)
        # la gente
        self._draw_dolls(img, tq, Tm)
        # la vigía tira del hilván: el hilo suelto va de sus manos a la puntada que sigue; ovillo a sus pies
        p0, p1 = T["pull"]
        if p0 <= tq < p1 + 1.2:
            rj = np.random.default_rng(int(tq * 100))
            n = len(self.tower_idx)
            if gone < n and tq < p1:
                nxt = self.hilvan[self.tower_idx[::-1][gone]][3]
                for sp in self._thread(self.pull_hand, nxt, 3.0 * u, BLK, rj, sag=0.06):
                    composite(img, sp, 0.5)
            r_ball = (6 + 13 * min(1.0, gone / max(1, n))) * u
            ang = np.linspace(0, 2 * np.pi * 6, 120)
            rr = r_ball * (0.5 + 0.5 * np.abs(np.sin(ang * 0.37)))
            ball = np.stack([self.ball_pt[0] + rr * np.cos(ang + np.sin(ang * 0.21)),
                             self.ball_pt[1] - r_ball * 0.7 + rr * 0.7 * np.sin(ang)], 1)
            for sp in Yarn(ball, 2.6 * u, BLK, np.random.default_rng(4), fuzz=0, kind="floss").chunks:
                composite(img, sp, 0.45)
            if tq < p1:
                for sp in self._thread(self.pull_hand, self.ball_pt + (0, -r_ball), 2.6 * u, BLK, rj, sag=0.25):
                    composite(img, sp, 0.4)
        # aguja
        nd, act = self._needle_layers(tq, img)
        for sp in nd:
            composite(img, sp, shadow=0.72)
            composite_T(Tm, sp)
        # vida de stop-motion cerca de la acción
        if act is not None:
            img = self._boil(img, act, k)
        elif p0 <= tq < p1:
            img = self._boil(img, self.pull_hand, k)
        # luz
        F, B, lamp = self.light(tq)
        glow = None
        if B > 0:
            Te = Tm * (1 - 0.96 * self.back_mask)[..., None]
            Te = np.maximum(Te, (self.pinholes * 0.75)[..., None] * (1 - 0.5 * self.back_mask)[..., None])
            if isinstance(lamp, str):
                field, gain = np.float32(1.0), 11.0
            elif "band" in lamp:
                xc = lamp["band"] * self.W
                wb = 0.2 * self.W
                field = (0.02 + np.exp(-((self.xx - xc) / wb) ** 2) + 0.8 * np.exp(-((self.xx - xc) / (0.06 * self.W)) ** 2)
                         )[..., None]
                gain = 14.0
            else:
                c, R = lamp["c"], lamp["R"]
                field = (0.006 + np.exp(-((self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2) / (2 * R * R)))[..., None]
                gain = 18.0
            x = Te * self.lamp * gain * B * field
            lum = x[..., 0] * 0.3 + x[..., 1] * 0.55 + x[..., 2] * 0.15
            lum2 = lum / (1 + lum)                                    # Reinhard: conserva el vitral
            glow = x * (lum2 / np.maximum(lum, 1e-5))[..., None]
            glow = glow + cv2.GaussianBlur(glow, (0, 0), 7 * u) * 0.2
        return np.clip(img, 0, 1), glow, alpha, (F, B, lamp)

    # ------------------------------------------------------------------------------------------------
    def frame(self, t):
        tq = q(t)
        img, glow, alpha, (F, B, lamp) = self.arpillera(t)
        k = int(round(tq * UFPS))
        return self.world.render_at(img, tq, T, F=F, B=B, lamp=lamp, flick=float(self.flick[k % len(self.flick)]),
                                    wool_tied=tq >= T["rise"][1] - 1e-6, dangle=self.dangle(tq), k=k,
                                    glow=glow, alpha=alpha)

    def frame_srgb8(self, t):
        return (linear_to_srgb(self.frame(t)) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------
# encuadres
# ------------------------------------------------------------------------------------------------------
def layout_for(W, H, sc):
    """Dónde cuelga cada arpillera en el encuadre final (mundo = último cuadro) y las tomas de cámara.
    Medidas pensadas para 1920×1080 (o 1080×1920) y escaladas al tamaño pedido."""
    from .territory import build_desert, build_towers, exit_points
    ex = exit_points(sc.wool.pts, sc.u)
    land = W >= H
    f = W / (1920 if land else 1080)
    F = lambda v: v * f
    if land:
        nb = [(int(round(1280 * f)), int(round(960 * f))), (int(round(1200 * f)), int(round(960 * f)))]
    else:
        nb = [(int(round(900 * f)), int(round(1200 * f))), (int(round(880 * f)), int(round(1180 * f)))]
    des = build_desert(*nb[0])
    tow = build_towers(*nb[1])
    nb_u = [min(s_) / 1080 for s_ in nb]
    sub = [("Guía para comunidades ante megaproyectos de energía", "small")]
    if land:
        main_x = (F(440), F(1480))
        z0 = 1.66
        title = dict(rect=tuple(F(v) for v in (600, 840, 1320, 1044)), font="timesi", lines=[
            (F(904), F(38), [("Sistema", "floss"), ("de", "floss"), ("Alerta", "wool")]),
            (F(962), F(38), [("Temprana", "floss"), ("Comunitario", "floss")]),
            (F(1008), F(19), sub)])
        line = (F(960), F(150), 0.000028 / f)
        nbx = [(F(-190), F(380)), (F(1540), F(2110))]
    else:
        main_x = (F(270), F(810))
        z0 = 1.9
        title = dict(rect=tuple(F(v) for v in (150, 1415, 930, 1705)), font="timesi", lines=[
            (F(1482), F(40), [("Sistema", "floss"), ("de", "floss")]),
            (F(1544), F(40), [("Alerta", "wool"), ("Temprana", "floss")]),
            (F(1606), F(40), [("Comunitario", "floss")]),
            (F(1660), F(19), sub)])
        line = (F(540), F(330), 0.000026 / f)
        nbx = [(F(-250), F(220)), (F(860), F(1330))]
    L = dict(line=line, main=dict(size=(sc.W, sc.H), x=main_x),
             neighbors=[dict(img=des["img"], x=nbx[0], exit=des["exit"], wool_w=8.0 * nb_u[0], red=des["red"],
                             lit=des["lit"]),
                        dict(img=tow["img"], x=nbx[1], exit=tow["exit"], wool_w=8.0 * nb_u[1], red=tow["red"],
                             lit=tow["lit"])],
             title=title, wool_exit=ex, wool_w=8.5 * sc.u, z0=z0)
    xm, ym, k = L["line"]
    y_top = ym + 3 * f
    cy = y_top + (H / 2 - F(66)) / z0
    L["c0"] = ((main_x[0] + main_x[1]) / 2, cy)
    # tomas: acercamientos por pasos (centros en la arpillera, sin salirse de la tela)
    s_img = (main_x[1] - main_x[0]) / sc.W

    def clamp(c, z):
        hx, hy = W / 2 / (z * s_img), H / 2 / (z * s_img)
        cx_ = float(np.clip(c[0], min(hx, sc.W / 2), max(sc.W - hx, sc.W / 2)))
        cy_ = float(np.clip(c[1], min(hy, sc.H / 2), max(sc.H - hy, sc.H / 2)))
        return ("img", (cx_, cy_))

    zs = z0 * 1.75
    c_st = np.mean(np.array(sc.info["stakes"]), axis=0) - (0, 30 * sc.u)
    c_no = sc.notice_c + (0, 20 * sc.u)
    c_in = sc.H4 + (-40 * sc.u, 10 * sc.u)
    ph = sc.pull_hand
    c_pu = (ph + sc.H1 - (0, 100 * sc.u)) / 2
    span = np.abs(ph - (sc.H1 - (0, 200 * sc.u))) + 260 * sc.u
    zp = z0 * float(np.clip(min(W / (span[0] * s_img * z0), H / (span[1] * s_img * z0)), 1.0, 1.6))
    L["shots"] = [(1.0, 1.35, zs, clamp(c_st, zs)), (1.8, 2.02, zs, clamp(c_no, zs)),
                  (2.42, 2.64, zs, clamp(c_in, zs)), (2.98, 3.32, z0, "c0"),
                  (10.15, 10.45, zp, clamp(c_pu, zp)), (T["dolly"][0], T["dolly"][1], 1.0, "final")]
    return L
