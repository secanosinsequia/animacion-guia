"""«Hilván»: línea de tiempo (16,93 s, 24 fps animados en dos = 12 imágenes únicas por segundo).

Conocer   0,0–1,2    póster: la arpillera cuelga de un cordel de cáñamo; una aguja grande, enhebrada con
                     hilo negro, cuelga cerca de la cámara (el foco pasa de la tela a la aguja) y baja a coser
Vigilar   1,2–4,6    corte a un primer plano quieto: la aguja hilvana las señales tempranas —estacas de
                     topógrafo, un AVISO prendido con alfiler de gancho, una cañería con su llave de paso
                     que sale del río—; corte al plano general: hilvana en el cerro la torre de alta
                     tensión, solo como contorno (lo que vendría); tira del hilo y la tela se frunce
Alertar   4,65–7,9   baja la luz de la sala y la tela se vuelve linterna; una lámpara de mano sigue el hilo
                     por las señales, sube a la torre del cerro y baja por la línea escondida, torre por
                     torre, hasta la casa roja; se levanta y se ve la línea completa (y el saco: HARINA);
                     clic: se apaga la lámpara y el tubo de la sala titila antes de prender
Responder 8,0–11,9   la lana roja calca por delante la ruta escondida (un nudo en cada torre) y pasa de mano
                     en mano: se enciende cada ventana; corte: la vigía tira del hilván, la torre se descose
                     hacia la maraña a sus pies y la ruta entera se frunce y se suelta; la lámpara vuelve
                     un instante: el revés está vacío, solo quedan los pinchazos y la lana encima
Red       11,9–16,93 la lana sube y se anuda al cordel; la cámara se aleja por pasos: el cordel sostiene otras
                     arpilleras; desde el nudo, el rojo corre por el cordel, baja a cada vecina y enciende su
                     ventana. La aguja del comienzo baja por un hilo rojo hasta la etiqueta y borda la «a»
                     final de «Alerta»
"""
import cv2
import numpy as np

from satc_intro.color import lin, linear_to_srgb
from satc_intro.geometry import catmull_rom, resample
from satc_intro.noise import smooth_noise, smoothstep
from . import territory
from .figures import needle_sprite, doll, doll_layer
from .signals import build_signals
from .thread import Sprite, Stitch, Yarn, composite, composite_T, knot
from .world import World

FPS = 24
UFPS = 12
DURATION = 16.93

T = dict(hook=(0.92, 1.17), stakes=(1.17, 1.8), notice=(1.85, 2.4), intake=(2.45, 2.98), pylon=(3.3, 3.95),
         taut=(4.0, 4.6), dim=(4.65, 4.9), full=(4.9, 7.59), curtain=(7.59, 7.92),
         trace=(8.0, 8.58), wool=(8.58, 9.08), pull=(9.42, 10.82), unstitch=(9.62, 10.17), route=(10.25, 10.75),
         flash=(10.84, 11.84), rise=(11.92, 12.42), dolly=(11.92, 13.42), front=(13.52, 14.82),
         end_stitch=(14.87, 15.92))
BLK = "#141516"
WOOL = "#c3241c"


def q(t):
    """Tiempo cuantizado a la imagen única (stop-motion en dos)."""
    return np.floor(t * UFPS + 1e-6) / UFPS


def fidx(tq, t0):
    """Cuántas imágenes únicas van desde la primera que cae en t0 o después (0, 1, 2…)."""
    return int(round(tq * UFPS)) - int(np.ceil(t0 * UFPS - 1e-6))


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def _wobbly(pts, rng, amp, step):
    """Camino a mano: la lana nunca va derecha (ondea largo y, cada tanto, se quiebra un poco donde la
    fijaron con una puntada)."""
    path = catmull_rom(np.array(pts, np.float64), 14)
    path = resample(path, step)
    n = len(path)
    off = np.stack([smooth_noise((1, n), 30, rng)[0] - 0.5, smooth_noise((1, n), 30, rng)[0] - 0.5], 1)
    fine = np.stack([smooth_noise((1, n), 7, rng)[0] - 0.5, smooth_noise((1, n), 7, rng)[0] - 0.5], 1)
    return path + off * amp + fine * amp * 0.35


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


def _ball_sprite(c, r, u, seed):
    """Un ovillo de hilo negro: una bola con luz y sombra, y las vueltas del hilo a la vista (círculos
    máximos en distintas direcciones; solo la mitad de adelante de cada vuelta)."""
    from .thread import LIGHT
    rng = np.random.default_rng(seed)
    pad = 4
    n = int(2 * r + 2 * pad) + 1
    x0, y0 = int(c[0] - r - pad), int(c[1] - r - pad)
    yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n].astype(np.float32) + 0.5
    dx, dy = (xx - c[0]) / r, (yy - c[1]) / r
    rr = np.sqrt(dx * dx + dy * dy)
    a = np.clip((1 - rr) * r + 0.5, 0, 1).astype(np.float32)
    nz = np.sqrt(np.clip(1 - rr * rr, 0, 1))
    lam = np.clip(dx * LIGHT[0] + dy * LIGHT[1] + nz * LIGHT[2], 0, 1)
    base = lin("#2b2c2e")
    val = (0.5 + 0.9 * lam) * (0.8 + 0.2 * nz)
    rgb = base[None, None, :] * val[..., None]
    # las vueltas: hebras un poco más claras (brillo del hilo) que dan la forma de bola
    strands = np.zeros((n, n), np.float32)
    for _ in range(int(10 + r / (2.2 * u))):
        ax = rng.normal(0, 1, 3)
        ax /= np.linalg.norm(ax) + 1e-9
        e1 = np.cross(ax, [0.3, 0.9, 0.1])
        e1 /= np.linalg.norm(e1) + 1e-9
        e2 = np.cross(ax, e1)
        t = np.linspace(0, 2 * np.pi, 90)
        P = np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2)
        vis = P[:, 2] > 0.05
        pts = np.stack([c[0] - x0 + P[:, 0] * r * 0.97, c[1] - y0 + P[:, 1] * r * 0.97], 1)
        for k in range(len(t) - 1):
            if vis[k] and vis[k + 1]:
                cv2.line(strands, (int(pts[k, 0] * 4), int(pts[k, 1] * 4)),
                         (int(pts[k + 1, 0] * 4), int(pts[k + 1, 1] * 4)), float(rng.uniform(0.5, 1.0)),
                         max(1, int(1.2 * u * 4 / 4)), cv2.LINE_AA, shift=2)
    strands = cv2.GaussianBlur(strands, (0, 0), 0.5)
    rgb = rgb * (1 - 0.5 * strands[..., None]) + lin("#77797d")[None, None, :] * (0.5 * strands * (0.3 + lam))[..., None]
    rgb = rgb * (1 - 0.5 * np.clip(rr - 0.8, 0, 1) / 0.2)[..., None]      # el borde se va a la sombra
    sh = cv2.GaussianBlur(a, (0, 0), max(1.0, r * 0.25))
    sh = cv2.warpAffine(sh, np.float32([[1, 0, r * 0.25], [0, 1, r * 0.3]]), (n, n))
    return Sprite(x0, y0, (rgb * a[..., None]).astype(np.float32), a, sh)


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
        self.pk = self._pucker_prepare([(self.H2, self.H3), (self.H3, self.H4)], [self.H2, self.H3, self.H4])
        # la ruta escondida (la línea de torres del revés): al tirar del hilván se frunce por delante a lo largo
        # de toda la ruta y se suelta desde la punta lejana, dejando pinchazos donde estaba cada torre
        self.route = [np.asarray(self.pylon_top, np.float64) + (0, 120 * u)] + \
                     [np.asarray(b_, np.float64) - (0, 95 * u) for b_ in info["back_line"]]
        pr = self._pucker_prepare(list(zip(self.route[:-1], self.route[1:])), self.route[1:], gather_k=0.12,
                                  tug_k=0.45, arc=True, seed=8, pin_k=0.6, irregular=1.0)
        pr.pop("xx")
        pr.pop("yy")
        self.pk_route = pr
        self.route_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.array(self.route), axis=0),
                                                                          axis=1))])
        self.route_holes = self._route_holes()
        self.lamp_plan = self._lamp_plan()
        # ---------------------------------------------------------------- vecinas y vecinos, y la lana roja
        self._dolls_prepare()
        self._wool_prepare(rng)
        # ---------------------------------------------------------------- humo y agujeros
        self.chimney = info["chimney"]
        self.tower_holes = [self._hole(p, np.random.default_rng(int(p[0] * 13 + p[1])), faint=True)
                            for p in self._tower_hole_pts()]
        # luz y grano
        self.flick = np.random.default_rng(seed + 99).normal(0, 0.013, 400)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.yy, self.xx = yy, xx
        self.lamp = np.array([1.0, 0.88, 0.72], np.float32)        # luz de tarde, cálida
        # ---------------------------------------------------------------- el mundo
        self.layout = layout_for(self.OW, self.OH, self)
        self.world = World(self.OW, self.OH, self.layout)

    # ------------------------------------------------------------------------------------------------
    def _tangle(self, amount):
        """La maraña de hilo negro a los pies de la vigía: lazadas flojas echadas en el suelo, más anchas que
        altas y abiertas (se ve la tela entre ellas: es hilo, no una mancha), que se van sumando (amount
        0..1). Siempre las mismas, en el mismo orden: primero las chicas, después las grandes."""
        u = self.u
        if not hasattr(self, "_tangle_loops"):
            rg = np.random.default_rng(81)
            loops = []
            for j in range(10):
                big = j in (3, 6, 8)
                cx = rg.normal(0, 9 if big else 7) * u
                cy = -abs(rg.normal(0, 2.0)) * u
                rx = (rg.uniform(15, 21) if big else rg.uniform(6, 11)) * u
                ry = (rg.uniform(5.5, 8.0) if big else rg.uniform(2.5, 4.5)) * u
                a0 = rg.uniform(0, 2 * np.pi)
                t = np.linspace(a0, a0 + rg.uniform(1.3, 1.8) * np.pi, 30)
                tilt = rg.normal(0, 0.18)
                ex_, ey_ = rx * np.cos(t), ry * np.sin(t)
                pts = np.stack([cx + ex_ * np.cos(tilt) - ey_ * np.sin(tilt) + rg.normal(0, 0.5, 30) * u,
                                cy + ex_ * np.sin(tilt) + ey_ * np.cos(tilt) + rg.normal(0, 0.5, 30) * u], 1)
                loops.append((pts, "#3a3b3e" if j % 3 == 1 else "#232426"))
            self._tangle_loops = loops
            self._tangle_w = 30 * u
        n = int(np.clip(round(len(self._tangle_loops) * amount), 1, len(self._tangle_loops)))
        out = []
        base = self.ball_pt + np.array([0.0, -1.5 * u])
        for j, (pts, col) in enumerate(self._tangle_loops[:n]):
            y_ = Yarn(pts + base, 2.0 * u, col, np.random.default_rng(90 + j), fuzz=0, kind="floss")
            out += y_.chunks
        return out

    def _lamp_plan(self):
        """La lámpara de mano, imagen por imagen: sigue el hilo por las señales (estacas, aviso, cañería),
        sube a la torre del cerro y baja por la línea escondida, torre por torre, hasta la casa roja, donde
        se queda; después se levanta (una imagen) y se abre: la línea completa, una sola vez (None)."""
        u = self.u
        R = 0.115 * max(self.W, self.H)
        B0 = 0.32                                      # la tela entera brilla; donde está la lámpara, mucho más
        signals = [np.mean(np.array(self.info["stakes"]), axis=0), np.asarray(self.notice_c, np.float64),
                   np.asarray(self.H4, np.float64)]
        plan = []
        for j, p_ in enumerate(signals):                       # (en las señales: solo su hebra, dos imágenes)
            rj = np.random.default_rng(30 + j)
            plan += [(p_ + rj.normal(0, 5 * u, 2), R * 0.8, B0, False) for _ in range(2)]
        stops = [np.asarray(p_, np.float64) for p_ in self.route]
        for j, p_ in enumerate(stops):
            if j > 0:
                plan.append(((stops[j - 1] + p_) / 2 + np.array([0.0, -30 * u]), R * 1.25, B0, True))   # la mueve
            rj = np.random.default_rng(40 + j)
            for _ in range(2 if j < len(stops) - 1 else 4):
                plan.append((p_ + rj.normal(0, 6 * u, 2), R, B0, True))                          # pulso de mano
        plan.append((stops[-1] + np.array([0.0, -60 * u]), R * 2.6, 0.4, True))                  # la levanta
        n = int(round((T["full"][1] - T["full"][0]) * UFPS))
        plan += [None] * max(0, n - len(plan))
        return plan

    def _hand_shadow(self, c, R, k):
        """La mano que sostiene la lámpara por detrás de la tela: el mango y el puño (con sus dedos) tapan el borde
        de abajo del foco, y el antebrazo se va hacia abajo; como están a unos centímetros de la tela, la sombra
        es blanda. Devuelve una máscara HxW (0..1)."""
        key = int(R)
        cache = self.__dict__.setdefault("_hand_cache", {})
        if key not in cache:
            ang = np.deg2rad(14)                       # la mano viene de abajo, un poco desde la derecha
            v = np.array([np.sin(ang), np.cos(ang)])  # a lo largo del brazo (hacia abajo)
            n_ = np.array([v[1], -v[0]])
            P_ = lambda s_, e_: s_ * R * v + e_ * R * n_
            polys = [
                [P_(0.42, -0.1), P_(0.42, 0.1), P_(1.25, 0.12), P_(1.25, -0.12)],       # el mango de la lámpara
                [P_(1.3, -0.25), P_(1.3, 0.27), P_(3.7, 0.38), P_(3.7, -0.34)],          # el antebrazo
            ]
            ell = [(P_(1.05, 0.02), (0.3, 0.36), 0.0)]                                  # el puño
            ell += [(P_(0.8 + 0.15 * j, 0.27), (0.1, 0.072), 0.0) for j in range(4)]    # los dedos
            ell += [(P_(0.84, -0.24), (0.07, 0.16), 25.0)]                              # el pulgar
            pts_all = np.vstack([np.array(pl) for pl in polys])
            pad = 0.5 * R
            x0, y0 = pts_all.min(axis=0) - pad
            x1, y1 = pts_all.max(axis=0) + pad
            w_, h_ = int(x1 - x0) + 1, int(y1 - y0) + 1
            m = np.zeros((h_, w_), np.float32)
            off = np.array([x0, y0])
            for pl in polys:
                q_ = (np.array(pl) - off) * 4
                cv2.fillPoly(m, [np.round(q_).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
            base_ang = float(np.rad2deg(np.arctan2(v[1], v[0])) - 90)
            for (c_, (ea, eb), extra) in ell:
                q_ = (np.asarray(c_) - off) * 4
                cv2.ellipse(m, (int(q_[0]), int(q_[1])), (int(ea * R * 4), int(eb * R * 4)), base_ang + extra,
                            0, 360, 1.0, -1, cv2.LINE_AA, shift=2)
            m = cv2.GaussianBlur(m, (0, 0), 0.06 * R)
            cache[key] = (m, off)
        m, off = cache[key]
        out = np.zeros((self.H, self.W), np.float32)
        rj = np.random.default_rng(700 + k)
        ox, oy = int(round(c[0] + off[0] + rj.normal(0, 2 * self.u))), int(round(c[1] + off[1] + rj.normal(0, 2 * self.u)))
        h_, w_ = m.shape
        xa, ya, xb, yb = max(0, ox), max(0, oy), min(self.W, ox + w_), min(self.H, oy + h_)
        if xb > xa and yb > ya:
            out[ya:yb, xa:xb] = m[ya - oy:yb - oy, xa - ox:xb - ox]
        return out

    def _route_pinholes(self):
        """Por donde pasaba el hilo de cada torre escondida quedan pinchazos: a contraluz dibujan su silueta
        en puntitos de luz."""
        u = self.u
        hgt = 200 * u
        m = np.zeros((self.H, self.W), np.float32)
        rg = np.random.default_rng(29)
        for base in self.info["back_line"]:
            bx, by = float(base[0]), float(base[1])
            wb, wt = hgt * 0.24, hgt * 0.07
            waist = by - hgt * 0.72
            segs = [((bx - wb / 2, by), (bx - wt / 2, waist)), ((bx + wb / 2, by), (bx + wt / 2, waist)),
                    ((bx - wt / 2, waist), (bx, by - hgt)), ((bx + wt / 2, waist), (bx, by - hgt)),
                    ((bx - hgt * 0.3, by - hgt * 0.8), (bx + hgt * 0.3, by - hgt * 0.8))]
            for (p0, p1) in segs:
                p0, p1 = np.array(p0), np.array(p1)
                n = max(2, int(np.hypot(*(p1 - p0)) / (15 * u)) + 1)
                for f in np.linspace(0, 1, n):
                    c = p0 + (p1 - p0) * f + rg.normal(0, 1.2 * u, 2)
                    ax = rg.uniform(0.8, 1.4) * u
                    cv2.ellipse(m, (int(c[0] * 4), int(c[1] * 4)), (max(1, int(ax * 4)), max(1, int(ax * 3))),
                                float(rg.uniform(0, 180)), 0, 360, float(rg.uniform(0.6, 1.0)), -1, cv2.LINE_AA,
                                shift=2)
        return np.clip(cv2.GaussianBlur(m, (0, 0), 0.6 * u), 0, 1)

    def _route_holes(self):
        """Los pinchazos que deja cada torre escondida cuando sale su hilo: (largo en la ruta, [sprites]).
        (La misma geometría con que se dibujan en el revés.)"""
        u = self.u
        hgt = 200 * u
        out = []
        for i, base in enumerate(self.info["back_line"]):
            bx, by = float(base[0]), float(base[1])
            wb, wt = hgt * 0.24, hgt * 0.07
            waist = by - hgt * 0.72
            pts = [(bx - wb / 2, by), (bx + wb / 2, by), (bx - wt / 2, waist), (bx + wt / 2, waist), (bx, by - hgt),
                   (bx - hgt * 0.3, by - hgt * 0.8), (bx + hgt * 0.3, by - hgt * 0.8)]
            sps = [self._hole(np.array(p_), np.random.default_rng(int(p_[0] * 7 + p_[1] * 3)), faint=True,
                              R=4.2 * u, k=0.9) for p_ in pts]
            out.append((float(self.route_arc[i + 1]), sps))
        return out

    def _tower_hole_pts(self):
        pts = []
        for i in self.tower_idx:
            for p in self.hilvan[i][2:4]:
                if all(np.hypot(*(p - q_)) > 3.2 * self.u for q_ in pts):
                    pts.append(p)
        return pts

    def _hole(self, p, rng, faint=False, R=None, k=None):
        """Agujero que deja una puntada al sacarla. faint: apenas una marca (memoria, no mancha)."""
        u = self.u
        R = 2.2 * u if R is None else R
        x0, y0 = int(p[0] - 8 * u), int(p[1] - 8 * u)
        n = int(16 * u) + 2
        yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n].astype(np.float32) + 0.5
        dx, dy = xx - p[0], yy - p[1]
        d = np.hypot(dx, dy)
        wob = 1 + 0.18 * np.sin(np.arctan2(dy, dx) * 3 + rng.uniform(0, 6))
        core = np.clip(R * 0.6 * wob - d + 0.5, 0, 1)
        rim = np.clip(1 - np.abs(d - R * 1.15 * wob) / (0.8 * u), 0, 1)
        lit = np.clip((dx * 0.64 + dy * 0.56) / (d + 1e-3), 0, 1)
        k = (0.55 if faint else 1.0) if k is None else k
        kr = 1.6 if faint else 1.0                                      # el anillo oscuro: se ve sobre el celeste
        a = np.clip((core * 0.8 + rim * (0.22 + 0.32 * lit) * kr) * k, 0, 1).astype(np.float32)
        hole_col = lin("#d9ccb4") if faint else lin("#1e150e")        # por el agujero se ve la pared, clara
        rgb = (hole_col[None, None, :] * core[..., None] * 0.8
               + lin("#3a2e22")[None, None, :] * (rim * (0.22 + 0.32 * lit) * kr)[..., None]) * k
        return Sprite(x0, y0, rgb.astype(np.float32), a, None)

    def _back_mask(self, rng):
        """El revés: una sola hebra gruesa (≈7 px) con holgura que une las señales hasta la torre del cerro,
        y desde ahí todo el proyecto hilvanado por detrás: una fila de torres que baja cruzando el valle,
        por encima de las casas y del río, con su cable. Solo se ve a contraluz. Se guardan por separado
        la hebra de las señales y la línea de torres (la lámpara las encuentra en ese orden), y los
        pinchazos por donde la línea atraviesa la tela (se ve cosida, no dibujada)."""
        W, H, u = self.W, self.H, self.u
        layers = {k_: (np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)) for k_ in ("sig", "line")}
        pins = np.zeros((H, W), np.float32)
        rp = np.random.default_rng(29)

        def stroke(pts, r0, rr=0.4, farness=None, layer="sig", pinholes=False):
            core, loose = layers[layer]
            pts = resample(np.asarray(pts, np.float64), 1.5)
            n = len(pts)
            if n < 2:
                return
            rad = r0 * (1 - rr / 2 + rr * smooth_noise((1, n), 25, rng)[0])
            far = smooth_noise((1, n), 60, rng)[0] if farness is None else np.full(n, farness)
            for (x, y), r, f in zip(pts, rad, far):
                cv2.circle(core, (int(x * 4), int(y * 4)), max(1, int(r * 4)), float(1 - 0.5 * f), -1, cv2.LINE_AA,
                           shift=2)
                cv2.circle(loose, (int(x * 4), int(y * 4)), max(1, int(r * 4)), float(f), -1, cv2.LINE_AA, shift=2)
            if pinholes:          # puntadas: cada tanto el hilo atraviesa la tela (a mano: ni parejas ni iguales)
                arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
                lo, hi = (1.7, 2.6) if self.portrait else (1.2, 2.1)
                s_ = rp.uniform(0, 12 * u)
                while s_ < arc[-1]:
                    if rp.random() > 0.1:                              # (alguna falta)
                        i_ = min(int(np.searchsorted(arc, s_)), len(pts) - 1)
                        c = pts[i_] + rp.normal(0, 1.0 * u, 2)
                        ax = rp.uniform(lo, hi) * u
                        ay = ax * rp.uniform(0.55, 0.9)
                        if rp.random() < 0.15:                         # alargado: la aguja entró sesgada
                            ax, ay = ax * 1.6, ay * 0.6
                        ang = float(rp.uniform(0, 180))
                        cv2.ellipse(pins, (int(c[0] * 4), int(c[1] * 4)), (max(1, int(ax * 4)), max(1, int(ay * 4))),
                                    ang, 0, 360, float(rp.uniform(0.7, 1.0)), -1, cv2.LINE_AA, shift=2)
                        if rp.random() < 0.1:                          # doble: se pinchó dos veces
                            c2 = c + rp.normal(0, 1.0, 2) * ax * 1.4
                            cv2.ellipse(pins, (int(c2[0] * 4), int(c2[1] * 4)),
                                        (max(1, int(ax * 2.4)), max(1, int(ay * 2.4))), ang, 0, 360,
                                        float(rp.uniform(0.6, 0.9)), -1, cv2.LINE_AA, shift=2)
                    s_ += 12 * u * rp.uniform(0.7, 1.3)

        def sag(a, b, k=0.012, wob=0.0):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            L = float(np.hypot(*(b - a)))
            nrm = np.array([-(b - a)[1], (b - a)[0]]) / (L + 1e-9)
            if wob > 0:                                 # hilo de verdad: la caída nunca es una curva perfecta
                sgn = rng.choice([-1, 1])
                pts_ = [a + (b - a) * f + np.array([0, 4 * f * (1 - f) * L * k]) + nrm * L * 0.02 * sgn * 4 * f * (1 - f)
                        + (nrm * rng.normal(0, wob) if 0 < f < 1 else 0) for f in (0, 0.22, 0.5, 0.78, 1)]
                return catmull_rom(np.array(pts_), 16)
            mid = (a + b) / 2 + nrm * L * 0.02 * rng.choice([-1, 1]) + np.array([0, L * k])
            return catmull_rom(np.array([a, mid, b]), 30)

        for a, b in zip(self.back_path[:-1], self.back_path[1:]):
            stroke(sag(a, b), 3.6 * u, layer="sig")
        # la línea de torres proyectada
        hgt = 200 * u
        tops = [np.asarray(self.pylon_top, np.float64)]
        arms = [(tops[0] - (hgt * 0.34, -hgt * 0.26), tops[0] + (hgt * 0.34, hgt * 0.26))]
        for base in self.info["back_line"]:
            bx, by = base
            wb, wt = hgt * 0.24, hgt * 0.07
            waist = by - hgt * 0.72
            top = np.array([bx, by - hgt])
            legs = [((bx - wb / 2, by), (bx - wt / 2, waist)), ((bx + wb / 2, by), (bx + wt / 2, waist)),
                    ((bx - wt / 2, waist), (bx, top[1])), ((bx + wt / 2, waist), (bx, top[1])),
                    ((bx - wb / 2, by), (bx + wb * 0.2, by - hgt * 0.36)), ((bx + wb / 2, by), (bx - wb * 0.2, by - hgt * 0.36)),
                    ((bx - hgt * 0.3, by - hgt * 0.8), (bx + hgt * 0.3, by - hgt * 0.8))]
            for (p0, p1) in legs:
                # con un poco de holgura: el hilo nunca queda de regla
                mid = (np.array(p0) + np.array(p1)) / 2 + rng.normal(0, 1.6 * u, 2)
                stroke(catmull_rom(np.array([p0, mid, p1]), 12), 4.0 * u, farness=0.0, layer="line", pinholes=True)
            tops.append(top)
            arms.append((np.array([bx - hgt * 0.3, by - hgt * 0.8]), np.array([bx + hgt * 0.3, by - hgt * 0.8])))
        for (a0, a1), (b0, b1) in zip(arms[:-1], arms[1:]):
            for pa, pb in ((a0, b0), (a1, b1)):
                stroke(sag(pa, pb, 0.09, wob=2.2 * u), 3.0 * u, rr=0.35, farness=0.1, layer="line", pinholes=True)
        out = {}
        for k_, (core, loose) in layers.items():
            core = cv2.GaussianBlur(np.clip(core, 0, 1), (0, 0), 0.6 * u)
            if k_ == "line":
                # hilo torcido, con pelusa: la sombra no es un trazo limpio (fibras sueltas en el borde y el
                # torcido que aclara y oscurece a lo largo)
                rf = np.random.default_rng(31)
                ys_, xs_ = np.nonzero(core > 0.5)
                hair = np.zeros_like(core)
                if len(xs_):
                    for i_ in rf.choice(len(xs_), min(len(xs_), 5000), replace=False):
                        a_ = rf.uniform(0, np.pi)
                        L_ = rf.uniform(2.5, 7.0) * u
                        x_, y_ = xs_[i_], ys_[i_]
                        cv2.line(hair, (int(x_ * 4), int(y_ * 4)),
                                 (int((x_ + L_ * np.cos(a_)) * 4), int((y_ + L_ * np.sin(a_)) * 4)),
                                 float(rf.uniform(0.35, 0.8)), 1, cv2.LINE_AA, shift=2)
                twist = 0.8 + 0.2 * smooth_noise(core.shape, 2.5 * u, rf)
                core = np.maximum(core * twist, cv2.GaussianBlur(hair, (0, 0), 0.5) * 0.7)
            halo = cv2.GaussianBlur(np.clip(loose, 0, 1), (0, 0), 3.2 * u)
            out[k_] = (np.clip(core * 1.25, 0, 1), np.clip(halo * 0.6, 0, 1))
        self.back_core_sig, self.back_halo_sig = out["sig"]
        self.back_core_line, self.back_halo_line = out["line"]
        self.back_core = np.maximum(self.back_core_sig, self.back_core_line)
        self.back_halo = np.maximum(self.back_halo_sig, self.back_halo_line)
        self.route_pins = np.clip(cv2.GaussianBlur(pins, (0, 0), 0.6 * u), 0, 1)
        return np.clip(self.back_core * 1.0 + self.back_halo, 0, 1)

    def _pinholes(self):
        """Pinchazos chicos e irregulares; tres variantes de brillo que se turnan (la luz de mano titila)."""
        u = self.u
        rg = np.random.default_rng(17)
        holes = []
        for it in self.hilvan:
            if np.hypot(*(it[3] - it[2])) < 7 * u:          # las letras del aviso no: serían una mancha
                continue
            for p in (it[2], it[3]):
                ax = rg.uniform(0.6, 1.3) * u
                holes.append((p, ax, ax * rg.uniform(0.5, 1.0), rg.uniform(0, 180), rg.uniform(0.35, 0.85)))
        self.pin_var = []
        for v in range(3):
            m = np.zeros((self.H, self.W), np.float32)
            for (p, ax, ay, ang, b) in holes:
                bb = float(np.clip(b * rg.uniform(0.6, 1.25), 0, 1))
                cv2.ellipse(m, (int(p[0] * 4), int(p[1] * 4)), (max(1, int(ax * 4)), max(1, int(ay * 4))),
                            float(ang), 0, 360, bb, -1, cv2.LINE_AA, shift=2)
            self.pin_var.append(np.clip(cv2.GaussianBlur(m, (0, 0), 0.6 * u), 0, 1))
        return self.pin_var[0]

    # ------------------------------------------------------------------------------------------------
    def _pucker_prepare(self, segs, pins, gather_k=0.2, tug_k=1.0, arc=False, seed=3, pin_k=1.0, irregular=0.0):
        """Frunce: pliegues perpendiculares al hilo que arrastran la tela (y su estampado), pliegues que
        irradian de cada huella, un tirón que acerca y ladea lo de los extremos y el ruedo que sube.
        Devuelve los campos; con arc, también a qué altura del hilo (largo desde el primer punto) está cada
        píxel, para soltar el frunce de a tramos."""
        u, W, H = self.u, self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        DX = np.zeros_like(xx)
        DY = np.zeros_like(xx)
        S = np.zeros_like(xx)
        TM = np.zeros_like(xx)
        ARC = np.zeros_like(xx) if arc else None
        GM = np.zeros_like(xx) if arc else None
        arc0 = 0.0
        rngp = np.random.default_rng(seed)
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
            g = (np.exp(-(e / (58 * u * (1 + 0.3 * wob))) ** 2) * smoothstep(-26 * u, 34 * u, s_)
                 * smoothstep(L + 26 * u, L - 34 * u, s_))
            beyond = np.maximum(np.maximum(-s_, s_ - L), 0)
            side = np.clip((s_ - L / 2) / (L / 2), -1, 1)
            tug = np.exp(-(e / (240 * u)) ** 2) * np.exp(-beyond / (380 * u)) * side * tug_k
            ph = 2 * np.pi * s_ / (lam * (1 + 0.45 * wob2)) + wob * 3.2 + e / (60 * u) * 0.9
            gather = gather_k * L
            if arc:                                       # el tramo que más pesa en cada píxel da su altura
                wgt = np.maximum(g, 0.3 * np.abs(tug))
                sel = wgt > GM
                ARC[sel] = (arc0 + np.clip(s_, 0, L))[sel]
                GM[sel] = wgt[sel]
                arc0 += L
            # el estampado se arrastra de verdad: compresión en los pliegues y ondulación lateral
            ds = (g * 0.2 * (np.clip(s_, 0, L) - L / 2) + gather / 2 * tug * (1 - g)
                  + g * 1.05 * lam / (2 * np.pi) * np.sin(ph))
            de = g * (0.06 * e + 1.8 * u * np.cos(ph) * (0.6 + 0.4 * wob2))
            # los extremos se ladean hacia el hilo (las casas se inclinan)
            de = de + 3.0 * u * np.abs(tug) * np.clip(e / (120 * u), -1, 1) * side
            DX += ds * t[0] + de * n[0]
            DY += ds * t[1] + de * n[1]
            hgt = np.abs(np.sin(ph / 2)) ** 0.5
            dh = np.gradient(hgt, axis=1) * t[0] + np.gradient(hgt, axis=0) * t[1]
            Ld = -0.64 * t[0] - 0.56 * t[1]
            S += g * (3.2 * lam / (2 * np.pi) * dh * Ld * 0.5 - 0.22 * (1 - hgt))
            TM += g * hgt
        # pliegues que irradian de cada huella (el hilo tira de un punto); con irregular, cada abanico tiene
        # su número de pliegues y se tuercen al azar (sin anillos parejos que parezcan curvas de nivel)
        for p in pins:
            p = np.asarray(p, np.float64)
            dx, dy = xx - p[0], yy - p[1]
            r = np.hypot(dx, dy) + 1e-3
            th = np.arctan2(dy, dx)
            R = 70 * u
            gr = np.exp(-r / R) * smoothstep(3 * u, 14 * u, r) * pin_k
            if irregular > 0:
                nf = int(rngp.integers(4, 8))
                ph = nf * th + irregular * 2.6 * (smooth_noise(r.shape, 34 * u, rngp) - 0.5) * 2 \
                    + (1 - irregular) * 1.3 * np.sin(r / (22 * u))
            else:
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
        return dict(xx=xx, yy=yy, DX=DX * win, DY=DY * win, S=S * win, TM=np.clip(TM, 0, 1) * win, ARC=ARC,
                    L=arc0)

    def _pucker(self, img, Tm, A, alpha=None, A2=0.0, front=None):
        """Aplica el frunce con intensidad A (0..1); también a la silueta de la tela (alpha). A2: el frunce
        de la ruta escondida, que ya se soltó más allá de `front` (largo a lo largo de la ruta)."""
        if A <= 0.001 and A2 <= 0.001:
            return img, Tm, alpha
        pk = self.pk
        DX, DY, S, TM = A * pk["DX"], A * pk["DY"], A * pk["S"], A * pk["TM"]
        if A2 > 0.001:
            pr = self.pk_route
            w = A2 if front is None else A2 * np.clip((front - pr["ARC"]) / (40 * self.u), 0, 1)
            DX, DY, S, TM = DX + w * pr["DX"], DY + w * pr["DY"], S + w * pr["S"], TM + w * pr["TM"]
        mx = (pk["xx"] + DX).astype(np.float32)
        my = (pk["yy"] + DY).astype(np.float32)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        Tm = cv2.remap(Tm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if alpha is not None:
            alpha = cv2.remap(alpha, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img *= np.clip(1 + S, 0.55, 1.45)[..., None]
        Tm *= (1 - 0.55 * np.clip(TM, 0, 1))[..., None]
        return img, Tm, alpha

    # ------------------------------------------------------------------------------------------------
    def _dolls_prepare(self):
        """Cada muñeca en sus poses. La vigía (la más cercana a las estacas) tiende la lana y descose."""
        self.dolls = []
        houses = [h["center"] for h in self.info["houses"]]
        mk_spec = lambda sp, **kw: doll(sp["fx"], sp["base"], sp["hgt"], np.random.default_rng(1), 0,
                                        dress=sp["dress"], dress_kind=sp["dress_kind"], dress2=sp["dress2"],
                                        skin=sp["skin"], hair=sp["hair"], u=sp["u"], kind=sp["kind"], **kw)
        self._mk_spec = mk_spec
        for i, sp in enumerate(self.info["dolls"]):
            near = int(np.argmin([np.hypot(sp["fx"] - c[0], sp["base"] - c[1]) for c in houses]))
            d = dict(spec=sp, house=near, rest=doll_layer(mk_spec(sp, arms=sp["arms"])))
            self.dolls.append(d)
        k = int(np.argmin([np.hypot(d["spec"]["fx"] - self.first_stitch[0], d["spec"]["base"] - self.first_stitch[1])
                           for d in self.dolls]))
        self.vigia = k
        sp = self.dolls[k]["spec"]
        h = sp["hgt"] * (0.74 if sp["kind"] == "child" else 1.0)
        neck = sp["base"] - h * 0.68
        # sostiene la lana: mano estirada hacia la torre del cerro (por donde empieza a calcar la ruta)
        side = 1.0 if self.H1[0] >= sp["fx"] else -1.0
        self.hold_hand = np.array([sp["fx"] + side * h * 0.32, neck + h * 0.06])
        other = np.array([sp["fx"] - side * h * 0.27, neck + h * 0.26])
        hands = (other, self.hold_hand) if side > 0 else (self.hold_hand, other)
        self.dolls[k]["hold"] = doll_layer(mk_spec(sp, hands=hands))
        # tira del hilván: las dos manos hacia la torre, el cuerpo inclinado hacia atrás
        tside = 1.0 if self.H1[0] >= sp["fx"] else -1.0
        hands = (np.array([sp["fx"] + tside * h * 0.28, neck + h * 0.04]),
                 np.array([sp["fx"] + tside * h * 0.33, neck + h * 0.12]))
        self.pull_hand = (hands[0] + hands[1]) / 2
        self.dolls[k]["pull"] = [doll_layer(mk_spec(sp, hands=hands), lean=-tside * ln) for ln in (0.07, 0.15, 0.22)]
        self.ball_pt = np.array([sp["fx"] - tside * h * 0.24, sp["base"] - 6 * self.u])

    def _draw_dolls(self, img, tq, Tm=None):
        lit_t = self.house_times
        p0, p1 = T["pull"]
        for i, d in enumerate(self.dolls):
            lay = d["rest"]
            if i == self.vigia:
                tl = lit_t.get(self.vigia_house)
                if p0 <= tq < p1:
                    kk = int(round((tq - p0) * UFPS))
                    lay = d["pull"][0] if kk < 3 else d["pull"][(1, 2, 2, 1)[kk % 4]]   # se afirma, y tira
                elif self.vigia_takes:
                    if tl is not None and tl - 1e-6 <= tq < p0 and "take" in d:
                        lay = d["take"]                  # (toma la lana en la cadena, como las demás)
                elif T["trace"][0] <= tq < p0 and not self.portrait:
                    lay = d["hold"]                      # (16:9) la lana sale de su mano
            else:
                tl = lit_t.get(d["house"])
                if tl is not None and tq >= tl - 1e-6 and "take" in d:
                    lay = d["take"]                          # toma la lana y la pasa: queda en su mano
            composite(img, lay, shadow=0.5)
            if Tm is not None:
                composite_T(Tm, lay)

    # ------------------------------------------------------------------------------------------------
    def _wool_prepare(self, rng):
        """La vigía responde con lana roja. Primero calca por delante la ruta escondida, torre por torre (un
        nudo rojo en cada una), hasta la casa roja: lo que estaba oculto queda a la vista de todos. Después
        la lana pasa de mano en mano: cada vecina y vecino la toma y se enciende su ventana. Al final sube
        por el borde derecho hacia el cordel.
        En 16:9 sale de la mano de la vigía hacia la primera torre escondida. En 9:16 (donde la vigía queda
        bajo la ruta y la lana se devolvería por encima de sí misma) parte anudada en la torre del cerro y baja
        por la ruta igual que la lámpara: nunca se cruza ni hace lazos."""
        u, W, H = self.u, self.W, self.H
        hs = self.info["houses"]
        doors = [np.asarray(h["door"]) for h in hs]
        own = 2                                               # amarilla (la de la vigía)
        route = [np.asarray(p_, np.float64) for p_ in self.route]
        if self.portrait:
            way = [route[0]]
            loops = [route[0]]                                # también un nudo donde se anuda, en el cerro
            order = [0, 1, 3]                                 # roja, azul, blanca (bajando y después a la derecha)
        else:
            # (la torre del cerro ya está a la vista: calca desde la primera torre escondida, la que está
            # junto a ella)
            way = [self.hold_hand]
            # si la ruta vuelve por donde subió la lana (una horquilla), la subida se abre en curva hacia afuera
            d1 = route[1] - self.hold_hand
            if len(route) > 2 and np.linalg.norm(d1) > 1:
                d2 = route[2] - route[1]
                if np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9) < -0.3:
                    perp = np.array([-d1[1], d1[0]]) / np.linalg.norm(d1)
                    if np.dot(perp, d2) > 0:
                        perp = -perp
                    way.append((self.hold_hand + route[1]) / 2 + perp * 0.45 * np.linalg.norm(d1))
            loops = []
            order = [0, 1, 3]                                 # roja, azul, blanca
        for p_ in route[1:]:                          # pasa por cada torre escondida (ahí deja un nudo)
            way.append(p_)
            loops.append(p_)
        last_ring = way[-1]
        # de mano en mano: cada quien estira el brazo hacia donde viene la lana
        prev = route[-1]
        hand_pts = {}
        for i in order:
            if i == own:
                j = self.vigia
            else:
                j = next((j for j, d in enumerate(self.dolls) if d["house"] == i and j != self.vigia), None)
            if j is None:
                pt = doors[i] + (0, 26 * u)
            else:
                sp = self.dolls[j]["spec"]
                h = sp["hgt"] * (0.74 if sp["kind"] == "child" else 1.0)
                neck = sp["base"] - h * 0.68
                side = 1.0 if prev[0] >= sp["fx"] else -1.0
                pt = np.array([sp["fx"] + side * h * 0.31, neck - h * 0.02])
                rest = np.array([sp["fx"] - side * h * 0.27, neck + h * 0.26])
                self.dolls[j]["take"] = doll_layer(self._mk_spec(sp, hands=(rest, pt) if side > 0 else (pt, rest)))
            hand_pts[i] = pt
            way.append(pt)
            prev = pt
        # la lana no pasa por encima del aviso: si un tramo lo cruza, lo rodea por debajo
        nc = np.asarray(self.notice_c, np.float64)
        for j in range(len(way) - 1):
            a_, b_ = np.asarray(way[j], np.float64), np.asarray(way[j + 1], np.float64)
            d_ = b_ - a_
            t_ = float(np.clip(np.dot(nc - a_, d_) / max(np.dot(d_, d_), 1e-9), 0, 1))
            if np.hypot(*(a_ + d_ * t_ - nc)) < 60 * u and 0.05 < t_ < 0.95:
                way.insert(j + 1, nc + np.array([0.0, 62 * u]))
                break
        ex = 0.962 * W
        last = way[-1]
        rise = [np.array([(last[0] + ex) / 2, last[1] - 30 * u]), np.array([ex, last[1] - 160 * u]),
                np.array([ex + 4 * u, 0.3 * H]), np.array([ex, 0.08 * H]), np.array([ex - 4 * u, -30 * u])]
        path = _wobbly(way + rise, rng, 6 * u, 3.0 * u)
        self.wool = Yarn(path, 8.5 * u, WOOL, rng, couch_every=19 * u, couch_color="#8e160f", fuzz=1.0)
        def arc_at(p_, after=0.0):                        # largo de lana hasta el punto (después de `after`)
            ok = self.wool.arc >= after
            d_ = np.linalg.norm(self.wool.pts - p_, axis=1) + np.where(ok, 0, 1e9)
            return float(self.wool.arc[int(np.argmin(d_))])
        # un nudo de lana de verdad en cada torre (cada uno de su tamaño, con su cabito suelto)
        self.route_knots = []
        prev_L = 0.0
        for p_ in loops:
            L_ = arc_at(p_, after=prev_L)
            q_ = self.wool.pts[int(np.argmin(np.abs(self.wool.arc - L_)))]
            rk = rng.uniform(9.0, 13.0) * u
            sps = [knot(q_, rk, "#b01d17", rng)]
            ang = rng.uniform(0, 2 * np.pi)
            tail = [q_ + rk * 0.6 * np.array([np.cos(ang), np.sin(ang)]),
                    q_ + rk * 1.9 * np.array([np.cos(ang + 0.3), np.sin(ang + 0.3)]),
                    q_ + rk * 2.8 * np.array([np.cos(ang + 0.1), np.sin(ang + 0.1) + 0.3])]
            sps += Yarn(catmull_rom(np.array(tail), 8), 5.5 * u, WOOL, rng, fuzz=0.8).chunks
            self.route_knots.append((L_, sps))
            prev_L = L_
        L_route = arc_at(last_ring, after=0.5 * arc_at(loops[-1]))
        Lh = []
        for i in order:
            Lh.append(arc_at(hand_pts[i], after=(Lh[-1] if Lh else L_route)))
        self.L_end_houses = Lh[-1]
        a, b = T["trace"]
        w0, w1 = T["wool"]
        r0, r1 = T["rise"]
        fr = {3: (0.43, 0.81, 1.0), 4: (0.34, 0.62, 0.84, 1.0)}[len(order)]      # cada tramo más rápido
        times = [w0 + (w1 - w0) * f_ for f_ in fr]
        self.wool_sched = [(a, 0.0), (b, L_route)] + list(zip(times, Lh)) + [(r0, Lh[-1]), (r1, self.wool.length)]
        self.house_times = {i: t_ for i, t_ in zip(order, times)}
        if own not in self.house_times:
            self.house_times[own] = a
        self.vigia_takes = own in order
        self.vigia_house = own

    def wool_len(self, tq):
        sc = self.wool_sched
        if tq < sc[0][0]:
            return 0.0
        for (t0, l0), (t1, l1) in zip(sc[:-1], sc[1:]):
            if tq < t1:
                return l0 + (l1 - l0) * (tq - t0) / max(1e-6, t1 - t0)
        return sc[-1][1]

    def wool_active(self, tq):
        a, _ = T["trace"]
        _, b = T["wool"]
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
        if x1 - x0 < 3 or y1 - y0 < 3:                  # (en el borde de la tela: no hay dónde hundirla)
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
        """La aguja antes de coser, colgando de su hilo desde fuera del cuadro, cerca de la cámara (grande y
        desenfocada) y meciéndose; en tres imágenes se acerca a la tela, sobre la primera estaca, y ahí se
        corta al primer plano. Devuelve dict(eye, tip, near) en coordenadas de la tela, o None."""
        h0, h1 = T["hook"]
        if tq >= h1 - 1e-6:
            return None
        u = self.u
        k = int(round(tq * UFPS))
        # (en 9:16 cuelga alto, sobre los cerros, entre la casa amarilla y las araucarias: su hilo solo cruza el
        # cielo, lejos del sol, y no parte el cuadro de arriba abajo)
        if self.portrait:
            base_eye = self.first_stitch + np.array([-20 * u, -760 * u])
        else:
            base_eye = self.first_stitch + np.array([150 * u, -330 * u])
        # cambio de foco: primero la arpillera nítida y la aguja desenfocada; antes de bajar, la aguja se
        # enfoca (y la tela se ablanda); al acercarse a la tela, el foco vuelve a la tela
        focus = {7: 0.35, 8: 0.7, 9: 1.0, 10: 1.0, 11: 0.6, 12: 0.3}.get(k, 0.0)
        if tq < h0 - 1e-6:
            sway = np.sin(k / UFPS * 2 * np.pi / 1.4) * 24 * u
            eye = base_eye + np.array([sway, 0])
            ang = np.sin(k / UFPS * 2 * np.pi / 1.4 + 0.9) * 0.07
            tip = eye + np.array([np.sin(ang), np.cos(ang)]) * 300 * u
            return dict(eye=eye, tip=tip, near=1.0, focus=focus)
        i = int(np.floor((tq - h0) * UFPS + 1e-6))       # baja en tres imágenes (escala y giro repartidos)
        f = (0.4, 0.72, 1.0)[min(i, 2)]
        end_tip = self.first_stitch + np.array([-4 * u, 6 * u])
        end_eye = end_tip + _unit(np.array([0.6, -0.8])) * 300 * u
        eye = base_eye + (end_eye - base_eye) * f
        tip = (base_eye + np.array([0, 300 * u])) + (end_tip - (base_eye + np.array([0, 300 * u]))) * f
        return dict(eye=eye, tip=tip, near=float(1 - f) * 0.85, focus=focus)

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
            return out, dg["tip"]                   # la dibuja el mundo, en primer plano
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
        f0, f1 = T["full"]
        c0, c1 = T["curtain"]
        l0, l1 = T["flash"]
        FMIN = 0.05                                    # la sala casi a oscuras: la luz viene de atrás
        if tq < d0:
            return 1.0, 0.0, None
        if tq < d1:                                    # baja la luz de la sala en tres imágenes
            i = min(2, max(0, fidx(tq, d0)))
            return (0.55, 0.3, 0.14)[i], 0.0, None
        if tq < f1:                                    # una lámpara de mano sigue el hilo, huella por huella
            i = max(0, fidx(tq, f0))
            st = self.lamp_plan[min(i, len(self.lamp_plan) - 1)]
            if st is None:                             # y se abre: toda la tela es linterna, la línea completa
                return FMIN, 1.0, "full"
            (cx, cy), R, base, line = st
            return FMIN, 1.0, dict(hand=(cx, cy), R=R, base=base, amp=2.6, line=line)
        if tq < c1:                                    # clic: se apaga la lámpara y la sala queda en penumbra (la
            i = max(0, fidx(tq, c0))                   # luz de la ventana); el tubo prende con un solo titileo suave
            return (0.42, 0.42, 1.0, 0.84)[min(i, 3)], 0.0, None  # (nunca a negro ni destellos: fotosensibilidad)
        if l0 - 1e-6 <= tq < l1:                       # otra vez la lámpara detrás: el revés, ahora vacío
            i = fidx(tq, l0)
            n = fidx(l1 - 1e-3, l0) + 1                # imágenes del destello
            if i <= 0:
                return 0.45, 0.0, None
            if i >= n - 1:
                return 0.85, 0.0, None
            return FMIN, 1.0, "empty"
        return 1.0, 0.0, None

    def pull_state(self, tq):
        """Cuántas puntadas de la torre ya salieron (de la última cosida a la primera). Antes, la vecina toma
        el hilo y se afirma."""
        a, b = T["unstitch"]
        n = len(self.tower_idx)
        if tq < a:
            return 0
        f = np.clip((tq - a) / (b - a - 0.08), 0, 1)
        return int(round(n * f ** 0.85))

    def route_state(self, tq):
        """El hilo de la ruta escondida: (A2, front). Primero se tensa (la tela se frunce por delante a lo
        largo de toda la línea de torres); después se suelta desde la punta lejana hacia el cerro."""
        r0, r1 = T["route"]
        L = float(self.route_arc[-1])
        if tq < r0 - 1e-6:
            return 0.0, None
        i = int(round((tq - r0) * UFPS))
        if i < 3:
            return (0.55, 1.0, 1.0)[i], L + 80 * self.u
        n_rel = max(1, int(round((r1 - r0) * UFPS)) - 3)
        f = (i - 2) / n_rel
        if f >= 1:
            return 0.0, None
        return 1.0, L + 50 * self.u - f * (L + 100 * self.u)

    def route_done(self, tq, arc):
        """¿Ya salió el hilo de la ruta en este punto (largo `arc` desde el cerro)?"""
        if tq >= T["route"][1] - 1e-6:
            return True
        A2, front = self.route_state(tq)
        return front is not None and A2 > 0 and front < arc and tq >= T["route"][0] + 3 / UFPS - 1e-6

    def _tug(self, img, p, toward, k):
        """La tela da un tironcito donde salta la puntada: se estira hacia la mano que tira."""
        u = self.u
        R = 26 * u
        x0, y0 = int(max(0, p[0] - 2 * R)), int(max(0, p[1] - 2 * R))
        x1, y1 = int(min(self.W, p[0] + 2 * R)), int(min(self.H, p[1] + 2 * R))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return img
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = _unit(np.asarray(toward, np.float64) - p)
        g = np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * (R * 0.55) ** 2)).astype(np.float32)
        amp = 3.2 * u * (0.7 + 0.3 * (k % 2))
        reg = np.ascontiguousarray(img[y0:y1, x0:x1])
        img[y0:y1, x0:x1] = cv2.remap(reg, (xx - x0 - d[0] * amp * g).astype(np.float32),
                                      (yy - y0 - d[1] * amp * g).astype(np.float32), cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        img[y0:y1, x0:x1] *= (1 - 0.12 * g)[..., None]
        return img

    def pucker_amount(self, tq):
        a, b = T["taut"]
        if tq < a:
            return 0.0
        if tq < b:
            return float(smoothstep(a + 0.08, b - 0.3, tq))
        # la tela se suelta a tirones (no de a poco): cada tanto salta una puntada que sostenía el frunce
        f = self.pull_state(tq) / max(1, len(self.tower_idx))
        return float(1.0 - 0.8 * np.floor(f * 5 + 1e-6) / 5)

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
        # frunce: la tela (y la silueta: el ruedo sube); lo cosido encima se corre con ella, sin deformarse
        A = self.pucker_amount(tq)
        A2, front = self.route_state(tq)
        alpha = self.world.ea_main.copy() if hasattr(self, "world") else None
        img, Tm, alpha = self._pucker(img, Tm, A, alpha, A2, front)

        def off(c):
            if A <= 0.001:
                return 0, 0
            x_, y_ = int(np.clip(c[0], 0, self.W - 1)), int(np.clip(c[1], 0, self.H - 1))
            return int(round(-A * self.pk["DX"][y_, x_])), int(round(-A * self.pk["DY"][y_, x_]))

        # lo que es una sola pieza rígida se corre entero: el aviso (papel, letras, alfiler) y la cañería
        # con su llave; si cada puntada se corriera por su cuenta, la A del aviso se cruzaría en una X
        anchor = {1: self.notice_c, 2: self.H4}
        off_h = lambda h, c: off(anchor[h]) if h in anchor else off(c)

        if gone:
            for hs_ in self.tower_holes:
                composite(img, hs_, shadow=0)
        for (arc_i, sps) in self.route_holes:          # por donde pasaba la ruta escondida quedan pinchazos
            if self.route_done(tq, arc_i):
                for hs_ in sps:
                    composite(img, hs_, shadow=0)
        for (tt, pc, h) in self.sig_pieces:
            if tq + 1e-6 >= tt:
                dx, dy = off_h(h, pc.center)
                if tq < tt + 1 / UFPS - 1e-6:
                    pc.draw(img, tq)
                else:
                    composite(img, pc.sprite, pc.shadow_k, dx=dx, dy=dy)
                    composite_T(Tm, pc.sprite, dx=dx, dy=dy)
        for (tt, sp, h) in self.extra:
            if tq + 1e-6 >= tt:
                dx, dy = off_h(h, (sp.x0 + sp.a.shape[1] / 2, sp.y0 + sp.a.shape[0] / 2))
                composite(img, sp, shadow=0.5, dx=dx, dy=dy)
                composite_T(Tm, sp, dx=dx, dy=dy)
        for i, it in enumerate(self.hilvan):
            if it[0] <= tq + 1e-6 and i not in removed:
                dx, dy = off_h(it[4], (it[2] + it[3]) / 2)
                composite(img, it[1], shadow=0.55, dx=dx, dy=dy)
                composite_T(Tm, it[1], dx=dx, dy=dy)
        # el cabo del hilo tras la torre (cortado), mientras la torre siga cosida
        if T["taut"][1] - 0.2 <= tq and gone == 0:
            rj = np.random.default_rng(5)
            for sp in self._thread(self.last_stitch, self.last_stitch + np.array([30, 46]) * u, 3.0 * u, BLK, rj,
                                   sag=0.05):
                composite(img, sp, 0.5)
        # lana roja (y un nudo en cada torre de la ruta que ya calcó)
        L = self.wool_len(tq)
        if L > 0:
            self.wool.draw(img, L)
            for (Lk_, sps) in self.route_knots:
                if L >= Lk_ - 1e-6:
                    for sp in sps:
                        composite(img, sp, shadow=0.5)
        # ventanas encendidas
        for i, lit in self.house_lit(tq).items():
            if lit:
                for sp in self.info["houses"][i]["lit"]:
                    composite(img, sp, shadow=0.3)
        # la gente
        self._draw_dolls(img, tq, Tm)
        # la vigía tira del hilván: el hilo suelto va de sus manos a la puntada que sigue; ovillo a sus pies
        p0, p1 = T["pull"]
        if p0 <= tq:                                  # (el ovillo queda a sus pies hasta el final)
            rj = np.random.default_rng(int(tq * 100))
            n = len(self.tower_idx)
            r0_, r1_ = T["route"]
            if gone < n and tq < p1:
                nxt = self.hilvan[self.tower_idx[::-1][gone]][3]
                if gone > 0:
                    img = self._tug(img, nxt, self.pull_hand, k)
                for sp in self._thread(self.pull_hand, nxt, 3.0 * u, BLK, rj,
                                       sag=(0.2 if not gone else (0.03, 0.14)[k % 2])):
                    composite(img, sp, 0.5)
            elif r0_ - 1e-6 <= tq < r1_:
                # sigue tirando: el hilo de la ruta sale por el pie de la torre, tenso, y la tela da tirones
                img = self._tug(img, self.H1, self.pull_hand, k)
                for sp in self._thread(self.pull_hand, self.H1, 3.0 * u, BLK, rj, sag=(0.02, 0.1)[k % 2]):
                    composite(img, sp, 0.5)
            route_f = float(np.clip((tq - r0_) / (r1_ - r0_), 0, 1))
            # el hilo que sale se junta en el suelo, a sus pies: una maraña floja de vueltas sueltas (no una
            # bola), que crece con lo que sale; el hilo baja de sus manos y se pierde entre las vueltas
            amount = 0.25 + 0.45 * min(1.0, gone / max(1, n)) + 0.3 * route_f
            for sp in self._tangle(amount):
                composite(img, sp, 0.35)
                composite_T(Tm, sp)                    # (a contraluz, su silueta)
            if tq < p1:
                end = self.ball_pt + np.array([self._tangle_w * 0.35, -3 * u])
                for sp in self._thread(self.pull_hand, end, 2.6 * u, BLK, rj, sag=0.35):
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
        # foco: donde está la acción; al descoser, entre las manos de la vigía y la torre que se deshace
        pull_focus = (self.pull_hand + np.asarray(self.pylon_top, np.float64) + (0, 60 * u)) / 2
        self.focus = act if act is not None else (pull_focus if p0 <= tq < p1 else None)
        # luz
        F, B, lamp = self.light(tq)
        glow = None
        if B > 0:
            Te = Tm * (1 - 0.99 * self.back_mask)[..., None]
            Te = np.maximum(Te, (self.pinholes * 0.75)[..., None] * (1 - 0.5 * self.back_mask)[..., None])
            if lamp == "empty":
                # el revés vacío: ya no hay hilo escondido; la luz pasa por los pinchazos que dejaron las
                # torres (su silueta en puntitos) y la lana roja queda encima, a contraluz
                Te = Tm * 0.55                           # (la lámpara, más lejos: la tela brilla menos que los pinchazos)
                # la lana es gruesa: casi no deja pasar luz (rojo vino oscuro, que varía con la torsión); solo las
                # fibras sueltas del borde filtran la luz y se encienden, rojas. Llega entera hasta la última mano,
                # como de día; los nudos quedan oscuros
                core_f = np.array([0.25, 0.022, 0.016], np.float32)
                rim_f = np.array([0.85, 0.12, 0.07], np.float32)
                wool_sps = [(ch, True) for ch in self.wool.chunks
                            if float(np.nanmin(np.where(np.isfinite(ch.smap), ch.smap, 1e12))) <= self.L_end_houses]
                for (_, sps) in self.route_knots:
                    wool_sps += [(sp, False) for sp in sps]
                wool_cov = np.zeros((self.H, self.W), np.float32)
                for sp, clip_ in wool_sps:
                    h_, w_ = sp.a.shape
                    x0_, y0_ = max(0, sp.x0), max(0, sp.y0)
                    x1_, y1_ = min(self.W, sp.x0 + w_), min(self.H, sp.y0 + h_)
                    if x1_ <= x0_ or y1_ <= y0_:
                        continue
                    sl_ = (slice(y0_ - sp.y0, y1_ - sp.y0), slice(x0_ - sp.x0, x1_ - sp.x0))
                    a_ = sp.a[sl_]
                    tw = np.ones_like(a_)
                    if sp.smap is not None:
                        sm = np.where(np.isfinite(sp.smap[sl_]), sp.smap[sl_], 0.0)
                        tw = 1 + 0.15 * np.sin(sm / (7 * self.u)) * np.cos(sm / (23 * self.u))
                        if clip_:
                            a_ = a_ * (sp.smap[sl_] <= self.L_end_houses)
                    k_ = smoothstep(0.6, 0.95, a_)[..., None]
                    filt = (rim_f * (1 - k_) + core_f * k_) * tw[..., None]
                    cover = np.clip(2 * a_, 0, 1)[..., None]
                    reg = Te[y0_:y1_, x0_:x1_]
                    reg[:] = reg * ((1 - cover) + cover * filt)
                    wc = wool_cov[y0_:y1_, x0_:x1_]
                    wc[:] = np.maximum(wc, cover[..., 0])
                # (la lana tapa los pinchazos que quedan debajo de ella)
                Te = np.maximum(Te, (np.maximum(self.pinholes, self.route_pins) * (1 - wool_cov))[..., None])
                field, gain = np.float32(1.0), 14.0
            elif isinstance(lamp, str):
                # (con el hilo puesto, por sus puntadas apenas se cuela un brillo al borde del hilo)
                Te = np.maximum(Te, (self.route_pins * 0.3 * (1 - self.back_core_line))[..., None])
                field, gain = np.float32(1.0), 14.0
            elif "hand" in lamp:
                c, R = lamp["hand"], lamp["R"]
                # el haz: un foco tibio de borde suave (centro más fuerte y caída más rápida) sobre la tela que
                # brilla pareja; la mano que sostiene la lámpara le muerde el borde de abajo
                d2 = (self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2
                beam = 0.65 * np.exp(-d2 / (2 * (0.55 * R) ** 2)) + 0.35 * np.exp(-d2 / (2 * R * R))
                field = lamp.get("base", 0.55) + lamp.get("amp", 0.6) * beam[..., None] * np.array([1.0, 0.9, 0.74],
                                                                                                     np.float32)
                if R < 0.2 * max(self.W, self.H):
                    field = field * (1 - 0.72 * self._hand_shadow(c, R, k))[..., None]
                gain = 16.0
                # la sombra blanda del hilo (a un milímetro de la tela) se corre al revés de la luz; y solo se
                # vuelve nítida y negra cerca de la lámpara (lejos, la tela difunde la luz y el hilo se pierde):
                # cada torre escondida aparece cuando la lámpara la encuentra
                sx = -(c[0] - self.W / 2) * 0.006
                sy = -(c[1] - self.H / 2) * 0.006
                line = lamp.get("line", True)
                core = self.back_core if line else self.back_core_sig
                halo0 = self.back_halo if line else self.back_halo_sig
                halo = cv2.warpAffine(halo0, np.float32([[1, 0, sx], [0, 1, sy]]), (self.W, self.H))
                near = np.exp(-((self.xx - c[0]) ** 2 + (self.yy - c[1]) ** 2) / (2 * (R * 0.95) ** 2))
                vis = np.clip(0.05 + 1.5 * near, 0, 1)
                thr = np.clip(core + halo, 0, 1)
                Te = Tm * (1 - 0.99 * thr * vis)[..., None]
                Te = np.maximum(Te, (self.pin_var[k % 3] * 0.75)[..., None] * (1 - 0.5 * thr * vis)[..., None])
                if line:                                # la línea está cosida: junto a sus puntadas se cuela la luz
                    Te = np.maximum(Te, (self.route_pins * 0.45 * vis * vis * (1 - thr))[..., None])
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
            if lamp != "empty":                        # (en el revés vacío, sin halo: la lana no es un neón)
                glow = glow + cv2.GaussianBlur(glow, (0, 0), 7 * u) * 0.2
        return np.clip(img, 0, 1), glow, alpha, (F, B, lamp)

    # ------------------------------------------------------------------------------------------------
    def frame(self, t):
        tq = q(t)
        img, glow, alpha, (F, B, lamp) = self.arpillera(t)
        k = int(round(tq * UFPS))
        return self.world.render_at(img, tq, T, F=F, B=B, lamp=lamp, flick=float(self.flick[k % len(self.flick)]),
                                    wool_tied=tq >= T["rise"][1] - 1e-6, dangle=self.dangle(tq), k=k,
                                    glow=glow, alpha=alpha, focus=getattr(self, "focus", None),
                                    needle_w=11 * self.u)

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
    sub1 = [("Guía para comunidades ante", "small")]
    sub2 = [("megaproyectos de energía", "small")]
    if land:
        main_x = (F(480), F(1440))
        z0 = 1.8
        # (la tira queda bajo el borde del plano general: no asoma en el póster ni en los planos abiertos)
        title = dict(rect=tuple(F(v) for v in (500, 796, 1420, 1008)), font="futuram", font_small="futural",
                     hand=0.02, lines=[
            (F(846), F(38), [("Sistema", "floss"), ("de", "floss"), ("Alerta", "wool"), ("Temprana", "floss")]),
            (F(906), F(38), [("Comunitario", "floss")]),
            (F(961), F(25), [("Guía para comunidades ante megaproyectos de energía", "small")])])
        line = (F(960), F(150), 0.000028 / f)
        nbx = [(F(-150), F(420)), (F(1500), F(2070))]
    else:
        main_x = (F(280), F(800))
        z0 = 1.97
        title = dict(rect=tuple(F(v) for v in (130, 1290, 950, 1676)), font="futuram", font_small="futural",
                     hand=0.02, lines=[
            (F(1368), F(46), [("Sistema", "floss"), ("de", "floss")]),
            (F(1438), F(46), [("Alerta", "wool"), ("Temprana", "floss")]),
            (F(1508), F(46), [("Comunitario", "floss")]),
            (F(1572), F(29), sub1),
            (F(1614), F(29), sub2)])
        line = (F(540), F(222), 0.000026 / f)
        nbx = [(F(-190), F(250)), (F(830), F(1270))]
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

    # un solo primer plano para las tres señales (estacas, aviso, cañería): la cámara no salta de una a
    # otra, se queda y se arrima muy despacio; la aguja se ve ir de una a otra
    sig_pts = np.array(list(sc.info["stakes"]) + [sc.notice_c, sc.H2, sc.H3, sc.H4], np.float64)
    lo, hi = sig_pts.min(axis=0), sig_pts.max(axis=0)
    c_v = (lo + hi) / 2 + np.array([30.0, 40.0]) * sc.u
    span_v = (hi - lo) + np.array([2 * 200.0, 2 * 170.0]) * sc.u
    zv = z0 * float(np.clip(min(W / (span_v[0] * s_img * z0), H / (span_v[1] * s_img * z0)), 1.3, 1.75))
    ph = sc.pull_hand
    c_pu = (ph + sc.H1 - (0, 100 * sc.u)) / 2
    span = np.abs(ph - (sc.H1 - (0, 200 * sc.u))) + 260 * sc.u
    zp = z0 * float(np.clip(min(W / (span[0] * s_img * z0), H / (span[1] * s_img * z0)), 1.0, 1.6))
    c_to = sc.pylon_top + (0, 120 * sc.u)
    zl = z0 * 1.32
    L["shots"] = [(1.17, 1.17, zv, clamp(c_v, zv)),                  # corte al primer plano de las señales
                  (1.3, 2.8, zv * 1.05, clamp(c_v + np.array([24.0, 12.0]) * sc.u, zv * 1.05)),   # se arrima
                  (3.0, 3.0, z0, "c0"),                               # corte al plano general
                  (T["pull"][0] - 0.05, T["pull"][0] - 0.05, zp, clamp(c_pu, zp)),   # corte: la vigía y la torre
                  (T["unstitch"][0], T["unstitch"][0] + 0.4, zp * 1.05, clamp(c_pu, zp * 1.05)),  # se acerca
                  (T["route"][0], T["route"][0], z0, "c0"),           # corte: toda la ruta se frunce y se suelta
                  (T["dolly"][0], T["dolly"][1], 1.0, "final")]
    return L
