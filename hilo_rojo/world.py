"""El mundo alrededor de la arpillera: pared de adobe encalada, el cordel de lana roja, perritos de
ropa, otras arpilleras colgadas (otros territorios) y la tira de tela con el título.

La arpillera cuelga desde el cuadro 0: el cordel y los perritos se ven arriba. Al final la cámara se
aleja por pasos (stop-motion) con paralaje: la pared está más lejos que el cordel, así que se agranda
menos. Todo se compone en raster a la resolución de salida; lo que no cambia se guarda por encuadre.
"""
import cv2
import numpy as np

from satc_intro.color import lin
from satc_intro.geometry import catmull_rom
from satc_intro.noise import fbm, smooth_noise, smoothstep
from .figures import hershey_strokes, hershey_strokes_es, needle_sprite
from .textile import fabric, fray_mask, shade
from .thread import Sprite, Stitch, Yarn, composite, knot, backstitch

LIGHT2 = np.array([-0.64, -0.56])
WOOL = "#c3241c"
HEMP = "#b49b6d"
# profundidades (cm) para el paralaje: la pared, la tira del título, el cordel con las arpilleras; D1 es la
# distancia de la cámara al muro en el encuadre final
Z_WALL, Z_STRIP, Z_ARP, D1 = 0.0, 0.4, 8.0, 40.0


# ------------------------------------------------------------------------------------------------------
# pared
# ------------------------------------------------------------------------------------------------------
def plaster(W, H, seed=5):
    """Pared de adobe encalada: cal irregular, marcas de llana, grietas finas, luz rasante."""
    rng = np.random.default_rng(seed)
    s = max(W, H) / 1920
    h = (0.55 * fbm((H, W), 260 * s, rng, octaves=4) + 0.30 * fbm((H, W), 40 * s, rng, octaves=3)
         + 0.15 * fbm((H, W), 6 * s, rng, octaves=2)).astype(np.float32)
    # pasadas de llana: manchones alargados en direcciones distintas, mezclados por zonas
    acc = np.zeros((H // 4, W // 4), np.float32)
    wsum = np.zeros_like(acc)
    for ang in (18, -34, 63, -72):
        n = rng.standard_normal((H // 4, W // 4)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=10 * s, sigmaY=2.2 * s)
        M = cv2.getRotationMatrix2D((W / 8, H / 8), ang + rng.normal(0, 6), 1.0)
        n = cv2.warpAffine(n, M, (W // 4, H // 4), borderMode=cv2.BORDER_REFLECT)
        wz = smooth_noise((H // 4, W // 4), 60 * s, rng) ** 3
        acc += n / (n.std() + 1e-6) * wz
        wsum += wz
    n = cv2.resize(acc / (wsum + 1e-3), (W, H), interpolation=cv2.INTER_CUBIC)
    h += 0.018 * n
    gy, gx = np.gradient(cv2.GaussianBlur(h, (0, 0), 1.2 * s))
    lam = 1 + 6.5 * s * (-gx * LIGHT2[0] - gy * LIGHT2[1])
    base = lin("#e3d8c4")
    tone = 1 + 0.08 * (fbm((H, W), 500 * s, rng, octaves=3) - 0.5)
    col = base[None, None, :] * (tone * np.clip(lam, 0.75, 1.25))[..., None]
    cr = np.zeros((H, W), np.float32)
    for _ in range(9):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        a = rng.uniform(0, 2 * np.pi)
        for k in range(int(rng.uniform(40, 140))):
            a += rng.normal(0, 0.35)
            nx, ny = x + np.cos(a) * 4 * s, y + np.sin(a) * 4 * s
            cv2.line(cr, (int(x * 4), int(y * 4)), (int(nx * 4), int(ny * 4)), 1.0, 1, cv2.LINE_AA, shift=2)
            x, y = nx, ny
    col *= (1 - 0.28 * cv2.GaussianBlur(cr, (0, 0), 0.6 * s))[..., None]
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    damp = smoothstep(0.80, 1.0, yy + 0.08 * (fbm((H, W), 120 * s, rng, octaves=3) - 0.5))
    col *= (1 - 0.12 * damp)[..., None]
    return np.clip(col, 0, 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------------
# perrito de ropa
# ------------------------------------------------------------------------------------------------------
def clothespin(p, ang, L, seed):
    """Perrito de ropa de madera visto de canto: dos patas redondeadas, la ranura donde muerde el
    cordel y el resorte de alambre. p = punto donde muerde; ang = giro del eje (0 = hacia abajo)."""
    rng = np.random.default_rng(seed)
    wood = lin("#d9b07a") * (0.9 + 0.15 * rng.random())
    pad = int(L * 0.8) + 6
    x0, y0 = int(p[0] - pad), int(p[1] - pad)
    n = 2 * pad
    ss = 3
    yy, xx = (np.mgrid[0:n * ss, 0:n * ss].astype(np.float32) + 0.5) / ss
    xx += x0
    yy += y0
    qx, qy = xx - p[0], yy - p[1]
    d = np.array([np.sin(ang), np.cos(ang)])            # eje (hacia abajo)
    lx = qx * d[0] + qy * d[1]
    ly = -qx * d[1] + qy * d[0]
    top, bot = -0.36 * L, 0.64 * L
    t = np.clip((lx - top) / (bot - top), 0, 1)
    gap = L * (0.010 + 0.028 * np.clip((0.30 - t) / 0.30, 0, 1) ** 1.5
               + 0.022 * np.exp(-((t - 0.70) / 0.05) ** 2))
    hw = L * 0.078 * (1 - 0.30 * np.clip((t - 0.62) / 0.38, 0, 1))
    hl = (bot - top) / 2
    mid = (top + bot) / 2
    L3 = np.array([-0.64, -0.56, 0.42])
    L3 = L3 / np.linalg.norm(L3)
    # veta: perfil al azar a lo ancho, ondulado a lo largo
    prof = np.convolve(rng.standard_normal(96), np.ones(3) / 3, mode="same")
    a = np.zeros(xx.shape, np.float32)
    lamv = np.zeros(xx.shape, np.float32)
    grainv = np.zeros(xx.shape, np.float32)
    for side in (-1, 1):
        c = side * (gap + hw)
        rc = hw * 0.8
        qa = np.abs(lx - mid) - (hl - rc)
        qb = np.abs(ly - c) - (hw - rc)
        sdf = np.hypot(np.maximum(qa, 0), np.maximum(qb, 0)) + np.minimum(np.maximum(qa, qb), 0) - rc
        m = np.clip(-sdf * ss + 0.5, 0, 1)
        e = np.clip((ly - c) / hw, -1, 1)
        ea = np.clip(np.maximum(qa, 0) / rc, 0, 1) * np.sign(lx - mid)
        # normal: redondeada a lo ancho y en las puntas
        nl = e * 0.85
        na = ea * 0.8
        nz = np.sqrt(np.clip(1 - nl * nl - na * na, 0.05, 1))
        nxs = nl * (-d[1]) + na * d[0]
        nys = nl * d[0] + na * d[1]
        lam = np.clip(nxs * L3[0] + nys * L3[1] + nz * L3[2], 0, 1)
        g = np.interp((ly - c) / (hw * 2) * 40 + 48 + 2.5 * np.sin(lx / L * 5 + side), np.arange(96), prof)
        take = m > a
        lamv = np.where(take, lam, lamv)
        grainv = np.where(take, g, grainv)
        a = np.maximum(a, m)
    shade_ = 0.30 + 0.95 * lamv
    rgb = wood[None, None, :] * (shade_ * (1 + 0.07 * grainv))[..., None]
    rgb *= (1 - 0.18 * np.clip((np.abs(lx - mid) - hl * 0.86) / (hl * 0.14), 0, 1))[..., None]   # testa
    # resorte de alambre (a un tercio de arriba)
    ts = 0.34
    ls = top + ts * (bot - top)
    band = np.clip((0.055 * L - np.abs(lx - ls)) * ss + 0.5, 0, 1) * np.clip((L * 0.21 - np.abs(ly)) * ss + 0.5, 0, 1)
    ph = (lx - ls) / (0.018 * L) * np.pi
    coil = 0.5 + 0.5 * np.cos(ph)
    ey = np.clip(ly / (L * 0.21), -1, 1)
    sl = np.clip(-ey * 0.55 * 0.9 + np.sqrt(np.clip(1 - ey * ey, 0, 1)) * 0.5, 0, 1)
    steel = lin("#b2b6ba")
    sm = band * (0.35 + 0.65 * coil)
    met = steel[None, None, :] * (0.25 + 1.1 * sl * coil + 0.6 * (sl * coil) ** 8)[..., None]
    rgb = rgb * (1 - sm[..., None]) + met * sm[..., None]
    a = np.maximum(a, sm)
    rgb = cv2.resize(rgb * a[..., None], (n, n), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)
    sh = cv2.GaussianBlur(a, (0, 0), L * 0.045)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, L * 0.09], [0, 1, L * 0.08]]), (n, n))
    return Sprite(x0, y0, rgb.astype(np.float32), a.astype(np.float32), sh.astype(np.float32))


# ------------------------------------------------------------------------------------------------------
# el mundo
# ------------------------------------------------------------------------------------------------------
class Hanging:
    """Una arpillera colgada del cordel con dos perritos. Coordenadas de mundo (encuadre final)."""

    def __init__(self, size, x0, x1, line_y, grip=3.0, pin_in=0.07, sag=0.012, folds=5.0, seed=0):
        iw, ih = size
        self.iw, self.ih = iw, ih
        w = x1 - x0
        self.s = w / iw                                  # px de mundo por px de imagen
        self.pins = [x0 + pin_in * w, x1 - pin_in * w]
        pa = np.array([self.pins[0], line_y(self.pins[0])])
        pb = np.array([self.pins[1], line_y(self.pins[1])])
        self.pin_pts = [pa, pb]
        self.theta = float(np.arctan2(pb[1] - pa[1], pb[0] - pa[0]))
        c, s_ = np.cos(self.theta), np.sin(self.theta)
        self.R = np.array([[c, -s_], [s_, c]])
        # origen (esquina sup. izq. de la imagen) en mundo
        self.O = pa + self.R @ np.array([-pin_in * w, grip])
        self.sag = sag
        self.folds = folds
        self.seed = seed

    def img_to_world(self, p):
        return self.O + self.R @ (np.asarray(p, np.float64) * self.s)

    def corners_world(self):
        return np.array([self.img_to_world(p) for p in [(0, 0), (self.iw, 0), (self.iw, self.ih), (0, self.ih)]])


def edge_alpha(iw, ih, seed=3):
    """Borde de saco: un poco irregular, con hilachas."""
    rng = np.random.default_rng(seed)
    m = np.ones((ih, iw), np.float32)
    b = int(3 * max(iw, ih) / 1920) + 2
    m[:b, :] = 0
    m[-b:, :] = 0
    m[:, :b] = 0
    m[:, -b:] = 0
    k = max(iw, ih) / 1920
    dx = (smooth_noise((ih, iw), 6 * k, rng) - 0.5) * 7 * k
    dy = (smooth_noise((ih, iw), 6 * k, rng) - 0.5) * 7 * k
    yy, xx = np.mgrid[0:ih, 0:iw].astype(np.float32)
    m = cv2.remap(m, xx + dx, yy + dy, cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m, (0, 0), 0.7)


class World:
    def __init__(self, W, H, layout, seed=5):
        """layout: dict(
             line=(x_mid, y_mid, k)  cordel: y = y_mid - k (x - x_mid)^2 (sube hacia los postes)
             main=dict(size=(iw, ih), x=(x0, x1))
             neighbors=[dict(img=rgb_lin, x=(x0, x1), exit, wool_w)]
             title=dict(rect, font, lines=[(base, alto, [(palabra, tipo)])])
             wool_exit=[(x, y guarda), (x, 0)] en la imagen principal; wool_w
             z0, c0            encuadre inicial (acercamiento y centro del plano del cordel, mundo)
        )"""
        self.W, self.H = W, H
        self.L = layout
        self.u = min(W, H) / 1080
        xm, ym, k = layout["line"]
        self.line_y = lambda x: ym - k * (x - xm) ** 2
        self.main = Hanging(layout["main"]["size"], *layout["main"]["x"], self.line_y, seed=1)
        self.nbs = []
        for i, nb in enumerate(layout["neighbors"]):
            ih, iw = nb["img"].shape[:2]
            hg = Hanging((iw, ih), *nb["x"], self.line_y, seed=10 + i, folds=4.0 + i)
            hg.img = nb["img"].astype(np.float32)
            hg.exit = nb.get("exit")
            hg.wool_w = nb.get("wool_w", 7.0)
            hg.red = nb.get("red")                    # su lana roja (Yarn, de arriba a la puerta)
            hg.lit = nb.get("lit", [])                # sus ventanas encendidas
            self.nbs.append(hg)
        self.ea_main = edge_alpha(*layout["main"]["size"])
        for hg in self.nbs:
            hg.ea = edge_alpha(hg.iw, hg.ih, seed=hg.seed)
        self.wall = plaster(W * 2, H * 2, seed=seed)
        self.title = self._title_strip(layout["title"])
        self.pin_L = 56 * self.u
        self.line_w = 6.4 * self.u
        # el cordel: pasa por todos los perritos, tenso entre los de una misma arpillera
        pins = []
        for hg in self.nbs + [self.main]:
            pins += [(p[0], p[1], id(hg)) for p in hg.pin_pts]
        pins.sort()
        pts = [(-0.6 * W, self.line_y(-0.6 * W))]
        for i, (x, y, owner) in enumerate(pins):
            pts.append((x, y - 1.5 * self.u))
            if i + 1 < len(pins):
                x2, y2, o2 = pins[i + 1]
                span = x2 - x
                sag = (0.0015 if o2 == owner else 0.035) * span
                pts.append(((x + x2) / 2, (y + y2) / 2 + sag))
        pts.append((1.6 * W, self.line_y(1.6 * W)))
        self.line_pts = catmull_rom(np.array(pts), 24)
        self._cache = {}
        # el frente rojo parte del nudo de la lana de la arpillera principal
        self.knot_x = None
        if layout.get("wool_exit") is not None:
            self.knot_x = float(self.main.img_to_world(layout["wool_exit"][1])[0])
        self.front_v = 1700 * self.u                   # ~140 px por imagen
        # tomas: los centros pueden venir en coordenadas de la arpillera principal ("img") o de mundo
        shots = []
        for sh in layout.get("shots", []):
            t0, t1, zt, ct = sh
            if isinstance(ct, tuple) and len(ct) == 2 and ct[0] == "img":
                ct = self.main.img_to_world(ct[1])
            elif isinstance(ct, str) and ct == "c0":
                ct = layout["c0"]
            elif isinstance(ct, str) and ct == "final":
                ct = (W / 2, H / 2)
            shots.append((t0, t1, zt, ct))
        layout["shots"] = shots
        self.main_u = min(layout["main"]["size"]) / 1080
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = xx / W - 0.5, yy / H - 0.5
        self.vign = (1 - 0.30 * (nx * nx * 1.1 + ny * ny * 1.3) ** 1.1).astype(np.float32)
        self.rake = (1 + 0.14 * (-(nx * 0.85 + ny * 0.5))).astype(np.float32)
        self.xx, self.yy = xx, yy
        # grano de película: 6 texturas que se turnan (cada imagen única tiene el suyo)
        rg = np.random.default_rng(seed + 7)
        self.grain = []
        for _ in range(6):
            gr = rg.standard_normal((H, W)).astype(np.float32)
            gr = cv2.GaussianBlur(gr, (0, 0), 0.7 * self.u)
            self.grain.append(gr / (gr.std() + 1e-6))
        # control: en el plano general del comienzo (póster) la tira del título no asoma ni un píxel
        vis = self._title_visible(self.camera(0.0))
        if vis > 0.004:
            raise ValueError(f"la tira del título asoma en el plano general (alfa máx. {vis:.3f}): bajarla")

    def _title_visible(self, cam):
        """Cuánto se ve de la tira del título (y sus tachuelas y su sombra) dentro del cuadro con esta cámara."""
        W, H = self.W, self.H
        zs = cam["zs"]
        c = np.asarray(cam["c"], np.float64)
        Ms = np.float32([[zs / 2, 0, W / 2 - c[0] * zs], [0, zs / 2, H / 2 - c[1] * zs]])
        acc = np.zeros((H, W), np.float32)
        for sp in self.title:
            for a_ in (sp.a,) + ((sp.sh,) if sp.sh is not None else ()):
                h, w = a_.shape
                M = Ms.copy()
                M[0, 2] = Ms[0, 2] + Ms[0, 0] * sp.x0
                M[1, 2] = Ms[1, 2] + Ms[1, 1] * sp.y0
                acc = np.maximum(acc, cv2.warpAffine(a_.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR))
        return float(acc.max())

    # ------------------------------------------------------------------------------------------------
    # cámara: se mueve por pasos desparejos, como en un rodaje cuadro a cuadro
    def _dist(self, z):
        return Z_ARP + (D1 - Z_ARP) / z

    def _build_shots(self):
        """Tomas: [(t0, t1, z, c)] en mundo. Cada movimiento va por pasos con tamaños desparejos, alguna
        imagen retenida, temblor de mano y un pequeño sobrepaso al asentarse."""
        rs = np.random.default_rng(23)
        self._shots = []
        for (t0, t1, zt, ct) in self.L.get("shots", []):
            if t1 <= t0 + 1e-6:                       # corte seco (y una imagen en que la cámara se asienta)
                j = rs.normal(0, 1.8, 2)
                self._shots.append((t0, t1, float(zt), np.asarray(ct, np.float64), np.array([1.0, 1.0]),
                                    np.array([j, [0.0, 0.0]])))
                continue
            n = max(2, int(round((t1 - t0) * 12)))
            base = np.diff(smoothstep(0, 1, np.linspace(0, 1, n + 1)))
            steps = base * rs.uniform(0.8, 1.2, n)
            if n >= 8:
                for hold in (int(n * 0.3), int(n * 0.68)):
                    steps[hold] = 0
            e = np.concatenate([[0], np.cumsum(steps)])
            e /= e[-1]
            e = np.concatenate([e[1:], [1.016, 1.005]])
            jits = rs.normal(0, 1.0, (len(e), 2))
            jits[-3:] *= 0.4
            self._shots.append((t0, t1, float(zt), np.asarray(ct, np.float64), e, jits))

    def camera(self, tq, T=None):
        if not hasattr(self, "_shots"):
            self._build_shots()
        z = float(self.L["z0"])
        c = np.asarray(self.L["c0"], np.float64)
        jit = (0.0, 0.0)
        for (t0, t1, zt, ct, e_, jits) in self._shots:
            if tq < t0 - 1e-6:
                break
            i = int(np.floor((tq - t0) * 12 + 1e-6))
            if i >= len(e_):
                z, c = zt, ct
                continue
            e = float(e_[i])
            d = self._dist(z) + (self._dist(zt) - self._dist(z)) * e
            c = c + (ct - c) * e
            z = (D1 - Z_ARP) / (d - Z_ARP)
            jit = tuple(jits[i])
            break
        d = self._dist(z)
        return dict(z=z, zw=D1 / d, zs=(D1 - Z_STRIP) / (d - Z_STRIP), c=c, jit=jit)

    @staticmethod
    def _title_recall(refs, emb_a, off, u2):
        """Control de legibilidad del bordado: por palabra, qué fracción del trazo de la letra (rasterizado
        con la misma fuente) queda cubierta por el hilo (dilatado 2 px de salida). Una letra que perdió su
        punto, su tilde o su travesaño baja la cifra."""
        cov = cv2.dilate((emb_a > 0.25).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                      (int(4 * u2) | 1,) * 2))
        out = {}
        for st, wid in refs:                     # cada trazo por separado: un punto perdido también cuenta
            m = np.zeros(emb_a.shape, np.uint8)
            q_ = np.round((np.asarray(st) - np.array(off)) * 8).astype(np.int32)
            if len(q_) == 1:
                q_ = np.vstack([q_, q_])
            cv2.polylines(m, [q_], False, 1, 1, cv2.LINE_8, shift=3)
            n = int(m.sum())
            r = int((m & cov).sum()) / max(n, 1)
            out[wid] = min(out.get(wid, 1.0), r)
        return [round(r, 3) for _, r in sorted(out.items())]

    # ------------------------------------------------------------------------------------------------
    def _title_strip(self, spec):
        """Tira de tocuyo con el título bordado (punto atrás; ALERTA en la misma lana roja), clavada a la
        pared con dos tachuelas, levemente torcida y combada. Se arma a 2x para que aguante el alejamiento."""
        rng = np.random.default_rng(77)
        u2 = 2 * self.u
        x0, y0, x1, y1 = [2 * v for v in spec["rect"]]
        pad = 50
        shape = (int(y1 - y0) + 2 * pad, int(x1 - x0) + 2 * pad)
        ox, oy = x0 - pad, y0 - pad
        fab, _ = fabric(shape, rng, "#efe5cd", kind="plain", scale=u2)
        m = fray_mask([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], shape, rng, offset=(ox, oy), fray=2.2)
        pf = shade(cv2.GaussianBlur(m, (0, 0), 5) * 3.2, strength=1.0, ambient=0.75)
        pf = pf / max(1e-6, float(np.percentile(pf[m > 0.9], 60)))
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        wave = 1 + 0.045 * np.sin(xx / shape[1] * np.pi * 3.0 + 0.8) * (yy / shape[0])
        rgb = fab * (np.clip(pf, 0.6, 1.2) * wave)[..., None] * m[..., None]
        sh = cv2.warpAffine(cv2.GaussianBlur(m, (0, 0), 8), np.float32([[1, 0, 12], [0, 1, 12]]), (shape[1], shape[0]))
        strip = Sprite(int(ox), int(oy), rgb.astype(np.float32), m.astype(np.float32), sh)
        items = []
        font = spec.get("font", "timesi")
        cx = (x0 + x1) / 2
        dark = spec.get("color", "#2b211b")
        # medir todas las líneas y achicar lo necesario para que quepan dentro de la tira (con margen)
        inner = (x1 - x0) * 0.88
        font_small, font_wool = spec.get("font_small", "futural"), spec.get("font_wool", font)
        fnt = lambda kind: font_small if kind == "small" else (font_wool if kind == "wool" else font)
        hand = float(spec.get("hand", 0.0))                # pulso de mano: los trazos no son de regla

        def trace(words, cap, base, k=1.0):
            ws = [(*hershey_strokes_es(txt, fnt(kind), cap * 2 * k, 0, base * 2, anchor="left", ref="H"), kind)
                  for (txt, kind) in words]
            gaps = [cap * 2 * k * (0.8 if "wool" in (ws[i][2], ws[i + 1][2]) else 0.64) for i in range(len(ws) - 1)]
            return ws, gaps, sum(w for _, w, _ in ws) + sum(gaps)

        widest = max(trace(words, cap, base)[2] for (base, cap, words) in spec["lines"])
        fit = min(1.0, inner / widest)
        tail = []
        refs = []                                  # (trazo, palabra): para el control de legibilidad
        word_id = 0
        wool_start = None
        for (base, cap, words) in spec["lines"]:
            ws, gaps, total = trace(words, cap, base, fit)
            space_list = gaps + [0.0]
            x = cx - total / 2
            x_line0 = x
            ph_line = rng.uniform(0, 2 * np.pi)
            for (strokes, w, kind), space in zip(ws, space_list):
                if kind == "wool":
                    xs_ = [float(np.mean(np.asarray(st_)[:, 0])) for st_ in strokes]
                    cut_x = max(xs_) - 0.16 * w if xs_ else 1e9      # la última letra
                # bordado a mano: la línea base ondula a lo largo del renglón y cada letra tiene su giro,
                # su tamaño y su altura (dos «a» nunca salen iguales); el punto y la tilde van con su letra
                sts = [np.asarray(st_, np.float64) + np.array([x, 0.0]) for st_ in strokes]
                gid = _glyph_groups(sts, 0.1 * cap * 2 * fit)
                gtf = {}
                for g in sorted(set(gid)):
                    pts_g = np.vstack([sts[i_] for i_ in range(len(sts)) if gid[i_] == g])
                    c_ = pts_g.mean(axis=0)
                    th_ = np.deg2rad(rng.normal(0, 1.4 if kind != "small" else 1.0))
                    sc_ = 1 + rng.normal(0, 0.03 if kind != "small" else 0.02)
                    R_ = np.array([[np.cos(th_), -np.sin(th_)], [np.sin(th_), np.cos(th_)]]) * sc_
                    wave = 1.5 * u2 * np.sin(2 * np.pi * (c_[0] - x_line0) / max(total * 1.15, 1.0) + ph_line)
                    gtf[g] = (R_, c_, wave + rng.normal(0, 0.9 * u2))
                for i_s, st in enumerate(sts):
                    R_, c_, dy_ = gtf[gid[i_s]]
                    st = (st - c_) @ R_.T + c_ + np.array([0, dy_])
                    if hand > 0 and len(st) >= 2:
                        st = _hand_wobble(st, hand * cap * 2 * fit, rng)
                    if kind == "wool":
                        if len(st) < 2:
                            continue
                        i_ = int(np.argmin(st[:, 0]))
                        if wool_start is None or st[i_, 0] < wool_start[0]:
                            wool_start = st[i_].copy()
                        if float(np.mean(st[:, 0])) - x > cut_x:
                            tail.append(st)                      # se borda al final, con la aguja
                            continue
                        y = Yarn(st, 5.4 * u2, WOOL, rng, fuzz=0.45, step=1.0)
                        items += y.chunks
                        refs.append((st, word_id))
                        continue
                    refs.append((st, word_id))
                    capx = cap * 2 * fit
                    size = float(max(np.ptp(st[:, 0]), np.ptp(st[:, 1])))
                    plen = float(np.sum(np.linalg.norm(np.diff(st, axis=0), axis=1))) if len(st) > 1 else 0.0
                    chord = float(np.hypot(*(st[-1] - st[0])))
                    col_ = "#34281f" if kind == "small" else dark
                    if len(st) > 2 and size < 0.3 * capx and chord < 0.35 * max(size, 1e-6):
                        # un punto (la i): un nudito francés, del tamaño del punto de la letra
                        items.append(knot(st.mean(axis=0), max((1.35 if kind == "small" else 2.0) * u2, 0.42 * size),
                                          col_, rng))
                    elif plen < 0.5 * capx and chord > 0.85 * plen:
                        # tilde o travesaño: una sola puntada tensa (nunca se pierde)
                        a_, b_ = st[0], st[-1]
                        if chord < 3 * u2:
                            d_ = (b_ - a_) / max(chord, 1e-6)
                            a_, b_ = (a_ + b_) / 2 - d_ * 1.5 * u2, (a_ + b_) / 2 + d_ * 1.5 * u2
                        items.append(Stitch.render(a_, b_, (2.0 if kind == "small" else 2.6) * u2, col_, rng))
                    elif kind == "small":
                        for _, sp in backstitch(st, 3.4 * u2, 2.3 * u2, col_, rng, jitter=0.1, wvar=0.3):
                            items.append(sp)
                    else:
                        for _, sp in backstitch(st, 6.2 * u2, 2.5 * u2, dark, rng, jitter=0.2, wvar=0.25):
                            items.append(sp)
                x += w + space
                word_id += 1
        # todo junto en una capa (se tuerce y se comba como una sola tela); el bordado, solo dentro de la tela
        layer_rgb = strip.rgb.copy()
        layer_a = strip.a.copy()
        emb_rgb = np.zeros_like(layer_rgb)
        emb_a = np.zeros_like(layer_a)
        for sp in items:
            _premult_over_local(emb_rgb, emb_a, sp, int(ox), int(oy))
        inside = cv2.erode((strip.a > 0.5).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(np.float32)
        inside = cv2.GaussianBlur(inside, (0, 0), 1.0)
        emb_rgb *= inside[..., None]
        emb_a *= inside
        # la última letra de «Alerta» está dibujada a lápiz (así se marca un bordado antes de coserlo)
        if tail:
            pm = np.zeros(layer_a.shape, np.float32)
            for st in tail:
                q_ = np.round((np.asarray(st) - np.array([ox, oy])) * 8).astype(np.int32)
                cv2.polylines(pm, [q_], False, 1.0, max(1, int(round(2.8 * u2))), cv2.LINE_AA, shift=3)
            grain = 0.62 + 0.38 * np.random.default_rng(78).random(pm.shape).astype(np.float32)
            pm = cv2.GaussianBlur(pm, (0, 0), 0.5) * grain * strip.a
            layer_rgb = layer_rgb * (1 - 0.85 * pm)[..., None] + lin("#4d4a47")[None, None, :] * (0.85 * pm)[..., None]
        self.title_check = self._title_recall(refs, emb_a, (ox, oy), u2)
        layer_rgb = layer_rgb * (1 - emb_a)[..., None] + emb_rgb
        layer_a = np.maximum(layer_a, emb_a)
        hgt_, wid_ = layer_a.shape
        ang = np.deg2rad(-0.8)
        M = cv2.getRotationMatrix2D((wid_ / 2, pad), float(np.rad2deg(ang)), 1.0)
        yy, xx = np.mgrid[0:hgt_, 0:wid_].astype(np.float32)
        sagm = (1 - ((xx - wid_ / 2) / (wid_ / 2)) ** 2) * (yy / hgt_) * 6 * u2
        layer_rgb = cv2.remap(layer_rgb, xx, (yy - sagm).astype(np.float32), cv2.INTER_LINEAR)
        layer_a = cv2.remap(layer_a, xx, (yy - sagm).astype(np.float32), cv2.INTER_LINEAR)
        layer_rgb = cv2.warpAffine(layer_rgb, M, (wid_, hgt_), flags=cv2.INTER_LINEAR)
        layer_a = cv2.warpAffine(layer_a, M, (wid_, hgt_), flags=cv2.INTER_LINEAR)
        shd = cv2.warpAffine(cv2.GaussianBlur(layer_a, (0, 0), 9), np.float32([[1, 0, 13], [0, 1, 13]]), (wid_, hgt_))
        out = [Sprite(int(ox), int(oy), layer_rgb, layer_a, shd)]

        def to_wall2(pts):                          # la misma comba y el mismo giro de la tira
            q_ = np.asarray(pts, np.float64) - np.array([ox, oy])
            sg = (1 - ((q_[:, 0] - wid_ / 2) / (wid_ / 2)) ** 2) * (q_[:, 1] / hgt_) * 6 * u2
            q_ = np.stack([q_[:, 0], q_[:, 1] + sg, np.ones(len(q_))], 1) @ M.T
            return q_ + np.array([ox, oy])

        bad = [r for r in self.title_check if r < 0.7]
        if bad:
            raise ValueError(f"el título bordado no se lee bien: cobertura por palabra {self.title_check}")
        self.title_tail = [to_wall2(st) for st in tail]
        self.title_tail_w = 5.4 * u2
        self.title_wool_start = to_wall2(np.array([wool_start]))[0] if wool_start is not None else None
        for (nx_, ny_) in ((x0 + 26, y0 + 22), (x1 - 26, y0 + 18)):
            out.append(knot((nx_, ny_), 7 * self.u, "#4a4440", rng))
        return out

    # ------------------------------------------------------------------------------------------------
    def _plane(self, z, c):
        """Transformación del plano del cordel (mundo -> pantalla)."""
        return np.array([[z, 0, self.W / 2 - c[0] * z], [0, z, self.H / 2 - c[1] * z]], np.float64)

    def _warp_hanging(self, hg, z, c):
        """Mapas pantalla -> imagen de una arpillera colgada: comba entre perritos, pliegues que bajan
        desde los perritos, bordes que ondulan, una esquina levantada, lomas suaves con luz rasante."""
        W, H = self.W, self.H
        P = self._plane(z, c)
        cw = hg.corners_world()
        sc = cw @ P[:, :2].T + P[:, 2]
        pad = 10 + int(hg.ih * hg.s * z * hg.sag * 1.6)
        bx0, by0 = int(np.floor(sc[:, 0].min())) - pad, int(np.floor(sc[:, 1].min())) - pad
        bx1, by1 = int(np.ceil(sc[:, 0].max())) + pad, int(np.ceil(sc[:, 1].max())) + pad
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
        if bx1 - bx0 < 3 or by1 - by0 < 3:                 # fuera de cuadro (o apenas una hebra)
            return None
        xs = np.arange(bx0, bx1, dtype=np.float64) + 0.5
        ys = np.arange(by0, by1, dtype=np.float64) + 0.5
        X, Y = np.meshgrid(xs, ys)
        wx = (X - P[0, 2]) / z
        wy = (Y - P[1, 2]) / z
        qx, qy = wx - hg.O[0], wy - hg.O[1]
        Ri = hg.R.T
        ix = (Ri[0, 0] * qx + Ri[0, 1] * qy) / hg.s
        iy = (Ri[1, 0] * qx + Ri[1, 1] * qy) / hg.s
        u_ = np.clip(ix / hg.iw, 0, 1)
        v_ = np.clip(iy / hg.ih, 0, 1)
        k_img = 1.0 / hg.s                                  # px de imagen por px de mundo
        rh = np.random.default_rng(100 + hg.seed)
        # lomas suaves
        hfield = np.zeros_like(u_)
        for _ in range(4):
            fu, fv = rh.uniform(0.6, 2.4), rh.uniform(0.4, 1.8)
            amp = rh.uniform(0.6, 1.0)
            hfield += amp * np.sin(2 * np.pi * (fu * u_ + fv * v_) + rh.uniform(0, 2 * np.pi))
        # dos pliegues diagonales que bajan desde los perritos hacia el centro
        pin_u = 0.07
        for side in (-1, 1):
            u_p = 0.5 + side * (0.5 - pin_u)
            ang = np.deg2rad(rh.uniform(28, 38))
            du = (u_ - u_p) * hg.iw
            dv = v_ * hg.ih
            along = -side * du * np.sin(ang) + dv * np.cos(ang)
            across = side * du * np.cos(ang) + dv * np.sin(ang)
            reach = hg.ih * 0.42
            fold = np.exp(-(across / (hg.ih * 0.045)) ** 2) * np.clip(along / (hg.ih * 0.05), 0, 1) \
                * np.exp(-np.maximum(along, 0) / reach)
            hfield += 2.2 * fold
        gy, gx = np.gradient(hfield)
        lg = -(0.64 * gx + 0.56 * gy)                      # luz rasante desde arriba a la izquierda
        cloth = 1 + 0.065 * lg / (float(np.std(lg)) + 1e-9)
        # bordes que ondulan: 2–3 ondas abajo (±4 px) y la esquina de abajo a la derecha algo levantada
        wv = 4.0 * k_img * np.sin(2 * np.pi * (2.4 + hg.seed * 0.1) * u_ + rh.uniform(0, 6)) * v_ ** 3
        corner = 6.0 * k_img * (np.clip((u_ - 0.8) / 0.2, 0, 1) * np.clip((v_ - 0.8) / 0.2, 0, 1)) ** 1.5
        ix = ix + 1.8 * k_img * np.sin(2 * np.pi * (1.3 * v_ + 0.7 * u_) + hg.seed) + corner * 0.7
        iy = iy + wv + corner
        cloth = cloth * (1 + 0.08 * corner / (6.0 * k_img + 1e-6))
        # comba entre perritos: el borde de arriba cuelga en el medio
        sag = hg.ih * hg.sag * (1 - ((u_ - 0.5) / 0.43) ** 2).clip(0, 1) * (1 - v_) ** 2
        iy = iy - sag
        k = hg.s * z
        f = 1.0
        if k < 0.8:
            f = k * 1.15
        folds = (1 + 0.04 * np.sin(u_ * np.pi * hg.folds + 0.6 + hg.seed) * (1 - v_) ** 1.5 * np.sin(u_ * np.pi)) \
            * np.clip(cloth, 0.8, 1.2)
        mx = (ix * f - 0.5 + 0.5 * f) if f != 1 else (ix - 0.5)
        my = (iy * f - 0.5 + 0.5 * f) if f != 1 else (iy - 0.5)
        return dict(box=(bx0, by0, bx1, by1), mx=mx.astype(np.float32), my=my.astype(np.float32), f=f,
                    folds=folds.astype(np.float32))

    @staticmethod
    def _sample(img, wp):
        if wp["f"] != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, int(round(w * wp["f"]))), max(1, int(round(h * wp["f"])))),
                             interpolation=cv2.INTER_AREA)
        return cv2.remap(img, wp["mx"], wp["my"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _shadow_into(self, out, a_full, cam):
        """Sombra de las telas colgadas en la pared: está 8 cm más atrás, así que se ve con la escala de la
        pared (se desliza respecto de la tela al alejarse) y corrida por la luz rasante."""
        z, zw = cam["z"], cam["zw"]
        u = self.u * zw
        r = zw / z
        M = np.float32([[r, 0, (1 - r) * self.W / 2 + 12 * u], [0, r, (1 - r) * self.H / 2 + 11 * u]])
        sh = cv2.warpAffine(a_full, M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, 7.5 * u))
        out *= (1 - 0.42 * sh)[..., None]

    def _wall_bg(self, cam):
        W, H = self.W, self.H
        zw, zs = cam["zw"], cam["zs"]
        c = np.asarray(cam["c"], np.float64)
        Mw = np.float32([[zw / 2, 0, W / 2 - c[0] * zw], [0, zw / 2, H / 2 - c[1] * zw]])
        bg = cv2.warpAffine(self.wall, Mw, (W, H), flags=cv2.INTER_AREA if zw < 2 else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        Ms = np.float32([[zs / 2, 0, W / 2 - c[0] * zs], [0, zs / 2, H / 2 - c[1] * zs]])
        self._draw_sprites(bg, self.title, Ms, shadow=0.38)
        self._Ms = Ms
        # foco: con la cámara cerca, la pared (8 cm detrás de las telas) queda algo desenfocada
        sig = 2.2 * max(0.0, cam["z"] - 1.0) * self.u
        if sig > 0.3:
            bg = cv2.GaussianBlur(bg, (0, 0), sig)
        return bg

    def _layers(self, cam):
        """Pared y cada tela colgada como capa de pantalla completa (se guardan por encuadre)."""
        key = ("lay", round(cam["z"], 5), round(cam["c"][0], 2), round(cam["c"][1], 2))
        if key in self._cache:
            return self._cache[key]
        W, H = self.W, self.H
        z, c = cam["z"], cam["c"]
        bg = self._wall_bg(cam)
        lays = []
        for hg in self.nbs:
            wp = self._warp_hanging(hg, z, c)
            if wp is None:
                continue
            x0, y0, x1, y1 = wp["box"]
            rgbF = np.zeros((H, W, 3), np.float32)
            aF = np.zeros((H, W), np.float32)
            aF[y0:y1, x0:x1] = self._sample(hg.ea, wp)
            rgbF[y0:y1, x0:x1] = self._sample(hg.img, wp) * wp["folds"][..., None]
            lays.append((hg, rgbF, aF, wp))
        wpm = self._warp_hanging(self.main, z, c)
        amF = np.zeros((H, W), np.float32)
        if wpm is not None:
            x0, y0, x1, y1 = wpm["box"]
            amF[y0:y1, x0:x1] = self._sample(self.ea_main, wpm)
        st = dict(bg=bg, lays=lays, wpm=wpm, amF=amF, P=self._plane(z, c))
        if len(self._cache) > 4:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = st
        return st

    # ------------------------------------------------------------------------------------------------
    # el frente rojo: la lana se enrosca en el cordel desde el nudo, de arpillera en arpillera
    def front_radius(self, tau):
        if self.knot_x is None or tau is None or tau < 0:
            return 0.0
        return self.front_v * tau

    def front_done(self, tau):
        far = max(abs(self.knot_x), abs(self.W - self.knot_x)) + 200 * self.u
        return tau is not None and self.front_v * tau > far

    def kink_dy(self, xw, tau):
        """Quiebre que viaja con el frente (y se apaga al salir del cuadro)."""
        xw = np.asarray(xw, np.float64)
        if tau is None or tau < 0:
            return np.zeros_like(xw)
        dd = np.abs(xw - self.knot_x)
        r = self.front_radius(tau)
        w = 48 * self.u
        return -24 * self.u * np.exp(-((dd - r) / w) ** 2) * np.exp(-dd / (2600 * self.u))

    def nb_progress(self, hg, tau):
        """Cuando el rojo llega por el cordel al nudo de una vecina: baja por su lana (up), entra en su tela
        hasta la puerta (inside) y se enciende la ventana (lit)."""
        out = dict(up=0.0, inside=0.0, lit=False, t_arrive=None)
        if tau is None or tau < 0 or hg.exit is None or self.knot_x is None:
            return out
        xk = float(hg.img_to_world(hg.exit[1])[0])
        t_arr = abs(xk - self.knot_x) / self.front_v
        out["t_arrive"] = t_arr
        dt = tau - t_arr
        if dt <= 0:
            return out
        out["up"] = float(np.clip(dt / 0.17, 0, 1))
        out["inside"] = float(np.clip((dt - 0.17) / 0.5, 0, 1))
        out["lit"] = dt >= 0.17 + 0.5
        return out

    def swing(self, hg, tq, tau):
        """Ángulo de péndulo de una tela cuando le llega el frente (±2,5°, se apaga en ~7 imágenes)."""
        if tau is None or tau < 0:
            return 0.0
        dmin = min(abs(p[0] - self.knot_x) for p in hg.pin_pts)
        t_hit = dmin / self.front_v
        dt = tau - t_hit
        if dt < 0:
            return 0.0
        amp = np.deg2rad(0.9 if hg is self.main else 2.6)
        return float(amp * np.exp(-dt / 0.3) * np.sin(2 * np.pi * dt / 0.55))

    def _pose(self, hg, P, tau, tq):
        """Afín en pantalla: la tela sigue a sus perritos (el quiebre) y se mece como péndulo."""
        pa, pb = hg.pin_pts
        da, db = self.kink_dy(np.array([pa[0], pb[0]]), tau)
        pa2, pb2 = pa + np.array([0, da]), pb + np.array([0, db])
        th0 = np.arctan2(pb[1] - pa[1], pb[0] - pa[0])
        th1 = np.arctan2(pb2[1] - pa2[1], pb2[0] - pa2[0])
        dth = float(th1 - th0) + self.swing(hg, tq, tau)
        S = lambda q_: np.asarray(q_) @ P[:, :2].T + P[:, 2]
        m0, m1 = S((pa + pb) / 2), S((pa2 + pb2) / 2)
        c_, s_ = np.cos(dth), np.sin(dth)
        R = np.array([[c_, -s_], [s_, c_]])
        t = m1 - R @ m0
        return np.float32([[c_, -s_, t[0]], [s_, c_, t[1]]]), dth, (pa2, pb2)

    def _cord(self, fg_rgb, fg_a, P, z, tau, red_all):
        """El cordel: cáñamo; rojo donde ya pasó el frente (o todo, si red_all)."""
        W, H = self.W, self.H
        u = self.u * z
        lw_ = self.line_pts.copy()
        lw_[:, 1] += self.kink_dy(lw_[:, 0], tau) if tau is not None else 0
        line = lw_ @ P[:, :2].T + P[:, 2]
        vis = (line[:, 0] > -80) & (line[:, 0] < W + 80) & (line[:, 1] > -80) & (line[:, 1] < H + 80)
        if not vis.any():
            return
        i0 = max(0, int(np.argmax(vis)) - 2)
        i1 = min(len(line), len(vis) - int(np.argmax(vis[::-1])) + 2)
        seg, segw = line[i0:i1], lw_[i0:i1]
        if len(seg) < 3:
            return
        if red_all:
            red = np.ones(len(seg), bool)
        else:
            r = self.front_radius(tau) if tau is not None else 0.0
            red = np.abs(segw[:, 0] - self.knot_x) < r
        # tramos continuos de un mismo color
        k0 = 0
        for k in range(1, len(seg) + 1):
            if k == len(seg) or red[k] != red[k0]:
                part = seg[max(0, k0 - 1):min(len(seg), k + 1)]
                if len(part) >= 2:
                    col = WOOL if red[k0] else HEMP
                    wdt = self.line_w * z * (1.9 if red[k0] else 1.0)     # la lana roja, gruesa: se ve
                    y = Yarn(part, wdt, col, np.random.default_rng(7 + k0), fuzz=0.9 if red[k0] else 1.2,
                             step=max(1.0, 1.6 * u))
                    for ch in y.chunks:
                        _premult_over(fg_rgb, fg_a, ch, shadow=0.45)
                k0 = k

    def _needle_fg(self, out, dg, P, z, amF, needle_w):
        """La aguja colgando cerca de la cámara: más grande, desenfocada y con la sombra lejana y blanda en la
        tela; al acercarse a la tela (near → 0) se achica, se enfoca y la sombra se le junta."""
        W, H = self.W, self.H
        near = float(dg["near"])
        S = lambda q_: np.asarray(q_) @ P[:, :2].T + P[:, 2]
        e_s = S(self.main.img_to_world(dg["eye"]))
        t_s = S(self.main.img_to_world(dg["tip"]))
        m = 1 + 1.0 * near                               # cerca de la cámara: el doble de grande
        c_s = (e_s + t_s) / 2
        c_s2 = c_s + (c_s - np.array([W / 2, H / 2])) * 0.12 * near
        e2 = c_s2 + (e_s - c_s) * m
        t2 = c_s2 + (t_s - c_s) * m
        w = needle_w * self.main.s * z * m
        lift = 0.9 + 3.6 * near
        rf = float(dg.get("focus", 0.0))
        sig = 1.5 * near * (1 - rf) * self.u * z / 1.8           # semienfocada desde el primer cuadro
        if rf > 0.01:                                     # el foco se va a la aguja: la tela se ablanda
            out[:] = cv2.GaussianBlur(out, (0, 0), 2.6 * rf * self.u)
        # el hilo, desde fuera del cuadro, pasa por el ojo; un cabo corto cuelga del otro lado
        eye_c = e2 + (t2 - e2) * 0.07
        top = np.array([e2[0] + 8.0, -80.0])
        pts = catmull_rom(np.array([eye_c, (eye_c + top) / 2 + (4.0, 0), top]), 12)
        ww = 4.6 * self.main_u * z * self.main.s * m
        yarn = Yarn(pts, ww, "#141516", np.random.default_rng(3), fuzz=0, kind="floss")
        Ln_ = float(np.hypot(*(t2 - e2)))
        sd = 1.0 if (t2 - e2)[0] <= 0 else -1.0
        tail = Yarn(catmull_rom(np.array([eye_c, eye_c + np.array([sd * 0.05 * Ln_, 0.14 * Ln_]),
                                          eye_c + np.array([sd * 0.07 * Ln_, 0.3 * Ln_])]), 10),
                    ww, "#141516", np.random.default_rng(4), fuzz=0, kind="floss")
        shm = np.zeros((H, W), np.float32)
        for ch in yarn.chunks:
            _alpha_into(shm, ch)
        nd = needle_sprite(e2, t2, w, lift=lift)
        u_ = self.u * z
        off = np.array([10, 14]) * u_ * (1 + 2.2 * near)
        shm = cv2.GaussianBlur(shm, (0, 0), (3.0 + 6 * near) * u_)
        shm = cv2.warpAffine(shm, np.float32([[1, 0, off[0]], [0, 1, off[1]]]), (W, H))
        out *= (1 - (0.28 - 0.1 * near) * shm * amF)[..., None]
        composite(out, Sprite(nd.x0, nd.y0, np.zeros_like(nd.rgb), np.zeros_like(nd.a), nd.sh), shadow=0.72 - 0.25 * near)
        for ch in yarn.chunks:
            composite(out, _blur_sprite(Sprite(ch.x0, ch.y0, ch.rgb, ch.a, None), sig), 0.0)
        composite(out, _blur_sprite(Sprite(nd.x0, nd.y0, nd.rgb, nd.a, None), sig), 0.0)
        for ch in tail.chunks:
            composite(out, _blur_sprite(Sprite(ch.x0, ch.y0, ch.rgb, ch.a, None), sig), 0.0)

    def _red_link(self, P, head, u_):
        """El camino de la lana que une la etiqueta con la red: nace en el nudo del cordel (donde se anudó la
        lana de la arpillera), baja por el hueco entre las telas y llega, por encima de la tira, a la «a»."""
        if self.knot_x is None:
            return None
        xk = float(self.knot_x)
        yk = float(np.interp(xk, self.line_pts[:, 0], self.line_pts[:, 1]))
        kn = np.array([xk, yk]) @ P[:, :2].T + P[:, 2]
        cw = self.main.corners_world() @ P[:, :2].T + P[:, 2]
        right = float(np.max(cw[:, 0]))
        bottom = float(np.max(cw[:, 1]))
        gx = right + 16 * u_
        p1 = np.array([gx, kn[1] + 40 * u_])
        p2 = np.array([gx - 4 * u_, bottom + 34 * u_])
        p3 = np.array([head[0] + 20 * u_, head[1] - 70 * u_])
        return catmull_rom(np.array([kn, p1, (p1 + p2) / 2 + np.array([5 * u_, 0]), p2,
                                     (p2 + p3) / 2 + np.array([0, 16 * u_]), p3, head]), 14)

    def _end_stitch(self, out, tq, win, cam, k, P=None):
        """La aguja del principio vuelve, ahora con la lana roja, y borda sobre el trazo a lápiz la última
        letra de «Alerta»; después pasa al revés de la tira y queda estacionada en el margen, clavada como en
        un costurero, con su cabo colgando (cierra el círculo con el primer cuadro). La hebra de trabajo es
        corta: nunca cruza el texto terminado."""
        a, b = win
        Ms = self._Ms
        S = lambda q_: np.asarray(q_, np.float64) @ Ms[:, :2].T + Ms[:, 2]
        pts = [S(st) for st in self.title_tail]
        L = [float(np.sum(np.linalg.norm(np.diff(p_, axis=0), axis=1))) for p_ in pts]
        tot = sum(L)
        t_in = a + 4 / 12                                 # baja por su hebra desde el cordel en cuatro imágenes
        f = float(np.clip((tq - t_in) / (b - t_in), 0, 1))
        done = f * tot
        w = self.title_tail_w * Ms[0, 0]
        rng = np.random.default_rng(55)
        u_ = self.u * cam["zs"]
        # la hebra roja que une la etiqueta con el nudo de la red, en el cordel (el título es parte de la red)
        link = self._red_link(P, pts[0][0], u_) if P is not None else None
        link_upto = None
        if link is not None:
            arc_l = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(link, axis=0), axis=1))])
            if tq < t_in:
                i_ = int(np.floor((tq - a) * 12 + 1e-6))
                link_upto = arc_l[-1] * (i_ + 1) / 4.5
                seg_ = link[arc_l <= link_upto]
            else:
                seg_ = link
            if len(seg_) >= 2:
                for ch in Yarn(seg_, w * 0.9, WOOL, np.random.default_rng(56), fuzz=0.6).chunks:
                    composite(out, ch, 0.35)
        head, head_d = None, np.array([1.0, 0.0])
        for p_, l_ in zip(pts, L):
            if done <= 0:
                break
            frac = min(1.0, done / max(l_, 1e-6))
            q_ = _cut_path(p_, frac)
            if len(q_) >= 2:
                for ch in Yarn(q_, w, WOOL, rng, fuzz=0.45, step=1.0).chunks:
                    composite(out, ch, 0.35)
                head = q_[-1]
                head_d = q_[-1] - q_[-2]
            done -= l_
        if head is None:
            head = pts[0][0]
        kk = int(round(tq * 12))
        Ln, wn = 150 * u_, 5.4 * u_
        th = -np.deg2rad(58 + (kk % 3) * 3)              # el ojo arriba a la derecha: tapa lo menos posible
        v = np.array([np.cos(th), np.sin(th)])
        eye_of = lambda e_, t_: e_ + (t_ - e_) * 0.07
        ww = w * 0.9
        if tq >= b and self.title_wool_start is not None:
            # estacionada: pasó al revés (la hebra va por detrás) y asoma en el margen izquierdo
            ws = S(self.title_wool_start)
            g2 = int(np.floor((tq - b) * 12 + 1e-6))
            if g2 < 1:                                        # un cuadro: se hunde al revés al final de la a
                tip, eye = head - v * 0.35 * Ln, head + v * 0.6 * Ln
                nd = needle_sprite(eye, tip, wn, hide_from=0.62, lift=0.6)
                e_pt = eye_of(eye, tip)
                for ch in Yarn(catmull_rom(np.array([head, (head + e_pt) / 2 + np.array([0, 6 * u_]), e_pt]), 10),
                               ww, WOOL, rng, fuzz=0.6).chunks:
                    composite(out, ch, 0.35)
                composite(out, nd, 0.6)
                return
            tip = ws + np.array([-66, 18]) * u_
            d_ = _unit2(np.array([-0.42, -0.9]))
            eye = tip + d_ * Ln * 0.95
            nd = needle_sprite(eye, tip, wn, hide_from=0.86, lift=0.8)
            sw = np.sin((tq - b) * 2 * np.pi / 1.7) * 2.5 * u_
            e_pt = eye_of(eye, tip)
            # la hebra: del ojo baja en un bucle y entra a la tela junto a la aguja; el cabo cuelga del ojo
            enter = tip - d_ * 0.12 * Ln + np.array([9, -4]) * u_
            loop = catmull_rom(np.array([e_pt, e_pt + np.array([16 + sw, 34]) * u_,
                                         enter + np.array([6, -14]) * u_, enter]), 10)
            tail = catmull_rom(np.array([e_pt, e_pt + np.array([-7 + sw, 16]) * u_,
                                         e_pt + np.array([-10 + 1.5 * sw, 30]) * u_]), 8)
            for pth in (loop, tail):
                for ch in Yarn(pth, ww, WOOL, rng, fuzz=0.6).chunks:
                    composite(out, ch, 0.35)
            composite(out, nd, 0.6)
            return
        if tq < t_in and link is not None:            # baja por su hebra, desde el nudo del cordel
            j_ = int(np.searchsorted(arc_l, link_upto))
            j_ = int(np.clip(j_, 1, len(link) - 1))
            tip = link[j_]
            d_ = _unit2(link[j_] - link[max(0, j_ - 3)])
            eye = tip - d_ * Ln                            # el ojo va detrás de la punta
            inside = False
            nd = needle_sprite(eye, tip, wn, lift=0.9)
            composite(out, nd, 0.6)
            return
        if tq < t_in:                  # (sin cordel a la vista: entra por la pared, entre la arpillera y la tira)
            i_ = int(np.floor((tq - a) * 12 + 1e-6))
            g = (i_ + 1) / 4.5
            rect = self.L["title"]["rect"]
            sl = S(np.array([2 * rect[0], 2 * rect[1]]))
            start = np.array([sl[0] - 260 * u_, sl[1] - 40 * u_])
            goal = head + np.array([-10, -22]) * u_
            tip = start + (goal - start) * g
            eye = tip - _unit2(goal - start) * Ln          # el ojo va detrás de la punta
            inside = False
        else:
            inside = (kk % 2 == 0) and f < 1
            if inside:
                tip, eye = head - v * 0.15 * Ln, head + v * 0.7 * Ln
            else:
                tip = head + _unit2(head_d) * 12 * u_ + np.array([0, -4 * u_])
                eye = tip + v * Ln
        nd = needle_sprite(eye, tip, wn, hide_from=0.82 if inside else None, lift=0.8)
        e_pt = eye_of(eye, tip)
        # hebra de trabajo corta: de la última puntada al ojo, y un cabo que cuelga del ojo
        mid = (head + e_pt) / 2 + np.array([0, 10 * u_])
        for pth in (catmull_rom(np.array([head, mid, e_pt]), 10),
                    catmull_rom(np.array([e_pt, e_pt + np.array([-5, 14]) * u_, e_pt + np.array([-6, 26]) * u_]), 8)):
            for ch in Yarn(pth, ww, WOOL, rng, fuzz=0.6).chunks:
                composite(out, ch, 0.35)
        composite(out, nd, 0.6)

    def _front_wraps(self, fg_rgb, fg_a, P, z, tau):
        """La punta del frente rojo: vueltas de lana enrolladas en espiral sobre el cordel."""
        if tau is None or self.knot_x is None:
            return
        r = self.front_radius(tau)
        u = self.u * z
        rng = np.random.default_rng(int(tau * 1000))
        for sgn in (-1, 1):
            xf = self.knot_x + sgn * r
            for j in range(5):
                xw = xf - sgn * (j * 8.5 + 3) * self.u
                if abs(xw - self.knot_x) > r:
                    continue
                yw = self.line_y(xw) + float(self.kink_dy(np.array([xw]), tau)[0]) - 1.5 * self.u
                p = np.array([xw, yw]) @ P[:, :2].T + P[:, 2]
                if not (-40 < p[0] < self.W + 40 and -40 < p[1] < self.H + 40):
                    continue
                d = np.array([2.8, 5.2]) * u * (1 if j % 2 else 0.9)
                sp = Stitch.render(p - d, p + d, 3.4 * u, WOOL, rng, kind="wool", hole=False, bow=0)
                _premult_over(fg_rgb, fg_a, sp, shadow=0.4)

    def _wool_up(self, rgb, a, hg, ex, P, z, w_img, M=None, tau=None, frac=1.0):
        """La lana que sale por el borde de arriba de una arpillera sube al cordel y se anuda.
        ex = [(x, y) en la guarda, (x, 0) en el borde] (coordenadas de la imagen). frac: cuánto se ve, desde
        el cordel hacia abajo (la alerta que baja)."""
        if frac <= 0:
            return
        p_in = hg.img_to_world(ex[0])
        p_edge = hg.img_to_world(ex[1])
        d = p_edge - p_in
        xt = p_edge[0] + d[0] * 0.25
        top = np.array([xt, self.line_y(xt) - 1.0 * self.u + float(self.kink_dy(np.array([xt]), tau)[0])])
        S = lambda q_: np.asarray(q_) @ P[:, :2].T + P[:, 2]
        a_s, b_s = S(p_in), S(p_edge)
        if M is not None:
            a_s = M[:, :2] @ a_s + M[:, 2]
            b_s = M[:, :2] @ b_s + M[:, 2]
        pts = catmull_rom(np.array([S(top), b_s, a_s]), 10)
        if frac < 1:
            pts = _cut_path(pts, frac)
        w = w_img * hg.s * z
        rng = np.random.default_rng(int(abs(ex[0][0])) + 3)
        if len(pts) >= 2:
            y = Yarn(pts, w, WOOL, rng, fuzz=0.9)
            for ch in y.chunks:
                _premult_over(rgb, a, ch, shadow=0.45)
        kp = S(top)
        r = max(w, self.line_w * z) * 0.66
        for dx, dy, k in ((-0.55, 0.15, 0.95), (0.6, 0.1, 1.0), (0.0, -0.1, 1.1)):
            _premult_over(rgb, a, knot(kp + np.array([dx, dy]) * r * 1.2, r * k, "#b41f18", rng), shadow=0.45)

    def _draw_sprites(self, out, sprites, M, shadow=0.4):
        """Dibuja sprites definidos a 2x sobre la pared, transformados por M."""
        W, H = self.W, self.H
        for sp in sprites:
            h, w = sp.a.shape
            corners = np.array([[sp.x0, sp.y0], [sp.x0 + w, sp.y0 + h]], np.float64)
            sc = corners @ M[:, :2].T + M[:, 2]
            x0, y0 = int(np.floor(sc[0, 0])) - 2, int(np.floor(sc[0, 1])) - 2
            x1, y1 = int(np.ceil(sc[1, 0])) + 2, int(np.ceil(sc[1, 1])) + 2
            if x1 < 0 or y1 < 0 or x0 > W or y0 > H or x1 - x0 < 2:
                continue
            Ml = M.copy()
            Ml[0, 2] = M[0, 2] + M[0, 0] * sp.x0 - x0
            Ml[1, 2] = M[1, 2] + M[1, 1] * sp.y0 - y0
            size = (x1 - x0, y1 - y0)
            fl = cv2.INTER_AREA if M[0, 0] < 1 else cv2.INTER_LINEAR
            rgb = cv2.warpAffine(sp.rgb, Ml, size, flags=fl)
            a = cv2.warpAffine(sp.a, Ml, size, flags=fl)
            sh = cv2.warpAffine(sp.sh, Ml, size, flags=fl) if sp.sh is not None else None
            composite(out, Sprite(x0, y0, rgb, a, sh), shadow=shadow)

    # ------------------------------------------------------------------------------------------------
    def _F_field(self, F):
        """Luz de frente en pantalla: un número, o la cortina que se abre desde la izquierda."""
        if not isinstance(F, dict):
            return float(F)
        W = self.W
        xe = -0.3 * W + 1.6 * W * F["curtain"]
        lit = smoothstep(xe, xe - 0.35 * W, self.xx[0])
        f = F["fmin"] + (1 - F["fmin"]) * lit
        return np.broadcast_to(f[None, :, None], (self.H, W, 1)).astype(np.float32)

    def render_at(self, main_lin, tq, T, F=1.0, B=0.0, lamp=None, flick=0.0, wool_tied=False, dangle=None, k=0,
                  glow=None, alpha=None, focus=None, needle_w=11.0):
        """Compone el cuadro completo en el instante tq. main_lin: la arpillera con luz de frente plena;
        glow: su resplandor a contraluz (o None); alpha: su silueta (con el frunce) o None."""
        W, H = self.W, self.H
        cam = self.camera(tq)
        z, c = cam["z"], cam["c"]
        st = self._layers(cam)
        P = st["P"]
        Ff = self._F_field(F)
        tau = (tq - T["front"][0]) if tq >= T["front"][0] else None
        dynamic = tau is not None and (not self.front_done(tau) or tau < 1.6)
        red_all = tau is not None and self.front_done(tau)
        out = st["bg"] * Ff
        # la arpillera principal (y las vecinas) en su pose
        mrgb = np.zeros((H, W, 3), np.float32)
        mglow = None
        amF = st["amF"]
        wpm = st["wpm"]
        if wpm is not None:
            x0, y0, x1, y1 = wpm["box"]
            mrgb[y0:y1, x0:x1] = self._sample(main_lin, wpm) * wpm["folds"][..., None]
            if glow is not None:
                mglow = np.zeros((H, W, 3), np.float32)
                mglow[y0:y1, x0:x1] = self._sample(glow, wpm)
            if alpha is not None:
                amF = np.zeros((H, W), np.float32)
                amF[y0:y1, x0:x1] = self._sample(alpha, wpm)
        if not dynamic:
            rb = np.random.default_rng(7000 + k)
            pm = np.mean([hg_ @ P[:, :2].T + P[:, 2] for hg_ in self.main.pin_pts], axis=0)
            Mb = cv2.getRotationMatrix2D((float(pm[0]), float(pm[1])), float(rb.normal(0, 0.03)), 1.0)
            Mb[:, 2] += rb.normal(0, 0.3, 2)
            mrgb = cv2.warpAffine(mrgb, Mb, (W, H), flags=cv2.INTER_LINEAR)
            amF = cv2.warpAffine(amF, Mb, (W, H), flags=cv2.INTER_LINEAR)
            if mglow is not None:
                mglow = cv2.warpAffine(mglow, Mb, (W, H), flags=cv2.INTER_LINEAR)
        items = []
        for (hg, r_, a_, wp_) in st["lays"]:
            pr = self.nb_progress(hg, tau)
            if pr["inside"] > 0 or pr["lit"]:
                key = (id(st), round(pr["inside"], 3), pr["lit"])
                cache = getattr(hg, "_redcache", None)
                if cache is not None and cache[0] == key:
                    r_ = cache[1]
                else:
                    # la lana roja baja por la arpillera vecina y enciende su ventana
                    img2 = hg.img.copy()
                    if hg.red is not None:
                        hg.red.draw(img2, hg.red.length * pr["inside"])
                    if pr["lit"]:
                        for sp in hg.lit:
                            composite(img2, sp, shadow=0.3)
                    x0, y0, x1, y1 = wp_["box"]
                    r_ = np.zeros_like(r_)
                    r_[y0:y1, x0:x1] = self._sample(img2, wp_) * wp_["folds"][..., None]
                    hg._redcache = (key, r_)
            items.append((hg, r_, a_))
        items.append((self.main, mrgb, amF))
        placed, poses = [], {}
        for hg, rgbF, aF in items:
            if dynamic:
                M, dth, pins2 = self._pose(hg, P, tau, tq)
                poses[id(hg)] = (M, dth, pins2)
                placed.append((cv2.warpAffine(rgbF, M, (W, H), flags=cv2.INTER_LINEAR),
                               cv2.warpAffine(aF, M, (W, H), flags=cv2.INTER_LINEAR)))
            else:
                poses[id(hg)] = (None, 0.0, hg.pin_pts)
                placed.append((rgbF, aF))
        a_all = np.zeros((H, W), np.float32)
        for _, a2 in placed:
            a_all = np.maximum(a_all, a2)
        self._shadow_into(out, a_all, cam)
        # la luz de atrás se escapa por los bordes y lava la pared
        if B > 0 and lamp is not None:
            if isinstance(lamp, str):
                fld = amF
            elif "hand" in lamp:
                lw = self.main.img_to_world(lamp["hand"])
                lsx, lsy = lw @ P[:, :2].T + P[:, 2]
                R = lamp["R"] * self.main.s * z
                fld = amF * (lamp.get("base", 0.5) + np.exp(-((self.xx - lsx) ** 2 + (self.yy - lsy) ** 2) / (2 * R * R)))
            elif "band" in lamp:
                lw = self.main.img_to_world((lamp["band"] * self.main.iw, self.main.ih / 2))
                lsx = float((lw @ P[:, :2].T + P[:, 2])[0])
                fld = amF * np.exp(-((self.xx - lsx) / (0.2 * self.main.iw * self.main.s * z)) ** 2)
            else:
                lw = self.main.img_to_world(lamp["c"])
                lsx, lsy = lw @ P[:, :2].T + P[:, 2]
                R = lamp["R"] * self.main.s * z
                fld = amF * np.exp(-((self.xx - lsx) ** 2 + (self.yy - lsy) ** 2) / (2 * R * R))
            spill = cv2.GaussianBlur(fld, (0, 0), 38 * self.u * z) * (1 - amF)
            lampc = np.array([1.0, 0.84, 0.6], np.float32)
            out += (B * 1.5 * spill)[..., None] * lampc * st["bg"]
        for r2, a2 in placed:
            out *= (1 - a2)[..., None]
            out += r2 * Ff * a2[..., None] if not isinstance(Ff, float) else r2 * Ff * a2[..., None]
        if mglow is not None:
            out += mglow * amF[..., None]
        # delante: cordel, lanas anudadas y perritos
        fg_rgb = np.zeros((H, W, 3), np.float32)
        fg_a = np.zeros((H, W), np.float32)
        self._cord(fg_rgb, fg_a, P, z, tau if dynamic else None, red_all)
        if dynamic and not red_all:
            self._front_wraps(fg_rgb, fg_a, P, z, tau)
        for hg in self.nbs + [self.main]:
            if hg is self.main:
                if not wool_tied:
                    continue
                ex, w_img = self.L.get("wool_exit"), self.L.get("wool_w", 8.5)
                frac = 1.0
            else:
                ex, w_img = hg.exit, hg.wool_w
                frac = 1.0 if red_all and not dynamic else self.nb_progress(hg, tau)["up"]
            if ex is not None:
                self._wool_up(fg_rgb, fg_a, hg, ex, P, z, w_img, M=poses.get(id(hg), (None,))[0],
                              tau=tau if dynamic else None, frac=frac)
        for hg in self.nbs + [self.main]:
            M, dth, pins2 = poses.get(id(hg), (None, 0.0, hg.pin_pts))
            for i, pw in enumerate(pins2):
                sp = pw @ P[:, :2].T + P[:, 2]
                if -200 < sp[0] < W + 200 and -200 < sp[1] < H + 200:
                    ang = -(hg.theta + dth) + np.deg2rad((-3, 4)[i] + 2 * np.sin(hg.pin_pts[i][0]))
                    spr = clothespin(sp, ang, self.pin_L * z, int(hg.pin_pts[i][0] * 7) % 1000)
                    _premult_over(fg_rgb, fg_a, spr, shadow=0.5)
        out *= (1 - fg_a)[..., None]
        out += fg_rgb * Ff
        # antes de coser: la aguja cuelga de su hilo cerca de la cámara (grande, desenfocada, con su sombra
        # lejana en la tela) y se acerca a la tela
        if dangle is not None:
            self._needle_fg(out, dangle, P, z, amF, needle_w)
        if T.get("end_stitch") and tq >= T["end_stitch"][0] and getattr(self, "title_tail", None):
            self._end_stitch(out, tq, T["end_stitch"], cam, k, P)
        zr = z / float(self.L["z0"])
        if zr > 1.15:
            yf = H / 2
            if focus is not None:
                yf = float((self.main.img_to_world(focus) @ P[:, :2].T + P[:, 2])[1])
            dm = smoothstep(0.14 * H, 0.52 * H, np.abs(self.yy[:, :1] - yf)) * float(np.clip((zr - 1.15) / 0.4, 0, 1))
            if dm.max() > 0.01:
                bl = cv2.GaussianBlur(out, (0, 0), 3.4 * self.u)
                dm = dm[..., None]
                out = out * (1 - dm) + bl * dm
            # de cerca, la cámara mira la tela un poco desde arriba: la parte de arriba está más cerca (se
            # agranda) y el foco cae hacia arriba y hacia abajo como en una lente, no como un filtro
            kp = 0.028 * float(np.clip((zr - 1.15) / 0.4, 0, 1))
            if kp > 0.002:
                src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
                dst = np.float32([[-W * kp, 0], [W * (1 + kp), 0], [W, H], [0, H]])
                out = cv2.warpPerspective(out, cv2.getPerspectiveTransform(src, dst), (W, H), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT)
        # desenfoque de movimiento (obturador abierto mientras la cámara se desplaza)
        prev = self.camera(tq - 1 / 12)
        if abs(prev["z"] - z) < 0.02 * z:
            dpx = (np.asarray(prev["c"]) - np.asarray(c)) * z
            mag = float(np.hypot(*dpx))
            if mag > 40:
                n_ = int(min(12, mag * 0.5 / 6)) + 2
                acc = np.zeros_like(out)
                for i_ in range(n_):
                    f_ = (i_ / (n_ - 1) - 0.5) * 0.5
                    acc += cv2.warpAffine(out, np.float32([[1, 0, dpx[0] * f_], [0, 1, dpx[1] * f_]]), (W, H),
                                          borderMode=cv2.BORDER_REFLECT)
                out = acc / n_
        # luz de la sala (una ventana a la izquierda), viñeta, parpadeo, grano y el leve bamboleo de película
        out *= (self.rake * self.vign * (1 + flick))[..., None]
        out *= (1 + 0.012 * self.grain[k % len(self.grain)])[..., None]
        rj = np.random.default_rng(4000 + k)
        jx, jy = rj.normal(0, 0.3, 2) + np.asarray(cam["jit"]) * 1.0
        if abs(jx) + abs(jy) > 0.02:
            out = cv2.warpAffine(out, np.float32([[1, 0, jx], [0, 1, jy]]), (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        return np.clip(out, 0, 1)


def _hand_wobble(st, amp, rng):
    """Un trazo hecho a pulso: se remuestrea y se ondula suavemente (dos ondas largas al azar), sin
    mover sus extremos más de lo que se mueve el resto."""
    st = np.asarray(st, np.float64)
    seg = np.linalg.norm(np.diff(st, axis=0), axis=1)
    L = float(seg.sum())
    if L < 1e-6:
        return st
    n = max(2, int(L / 2.0) + 1)
    s_ = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, L, n)
    pts = np.stack([np.interp(t, s_, st[:, 0]), np.interp(t, s_, st[:, 1])], 1)
    k1, k2 = rng.uniform(0.6, 1.6), rng.uniform(1.8, 3.5)
    p1, p2 = rng.uniform(0, 2 * np.pi, 2), rng.uniform(0, 2 * np.pi, 2)
    ph = t / max(L, 1e-6) * 2 * np.pi
    off = np.stack([np.sin(k1 * ph + p1[0]) + 0.5 * np.sin(k2 * ph + p2[0]),
                    np.sin(k1 * ph + p1[1]) + 0.5 * np.sin(k2 * ph + p2[1])], 1) * amp * min(1.0, L / (8 * amp + 1e-6))
    return pts + off


def _glyph_groups(strokes, tol):
    """Agrupa los trazos de una palabra por letra: trazos que se superponen o se tocan a lo ancho (a menos
    de `tol`): el palito de la a con su panza, el punto de la i, la tilde o el travesaño con su letra."""
    iv = [(float(np.min(st[:, 0])), float(np.max(st[:, 0]))) for st in strokes]
    order = np.argsort([a for a, _ in iv])
    gid = [0] * len(strokes)
    g, hi_g = -1, -1e18
    for idx in order:
        lo, hi = iv[idx]
        if lo > hi_g + tol:
            g += 1
            hi_g = hi
        else:
            hi_g = max(hi_g, hi)
        gid[idx] = g
    return gid


def _blur_sprite(sp, sig):
    if sig < 0.3:
        return sp
    pad = int(3 * sig) + 2
    rgb = cv2.copyMakeBorder(sp.rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    a = cv2.copyMakeBorder(sp.a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    rgb = cv2.GaussianBlur(rgb, (0, 0), sig)
    a = cv2.GaussianBlur(a, (0, 0), sig)
    sh = None
    if sp.sh is not None:
        sh = cv2.copyMakeBorder(sp.sh, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return Sprite(sp.x0 - pad, sp.y0 - pad, rgb, a, sh)


def _alpha_into(m, sp):
    H, W = m.shape
    h, w = sp.a.shape
    x0, y0 = sp.x0, sp.y0
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    m[cy0:cy1, cx0:cx1] = np.maximum(m[cy0:cy1, cx0:cx1], sp.a[sl])


def _unit2(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def _cut_path(pts, frac):
    """Los primeros `frac` (0..1) del largo de un camino."""
    pts = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    L = arc[-1] * frac
    k = int(np.searchsorted(arc, L))
    if k <= 0:
        return pts[:1]
    k = min(k, len(pts) - 1)
    f = (L - arc[k - 1]) / max(1e-9, arc[k] - arc[k - 1])
    return np.vstack([pts[:k], pts[k - 1] + (pts[k] - pts[k - 1]) * f])


def _premult_over_local(rgb, a, sp, ox, oy, shadow=0.5):
    """Compone un sprite (coordenadas absolutas) sobre una capa local premultiplicada con origen (ox, oy)."""
    H, W = a.shape
    h, w = sp.a.shape
    x0, y0 = sp.x0 - ox, sp.y0 - oy
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    reg_rgb = rgb[cy0:cy1, cx0:cx1]
    reg_a = a[cy0:cy1, cx0:cx1]
    if sp.sh is not None and shadow > 0:
        s_ = shadow * sp.sh[sl] * (1 - sp.a[sl])
        reg_rgb *= (1 - s_)[..., None]
    sa = sp.a[sl]
    reg_rgb *= (1 - sa)[..., None]
    reg_rgb += sp.rgb[sl]
    reg_a *= (1 - sa)
    reg_a += sa


def _premult_over(rgb, a, sp, shadow=0.5):
    """Compone un sprite sobre una capa premultiplicada del tamaño de la pantalla."""
    H, W = a.shape
    h, w = sp.a.shape
    x0, y0 = sp.x0, sp.y0
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    reg_rgb = rgb[cy0:cy1, cx0:cx1]
    reg_a = a[cy0:cy1, cx0:cx1]
    if sp.sh is not None and shadow > 0:
        # la sombra oscurece lo que ya hay y también cae sobre lo de atrás (se guarda como alfa negro)
        s = shadow * sp.sh[sl] * (1 - sp.a[sl])
        reg_rgb *= (1 - s)[..., None]
        reg_a += s * (1 - reg_a)
    sa = sp.a[sl]
    reg_rgb *= (1 - sa)[..., None]
    reg_rgb += sp.rgb[sl]
    reg_a *= (1 - sa)
    reg_a += sa
