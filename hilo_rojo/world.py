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
from .figures import hershey_strokes
from .textile import fabric, fray_mask, shade
from .thread import Sprite, Yarn, composite, knot, backstitch

LIGHT2 = np.array([-0.64, -0.56])
WOOL = "#c3241c"


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
             neighbors=[dict(img=rgb_lin, x=(x0, x1))]
             title=((x0, y0, x1, y1), [(texto, fuente, alto, cx, base, color, grosor)])
             wool_exit=x_img   dónde sale la lana de la arpillera principal (x en la imagen)
             z0, c0            encuadre inicial (acercamiento y centro, mundo)
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
            self.nbs.append(hg)
        self.ea_main = edge_alpha(*layout["main"]["size"])
        for hg in self.nbs:
            hg.ea = edge_alpha(hg.iw, hg.ih, seed=hg.seed)
        self.wall = plaster(W * 2, H * 2, seed=seed)
        self.title = self._title_strip(layout["title"])
        self.pin_L = 56 * self.u
        self.line_w = 6.2 * self.u
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
        self.pulse = None
        if layout.get("wool_exit") is not None:
            xs = float(self.main.img_to_world(layout["wool_exit"][1])[0])
            self.pulse = dict(x=xs, v=1450 * self.u, A=14 * self.u, w=105 * self.u, decay=2600 * self.u)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = xx / W - 0.5, yy / H - 0.5
        self.vign = (1 - 0.30 * (nx * nx * 1.1 + ny * ny * 1.3) ** 1.1).astype(np.float32)
        self.rake = (1 + 0.07 * (-(nx * 0.8 + ny * 0.6))).astype(np.float32)
        self.xx, self.yy = xx, yy

    # ------------------------------------------------------------------------------------------------
    def _title_strip(self, spec):
        """Tira de tocuyo con el título bordado (punto atrás; ALERTA en la misma lana roja), clavada a la
        pared. Se arma a 2x para que aguante el alejamiento."""
        rng = np.random.default_rng(77)
        u2 = 2 * self.u
        x0, y0, x1, y1 = [2 * v for v in spec["rect"]]
        pad = 40
        shape = (int(y1 - y0) + 2 * pad, int(x1 - x0) + 2 * pad)
        ox, oy = x0 - pad, y0 - pad
        fab, _ = fabric(shape, rng, "#efe5cd", kind="plain", scale=u2)
        m = fray_mask([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], shape, rng, offset=(ox, oy), fray=2.2)
        pf = shade(cv2.GaussianBlur(m, (0, 0), 5) * 3.2, strength=1.0, ambient=0.75)
        pf = pf / max(1e-6, float(np.percentile(pf[m > 0.9], 60)))
        # una leve onda de tela clavada solo arriba
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        wave = 1 + 0.035 * np.sin(xx / shape[1] * np.pi * 3.0 + 0.8) * (yy / shape[0])
        rgb = fab * (np.clip(pf, 0.6, 1.2) * wave)[..., None] * m[..., None]
        sh = cv2.warpAffine(cv2.GaussianBlur(m, (0, 0), 7), np.float32([[1, 0, 11], [0, 1, 10]]),
                            (shape[1], shape[0]))
        out = [Sprite(int(ox), int(oy), rgb.astype(np.float32), m.astype(np.float32), sh)]
        font = spec.get("font", "scripts")
        cx = (x0 + x1) / 2
        dark = spec.get("color", "#2b211b")
        for (base, cap, words) in spec["lines"]:
            ws = []
            for (txt, kind) in words:
                strokes, w = hershey_strokes(txt, font, cap * 2, 0, base * 2, anchor="left", ref="H")
                ws.append((strokes, w, kind))
            space = cap * 2 * 0.45
            total = sum(w for _, w, _ in ws) + space * (len(ws) - 1)
            x = cx - total / 2
            for strokes, w, kind in ws:
                for st in strokes:
                    st = st + np.array([x, 0.0])
                    if kind == "wool":
                        if len(st) < 2:
                            continue
                        y = Yarn(st, 5.0 * u2, WOOL, rng, fuzz=0.45, step=1.0)
                        out += y.chunks
                    else:
                        for _, sp in backstitch(st, 6.5 * u2, 2.3 * u2, dark, rng, jitter=0.2):
                            out.append(sp)
                x += w + space
        for (nx, ny) in ((x0 + 24, y0 + 22), (x1 - 24, y0 + 22)):
            out.append(knot((nx, ny), 6.5 * u2, "#4a4440", rng))
        return out

    # ------------------------------------------------------------------------------------------------
    def _plane(self, z, c):
        """Transformación del plano del cordel (mundo -> pantalla)."""
        return np.array([[z, 0, self.W / 2 - c[0] * z], [0, z, self.H / 2 - c[1] * z]], np.float64)

    def _warp_hanging(self, hg, z, c, iw_src=None):
        """Mapas pantalla -> imagen de una arpillera colgada (con comba y pliegues). Devuelve un dict
        con la caja en pantalla, los mapas, el alfa y los pliegues; o None si no se ve."""
        W, H = self.W, self.H
        P = self._plane(z, c)
        cw = hg.corners_world()
        sc = cw @ P[:, :2].T + P[:, 2]
        pad = 6 + int(hg.ih * hg.s * z * hg.sag * 1.5)
        bx0, by0 = int(np.floor(sc[:, 0].min())) - pad, int(np.floor(sc[:, 1].min())) - pad
        bx1, by1 = int(np.ceil(sc[:, 0].max())) + pad, int(np.ceil(sc[:, 1].max())) + pad
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
        if bx1 <= bx0 or by1 <= by0:
            return None
        # pantalla -> mundo -> imagen
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
        # la tela colgada no es un plano: lomas suaves (luz rasante) y bordes que ondulan un poco
        rh = np.random.default_rng(100 + hg.seed)
        cloth = np.ones_like(u_)
        for _ in range(4):
            fu, fv = rh.uniform(0.6, 2.4), rh.uniform(0.4, 1.8)
            amp = hg.ih * 0.0045 * rh.uniform(0.6, 1.0)
            cs = np.cos(2 * np.pi * (fu * u_ + fv * v_) + rh.uniform(0, 2 * np.pi))
            cloth += 1.2 * (0.64 * amp * cs * 2 * np.pi * fu / hg.iw + 0.56 * amp * cs * 2 * np.pi * fv / hg.ih)
        for axis in (0, 1):
            b = hg.ih * 0.0026
            fu, fv = rh.uniform(0.8, 2.2), rh.uniform(0.6, 1.6)
            dd = b * np.sin(2 * np.pi * (fu * u_ + fv * v_) + rh.uniform(0, 2 * np.pi))
            if axis == 0:
                ix = ix + dd
            else:
                iy = iy + dd
        # comba entre perritos: el borde de arriba cuelga en el medio
        sag = hg.ih * hg.sag * (1 - ((u_ - 0.5) / 0.43) ** 2).clip(0, 1) * (1 - v_) ** 2
        iy = iy - sag
        k = hg.s * z                                       # escala de pantalla
        f = 1.0
        if k < 0.8:
            f = k * 1.15
        folds = (1 + 0.045 * np.sin(u_ * np.pi * hg.folds + 0.6 + hg.seed) * (1 - v_) ** 1.5 * np.sin(u_ * np.pi)) * cloth
        return dict(box=(bx0, by0, bx1, by1), mx=(ix * f - 0.5 + 0.5 * f).astype(np.float32) if f != 1 else (ix - 0.5).astype(np.float32),
                    my=(iy * f - 0.5 + 0.5 * f).astype(np.float32) if f != 1 else (iy - 0.5).astype(np.float32),
                    f=f, folds=folds.astype(np.float32))

    @staticmethod
    def _sample(img, wp):
        if wp["f"] != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, int(round(w * wp["f"]))), max(1, int(round(h * wp["f"])))),
                             interpolation=cv2.INTER_AREA)
        return cv2.remap(img, wp["mx"], wp["my"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _shadow_into(self, out, a_full, z):
        """Sombra de una tela colgada sobre la pared (luz rasante arriba-izquierda)."""
        u = self.u * z
        sh = cv2.GaussianBlur(a_full, (0, 0), max(1.0, 7 * u))
        off = 11 * u
        sh = cv2.warpAffine(sh, np.float32([[1, 0, off], [0, 1, off * 0.9]]), (self.W, self.H))
        out *= (1 - 0.40 * sh)[..., None]

    def _static(self, z, c):
        key = (round(z, 5), round(c[0], 3), round(c[1], 3))
        if key in self._cache:
            return self._cache[key]
        W, H = self.W, self.H
        zw = 1 + (z - 1) * 0.72                            # paralaje: la pared está más lejos
        cw = np.asarray(c, np.float64)
        Mw = np.float32([[zw / 2, 0, W / 2 - cw[0] * zw], [0, zw / 2, H / 2 - cw[1] * zw]])
        bg = cv2.warpAffine(self.wall, Mw, (W, H), flags=cv2.INTER_AREA if zw < 2 else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        self._draw_sprites(bg, self.title, Mw, shadow=0.35)
        # alfas en pantalla de todas las telas colgadas (para sombras)
        a_all = np.zeros((H, W), np.float32)
        layers = []
        for hg in self.nbs:
            wp = self._warp_hanging(hg, z, c)
            if wp is None:
                continue
            x0, y0, x1, y1 = wp["box"]
            a = self._sample(hg.ea, wp)
            rgb = self._sample(hg.img, wp) * wp["folds"][..., None]
            a_all[y0:y1, x0:x1] = np.maximum(a_all[y0:y1, x0:x1], a)
            layers.append((wp["box"], rgb, a))
        wpm = self._warp_hanging(self.main, z, c)
        am = np.zeros((H, W), np.float32)
        if wpm is not None:
            x0, y0, x1, y1 = wpm["box"]
            am[y0:y1, x0:x1] = self._sample(self.ea_main, wpm)
        a_all = np.maximum(a_all, am)
        self._shadow_into(bg, a_all, z)
        for (x0, y0, x1, y1), rgb, a in layers:
            reg = bg[y0:y1, x0:x1]
            reg *= (1 - a)[..., None]
            reg += rgb * a[..., None]
        # delante: cordel, nudos y perritos
        fg_rgb = np.zeros((H, W, 3), np.float32)
        fg_a = np.zeros((H, W), np.float32)
        P = self._plane(z, c)
        u = self.u * z
        line = self.line_pts @ P[:, :2].T + P[:, 2]
        vis = (line[:, 0] > -80) & (line[:, 0] < W + 80) & (line[:, 1] > -80) & (line[:, 1] < H + 80)
        if vis.any():
            i0 = max(0, int(np.argmax(vis)) - 2)
            i1 = min(len(line), len(vis) - int(np.argmax(vis[::-1])) + 2)
            seg = line[i0:i1]
            if len(seg) > 2:
                y = Yarn(seg, self.line_w * z, WOOL, np.random.default_rng(7), fuzz=0.9, step=max(1.0, 1.6 * u))
                self._draw_yarn_premult(fg_rgb, fg_a, y)
        # nudos de las lanas de las otras arpilleras
        for hg in self.nbs:
            if hg.exit is not None:
                self._wool_up(fg_rgb, fg_a, hg, hg.exit, P, z, hg.wool_w)
        for hg in self.nbs + [self.main]:
            for i, pw in enumerate(hg.pin_pts):
                sp = pw @ P[:, :2].T + P[:, 2]
                if -200 < sp[0] < W + 200 and -200 < sp[1] < H + 200:
                    ang = -hg.theta + np.deg2rad((-3, 4)[i] + 2 * np.sin(pw[0]))
                    spr = clothespin(sp, ang, self.pin_L * z, int(pw[0] * 7) % 1000)
                    _premult_over(fg_rgb, fg_a, spr, shadow=0.5)
        # fondo con la luz rasante de la sala
        st = dict(bg=bg, am=am, wpm=wpm, fg_rgb=fg_rgb, fg_a=fg_a, P=P, z=z)
        # la lana de la arpillera principal hasta el cordel (aparece cuando la lana llega arriba)
        kr, ka = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
        if self.L.get("wool_exit") is not None:
            self._wool_up(kr, ka, self.main, self.L["wool_exit"], P, z, self.L.get("wool_w", 8.5))
        st["knot_rgb"], st["knot_a"] = kr, ka
        if len(self._cache) > 3:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = st
        return st

    def _wool_up(self, rgb, a, hg, ex, P, z, w_img, M=None, tau=None):
        """La lana que sale por el borde de arriba de una arpillera sube al cordel y se anuda.
        ex = [(x, y) en la guarda, (x, 0) en el borde] (coordenadas de la imagen). M: pose de la tela
        (afín en pantalla) mientras pasa el pulso; tau: tiempo del pulso (desplaza el cordel)."""
        p_in = hg.img_to_world(ex[0])
        p_edge = hg.img_to_world(ex[1])
        d = p_edge - p_in
        xt = p_edge[0] + d[0] * 0.25
        top = np.array([xt, self.line_y(xt) - 1.0 * self.u + float(self.pulse_dy(np.array([xt]), tau)[0])])
        S = lambda q: np.asarray(q) @ P[:, :2].T + P[:, 2]
        a_s, b_s = S(p_in), S(p_edge)
        if M is not None:
            a_s = M[:, :2] @ a_s + M[:, 2]
            b_s = M[:, :2] @ b_s + M[:, 2]
        pts = np.array([a_s, b_s, S(top)])
        w = w_img * hg.s * z
        rng = np.random.default_rng(int(abs(ex[0][0])) + 3)
        y = Yarn(catmull_rom(pts, 10), w, WOOL, rng, fuzz=0.9)
        self._draw_yarn_premult(rgb, a, y)
        kp = pts[-1]
        r = max(w, self.line_w * z) * 0.62
        for dx, dy, k in ((-0.55, 0.15, 0.95), (0.6, 0.1, 1.0), (0.0, -0.1, 1.1)):
            _premult_over(rgb, a, knot(kp + np.array([dx, dy]) * r * 1.2, r * k, "#b41f18", rng), shadow=0.45)

    # ------------------------------------------------------------------------------------------------
    # el pulso: la alerta recorre el cordel de arpillera en arpillera
    def pulse_dy(self, xw, tau):
        """Desplazamiento vertical (mundo) del cordel en x, tau segundos después de partir el pulso."""
        xw = np.asarray(xw, np.float64)
        p = self.pulse
        if p is None or tau is None or tau < 0:
            return np.zeros_like(xw)
        dd = np.abs(xw - p["x"])
        r = p["v"] * tau
        env = np.exp(-dd / p["decay"])
        bump = np.exp(-((dd - r) / p["w"]) ** 2) - 0.38 * np.exp(-((dd - r + 1.7 * p["w"]) / p["w"]) ** 2)
        return -p["A"] * env * bump

    def pulse_active(self, tau):
        if self.pulse is None or tau is None or tau < 0:
            return False
        far = max(abs(self.pulse["x"]), abs(self.W - self.pulse["x"])) + 4 * self.pulse["w"]
        return self.pulse["v"] * tau < far

    def _pose(self, hg, P, tau):
        """Afín en pantalla que lleva una tela colgada a su pose con el cordel desplazado."""
        pa, pb = hg.pin_pts
        da, db = self.pulse_dy(np.array([pa[0], pb[0]]), tau)
        pa2, pb2 = pa + np.array([0, da]), pb + np.array([0, db])
        th0 = np.arctan2(pb[1] - pa[1], pb[0] - pa[0])
        th1 = np.arctan2(pb2[1] - pa2[1], pb2[0] - pa2[0])
        dth = float(th1 - th0)
        S = lambda q: np.asarray(q) @ P[:, :2].T + P[:, 2]
        m0, m1 = S((pa + pb) / 2), S((pa2 + pb2) / 2)
        c_, s_ = np.cos(dth), np.sin(dth)
        R = np.array([[c_, -s_], [s_, c_]])
        t = m1 - R @ m0
        return np.float32([[c_, -s_, t[0]], [s_, c_, t[1]]]), dth, (pa2, pb2)

    def _dyn(self, z, c):
        """Capas separadas (pared, cada tela en pantalla completa) para mover las telas con el pulso."""
        key = ("dyn", round(z, 5), round(c[0], 3), round(c[1], 3))
        if key in self._cache:
            return self._cache[key]
        W, H = self.W, self.H
        zw = 1 + (z - 1) * 0.72
        cw = np.asarray(c, np.float64)
        Mw = np.float32([[zw / 2, 0, W / 2 - cw[0] * zw], [0, zw / 2, H / 2 - cw[1] * zw]])
        bg = cv2.warpAffine(self.wall, Mw, (W, H), flags=cv2.INTER_AREA if zw < 2 else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        self._draw_sprites(bg, self.title, Mw, shadow=0.35)
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
            lays.append((hg, rgbF, aF))
        wpm = self._warp_hanging(self.main, z, c)
        amF = np.zeros((H, W), np.float32)
        if wpm is not None:
            x0, y0, x1, y1 = wpm["box"]
            amF[y0:y1, x0:x1] = self._sample(self.ea_main, wpm)
        st = dict(bg=bg, lays=lays, wpm=wpm, amF=amF, P=self._plane(z, c))
        if len(self._cache) > 3:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = st
        return st

    def render_pulse(self, main_lin, z, c, tau, flick=0.0):
        """Cuadro con el pulso recorriendo el cordel (las telas siguen a sus perritos)."""
        W, H = self.W, self.H
        st = self._dyn(z, c)
        P = st["P"]
        u = self.u * z
        out = st["bg"].copy()
        mrgb = np.zeros((H, W, 3), np.float32)
        wpm = st["wpm"]
        if wpm is not None:
            x0, y0, x1, y1 = wpm["box"]
            mrgb[y0:y1, x0:x1] = self._sample(main_lin, wpm) * wpm["folds"][..., None]
        placed, poses = [], {}
        for hg, rgbF, aF in st["lays"] + [(self.main, mrgb, st["amF"])]:
            M, dth, pins2 = self._pose(hg, P, tau)
            poses[id(hg)] = (M, dth, pins2)
            placed.append((cv2.warpAffine(rgbF, M, (W, H), flags=cv2.INTER_LINEAR),
                           cv2.warpAffine(aF, M, (W, H), flags=cv2.INTER_LINEAR)))
        a_all = np.zeros((H, W), np.float32)
        for _, a2 in placed:
            a_all = np.maximum(a_all, a2)
        self._shadow_into(out, a_all, z)
        for r2, a2 in placed:
            out *= (1 - a2)[..., None]
            out += r2 * a2[..., None]
        fg_rgb = np.zeros((H, W, 3), np.float32)
        fg_a = np.zeros((H, W), np.float32)
        lw_ = self.line_pts.copy()
        lw_[:, 1] += self.pulse_dy(lw_[:, 0], tau)
        line = lw_ @ P[:, :2].T + P[:, 2]
        vis = (line[:, 0] > -80) & (line[:, 0] < W + 80) & (line[:, 1] > -80) & (line[:, 1] < H + 80)
        if vis.any():
            i0 = max(0, int(np.argmax(vis)) - 2)
            i1 = min(len(line), len(vis) - int(np.argmax(vis[::-1])) + 2)
            seg = line[i0:i1]
            if len(seg) > 2:
                y = Yarn(seg, self.line_w * z, WOOL, np.random.default_rng(7), fuzz=0.9, step=max(1.0, 1.6 * u))
                self._draw_yarn_premult(fg_rgb, fg_a, y)
        for hg in self.nbs + [self.main]:
            ex = self.L.get("wool_exit") if hg is self.main else hg.exit
            w_img = self.L.get("wool_w", 8.5) if hg is self.main else hg.wool_w
            if ex is not None:
                self._wool_up(fg_rgb, fg_a, hg, ex, P, z, w_img, M=poses[id(hg)][0], tau=tau)
        for hg in self.nbs + [self.main]:
            M, dth, pins2 = poses[id(hg)]
            for i, pw in enumerate(pins2):
                sp = pw @ P[:, :2].T + P[:, 2]
                if -200 < sp[0] < W + 200 and -200 < sp[1] < H + 200:
                    ang = -(hg.theta + dth) + np.deg2rad((-3, 4)[i] + 2 * np.sin(hg.pin_pts[i][0]))
                    spr = clothespin(sp, ang, self.pin_L * z, int(hg.pin_pts[i][0] * 7) % 1000)
                    _premult_over(fg_rgb, fg_a, spr, shadow=0.5)
        out *= (1 - fg_a)[..., None]
        out += fg_rgb
        out *= (self.rake * self.vign * (1 + flick))[..., None]
        return np.clip(out, 0, 1)

    def _draw_yarn_premult(self, rgb, a, y):
        for ch in y.chunks:
            _premult_over(rgb, a, ch, shadow=0.45)

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
    def render(self, main_lin, z, c, F=1.0, B=0.0, lamp=None, flick=0.0, wool_tied=False):
        """main_lin: la arpillera principal ya iluminada (lineal, su propio encuadre).
        F: luz de frente de la sala; B, lamp: contraluz (lamp = centro en la imagen o 'full')."""
        st = self._static(z, c)
        out = st["bg"] * F
        wpm = st["wpm"]
        if wpm is not None:
            x0, y0, x1, y1 = wpm["box"]
            a = st["am"][y0:y1, x0:x1]
            rgb = self._sample(main_lin, wpm) * wpm["folds"][..., None]
            reg = out[y0:y1, x0:x1]
            reg *= (1 - a)[..., None]
            reg += rgb * a[..., None]
            if B > 0 and lamp is not None:
                # la lámpara de atrás se escapa por los bordes y lava la pared
                if isinstance(lamp, str):
                    fld = st["am"]
                else:
                    P = st["P"]
                    lw = self.main.img_to_world(lamp["c"])
                    lsx, lsy = lw @ P[:, :2].T + P[:, 2]
                    R = lamp["R"] * self.main.s * z
                    fld = st["am"] * np.exp(-((self.xx - lsx) ** 2 + (self.yy - lsy) ** 2) / (2 * R * R))
                spill = cv2.GaussianBlur(fld, (0, 0), 38 * self.u * z) * (1 - st["am"])
                lampc = np.array([1.0, 0.86, 0.66], np.float32)
                out += (B * 1.4 * spill)[..., None] * lampc * st["bg"]
        if wool_tied:
            out *= (1 - st["knot_a"])[..., None]
            out += st["knot_rgb"] * F
        out *= (1 - st["fg_a"])[..., None]
        out += st["fg_rgb"] * F
        out *= (self.rake * self.vign * (1 + flick))[..., None]
        return np.clip(out, 0, 1)


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
