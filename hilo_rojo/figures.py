"""Figuras de la arpillera: casas, árboles, araucarias, sol, nubes, muñecas de tela, turbinas de
organza, la aguja y la etiqueta bordada. Todo se arma con retazos (Piece), lana (Yarn), puntadas
(Stitch) y nudos, nunca con rellenos planos."""
import cv2
import numpy as np
from HersheyFonts import HersheyFonts

from satc_intro.color import lin
from satc_intro.geometry import catmull_rom, resample
from satc_intro.noise import smooth_noise, smoothstep
from .pieces import Piece, ellipse_poly, UFPS
from .textile import fabric, fray_mask, organza
from .thread import Sprite, Stitch, Yarn, composite, knot, running_stitch, satin_fill

INK_THREAD = "#2a2019"


# --- texto de un solo trazo (fuentes Hershey, dominio público) -------------------------------------
def hershey_strokes(text, font, height, x, y, anchor="left", tracking=0.0):
    """Trazos (polilíneas) del texto, con la línea base en y. height: altura total del texto (de la
    línea base al punto más alto)."""
    f = HersheyFonts()
    f.load_default_font(font)
    f.normalize_rendering(100)
    segs = list(f.lines_for_text(text))
    strokes, cur = [], None
    for (a, b) in segs:
        if cur is not None and np.allclose(cur[-1], a):
            cur.append(b)
        else:
            if cur:
                strokes.append(cur)
            cur = [a, b]
    if cur:
        strokes.append(cur)
    ys = np.concatenate([np.asarray(s)[:, 1] for s in strokes])
    base_y, top_y = float(ys.min()), float(ys.max())
    k = height / max(1e-6, top_y - base_y)
    xs = np.concatenate([np.asarray(s)[:, 0] for s in strokes])
    w = (xs.max() - xs.min()) * k * (1 + tracking)
    ox = x - (w / 2 if anchor == "center" else (w if anchor == "right" else 0)) - xs.min() * k
    out = []
    for s in strokes:
        s = np.asarray(s, np.float64)
        out.append(np.stack([ox + s[:, 0] * k * (1 + tracking), y - (s[:, 1] - base_y) * k], 1))
    return out, w


# --- la aguja ---------------------------------------------------------------------------------------
def needle_sprite(back, tip, width, hide_from=None):
    """Aguja de acero: brillo a lo largo, ojo abierto, sombra. hide_from: fracción desde la cual la
    punta ya entró en la tela (no se ve)."""
    back, tip = np.asarray(back, np.float64), np.asarray(tip, np.float64)
    d = tip - back
    L = float(np.hypot(*d)) + 1e-9
    t = d / L
    n = np.array([-t[1], t[0]])
    pad = width * 3 + 6
    x0 = int(np.floor(min(back[0], tip[0]) - pad))
    y0 = int(np.floor(min(back[1], tip[1]) - pad))
    x1 = int(np.ceil(max(back[0], tip[0]) + pad))
    y1 = int(np.ceil(max(back[1], tip[1]) + pad))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    qx, qy = xx - back[0], yy - back[1]
    s = (qx * t[0] + qy * t[1]) / L
    e = qx * n[0] + qy * n[1]
    r = width / 2 * np.clip(np.where(s < 0.72, 1.0, (1 - s) / 0.28), 0, 1) ** 0.8
    r = np.where((s < -0.02) | (s > 1), 0, r)
    r = np.where(s < 0.03, width / 2 * np.clip(s / 0.03, 0, 1) ** 0.5, r)
    a = np.clip(r - np.abs(e) + 0.5, 0, 1)
    # ojo: ranura alargada cerca del extremo trasero
    eye = (np.abs(e) < width * 0.18) & (s > 0.04) & (s < 0.12)
    a = np.where(eye, a * 0.05, a)
    if hide_from is not None:
        a = a * np.clip((hide_from - s) * L / 1.5 + 0.5, 0, 1)
    dz = np.clip(e / np.maximum(r, 1e-3), -1, 1)
    spec = np.exp(-((dz + 0.35) / 0.22) ** 2) * 0.9 + np.exp(-((dz - 0.55) / 0.3) ** 2) * 0.15
    base = 0.30 + 0.25 * (1 - np.abs(dz))
    val = np.clip(base + spec, 0, 1.4)
    steel = lin("#b9c0c6")
    rgb = steel[None, None, :] * val[..., None] + np.array([0.02, 0.02, 0.025])[None, None, :]
    a = a.astype(np.float32)
    sh = cv2.GaussianBlur(a, (0, 0), 2.2)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, 5], [0, 1, 6]]), (sh.shape[1], sh.shape[0]))
    return Sprite(x0, y0, (rgb * a[..., None]).astype(np.float32), a, sh * 0.7)


# --- casas, árboles, sol, nubes ----------------------------------------------------------------------
def house(cx, base, w, h, rng, t, wall="#c0392b", roof="#6e6a66", wall_kind="plain", roof_kind="cord",
          windows=1, door_side=0.0, u=1.0):
    parts = []
    x0, x1 = cx - w / 2, cx + w / 2
    top = base - h
    wall_poly = [(x0, base), (x0, top), (x1, top), (x1, base)]
    parts.append(Piece(wall_poly, wall, rng, kind=wall_kind, t_place=t, stitch=("#3d2a1e", 7 * u, 5 * u, 1.6 * u),
                       fabric_scale=u))
    ov = w * 0.12
    roof_poly = [(x0 - ov, top + 4 * u), (cx - w * 0.08, top - h * 0.62), (cx + w * 0.08, top - h * 0.62),
                 (x1 + ov, top + 4 * u)]
    parts.append(Piece(roof_poly, roof, rng, kind=roof_kind, t_place=t, stitch=("#2b2622", 7 * u, 5 * u, 1.6 * u),
                       fabric_scale=u, angle=np.pi / 2))
    dw, dh = w * 0.24, h * 0.52
    dx = cx + door_side * w * 0.22
    parts.append(Piece([(dx - dw / 2, base), (dx - dw / 2, base - dh), (dx + dw / 2, base - dh), (dx + dw / 2, base)],
                       "#5a3a22", rng, kind="felt", t_place=t, stitch=None, inset=3, margin=6))
    wins = []
    for k in range(windows):
        wx = cx - door_side * w * 0.22 + (k - (windows - 1) / 2) * w * 0.30
        ww = w * 0.2
        wy = top + h * 0.22
        poly = [(wx - ww / 2, wy), (wx + ww / 2, wy), (wx + ww / 2, wy + ww), (wx - ww / 2, wy + ww)]
        parts.append(Piece(poly, "#2c3440", rng, kind="felt", t_place=t, stitch=None, margin=6))
        lit = satin_fill([(p[0] + 1.2 * u, p[1] + 1.2 * u) if i == 0 else p for i, p in enumerate(poly)],
                         np.deg2rad(8), 2.1 * u, 2.4 * u, "#f3c33a", rng)
        wins.append(lit)
    door_x = dx
    return parts, wins, (door_x, base - dh * 0.45)


def round_tree(cx, base, r, rng, t, color="#3f6b3c", u=1.0):
    trunk = Yarn([(cx, base), (cx + rng.normal(0, 1), base - r * 1.2)], 4.5 * u, "#5b3b22", rng, fuzz=0.6)
    crown = Piece(ellipse_poly(cx, base - r * 1.55, r, r * 0.92, wobble=0.05, rng=rng), color, rng, kind="felt",
                  t_place=t, stitch=("#26401f", 6 * u, 5 * u, 1.5 * u), fabric_scale=u)
    return [trunk], [crown]


def araucaria(cx, base, hgt, rng, t, u=1.0):
    """La araucaria: tronco recto y copa de «paraguas» en pisos."""
    trunk = Yarn([(cx, base), (cx, base - hgt * 0.92)], 5.5 * u, "#4a3220", rng, fuzz=0.5)
    tiers = []
    for i, (fy, fw) in enumerate([(0.70, 0.62), (0.82, 0.50), (0.94, 0.34)]):
        y = base - hgt * fy
        wdt = hgt * fw
        poly = [(cx - wdt / 2, y + 6 * u), (cx - wdt * 0.35, y - 9 * u), (cx, y - 14 * u), (cx + wdt * 0.35, y - 9 * u),
                (cx + wdt / 2, y + 6 * u), (cx, y + 1 * u)]
        tiers.append(Piece(poly, "#23452e", rng, kind="felt", t_place=t + 0.02 * i,
                           stitch=("#132a18", 5 * u, 4 * u, 1.3 * u), fabric_scale=u))
    return [trunk], tiers


def sun(cx, cy, r, rng, t, u=1.0):
    disc = Piece(ellipse_poly(cx, cy, r, r, n=48), "#f1b82d", rng, kind="felt", t_place=t,
                 stitch=("#c8761b", 6 * u, 5 * u, 1.8 * u), fabric_scale=u)
    rays = []
    for k in range(14):
        a = k * 2 * np.pi / 14 + rng.normal(0, 0.04)
        p0 = (cx + np.cos(a) * r * 1.18, cy + np.sin(a) * r * 1.18)
        p1 = (cx + np.cos(a) * r * (1.55 + 0.12 * (k % 2)), cy + np.sin(a) * r * (1.55 + 0.12 * (k % 2)))
        rays.append(Stitch.render(p0, p1, 2.8 * u, "#e07b1a", rng))
    return disc, rays


def cloud(cx, cy, w, h, rng, t, u=1.0):
    """Nube de retazo: unión de lóbulos (contorno real, cortado a tijera)."""
    pad = 20
    W_, H_ = int(w * 1.3 + 2 * pad), int(h * 1.6 + 2 * pad)
    m = np.zeros((H_, W_), np.uint8)
    ox, oy = cx - W_ / 2, cy - H_ / 2
    base_y = H_ / 2 + h * 0.35
    n = 5
    for k in range(n):
        fx = (k - (n - 1) / 2) / ((n - 1) / 2)
        rx = w * (0.16 + 0.06 * (1 - abs(fx))) * rng.uniform(0.9, 1.1)
        ry = h * (0.30 + 0.28 * (1 - abs(fx))) * rng.uniform(0.9, 1.1)
        ccx = W_ / 2 + fx * w * 0.36
        ccy = base_y - ry * 0.8
        cv2.ellipse(m, (int(ccx), int(ccy)), (int(rx), int(ry)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.rectangle(m, (int(W_ / 2 - w * 0.45), int(base_y - h * 0.25)), (int(W_ / 2 + w * 0.45), int(base_y)), 255, -1)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    c = resample(np.vstack([c, c[:1]]), 6.0)[:-1] + [ox, oy]
    return Piece(c, "#f4f0e6", rng, kind="plain", t_place=t, stitch=("#b9b3a6", 6 * u, 5 * u, 1.5 * u),
                 fabric_scale=u)


# --- muñecas de tela (los «monitos» de la arpillera) -------------------------------------------------
def doll(fx, base, hgt, rng, t, dress="#2f6fa6", dress_kind="dots", dress2="#f2eadb", skin="#c68e67",
         hair="#1f1712", hand_y=None, arms="up", u=1.0):
    """Muñeca cosida: vestido de retazo, cabeza rellena, pelo de lana, brazos de lana tomando el hilo."""
    head_r = hgt * 0.16
    neck_y = base - hgt * 0.68
    hip_y = base - hgt * 0.22
    dress_poly = [(fx - hgt * 0.10, neck_y), (fx + hgt * 0.10, neck_y), (fx + hgt * 0.24, hip_y + hgt * 0.02),
                  (fx - hgt * 0.24, hip_y + hgt * 0.02)]
    body = Piece(dress_poly, dress, rng, kind=dress_kind, color2=dress2, t_place=t, stitch=("#2a2019", 5 * u, 4 * u,
                                                                                           1.3 * u),
                 fabric_scale=u * 0.8, margin=8, inset=3.5)
    legs = [Yarn([(fx - hgt * 0.07, hip_y), (fx - hgt * 0.08, base - 2)], 3.2 * u, "#2a2019", rng, fuzz=0.4),
            Yarn([(fx + hgt * 0.07, hip_y), (fx + hgt * 0.08, base - 2)], 3.2 * u, "#2a2019", rng, fuzz=0.4)]
    hy = hand_y if hand_y is not None else neck_y - hgt * 0.10
    sh_l = (fx - hgt * 0.10, neck_y + hgt * 0.05)
    sh_r = (fx + hgt * 0.10, neck_y + hgt * 0.05)
    hand_l = np.array([fx - hgt * 0.30, hy])
    hand_r = np.array([fx + hgt * 0.30, hy])
    arm_l = Yarn(catmull_rom(np.array([sh_l, ((sh_l[0] + hand_l[0]) / 2 - 3 * u, (sh_l[1] + hand_l[1]) / 2 + 4 * u),
                                       hand_l]), 6), 3.4 * u, skin, rng, fuzz=0.3)
    arm_r = Yarn(catmull_rom(np.array([sh_r, ((sh_r[0] + hand_r[0]) / 2 + 3 * u, (sh_r[1] + hand_r[1]) / 2 + 4 * u),
                                       hand_r]), 6), 3.4 * u, skin, rng, fuzz=0.3)
    # cabeza rellena: esfera de tela con luz fuerte
    hc = (fx, neck_y - head_r * 0.92)
    head = stuffed_head(hc, head_r, skin, rng)
    hair_y = []
    for k in range(9):
        a = np.pi * (0.95 + 1.1 * k / 8)
        p = (hc[0] + np.cos(a) * head_r * 0.92, hc[1] + np.sin(a) * head_r * 0.92)
        L = head_r * (1.25 if k in (0, 8) else 1.12)
        q = (hc[0] + np.cos(a) * L, hc[1] + np.sin(a) * L + (head_r * 0.9 if k in (0, 8) else 0))
        hair_y.append(Yarn(catmull_rom(np.array([p, ((p[0] + q[0]) / 2 + rng.normal(0, 1), (p[1] + q[1]) / 2 - 2 * u),
                                                 q]), 5), 3.4 * u, hair, rng, fuzz=0.8))
    eyes = []   # sin carita: como en las arpilleras
    shoes = [knot((fx - hgt * 0.08, base - 1), 2.4 * u, "#1a1410", rng),
             knot((fx + hgt * 0.08, base - 1), 2.4 * u, "#1a1410", rng)]
    return dict(t=t, body=body, legs=legs, arms=[arm_l, arm_r], head=head, hair=hair_y, eyes=eyes, shoes=shoes,
                hands=(hand_l, hand_r))


def stuffed_head(c, r, color, rng):
    col = lin(color)
    cx, cy = c
    pad = r * 1.6 + 4
    x0, y0 = int(cx - pad), int(cy - pad)
    x1, y1 = int(cx + pad), int(cy + pad)
    shape = (y1 - y0, x1 - x0)
    fab, _ = fabric(shape, rng, color, kind="plain", pitch=1.8)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    dx, dy = (xx - cx) / r, (yy - cy) / r
    rr = np.sqrt(dx * dx + dy * dy)
    a = np.clip((1 - rr) * r + 0.5, 0, 1).astype(np.float32)
    nz = np.sqrt(np.clip(1 - rr * rr, 0, 1))
    L = np.array([-0.55, -0.62, 0.56])
    L /= np.linalg.norm(L)
    lam = np.clip(dx * L[0] + dy * L[1] + nz * L[2], 0, 1)
    shade_ = 0.35 + 0.85 * lam
    rgb = fab * shade_[..., None] * a[..., None]
    sh = cv2.GaussianBlur(a, (0, 0), r * 0.3)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, r * 0.25], [0, 1, r * 0.32]]), (shape[1], shape[0]))
    return Sprite(x0, y0, rgb.astype(np.float32), a, sh)


def draw_doll(canvas, d, t, lift=False):
    if t < d["t"]:
        return
    k = (t - d["t"]) * UFPS
    dy = -6 if k < 1 else 0
    for y in d["legs"]:
        _yarn_full(canvas, y, dy)
    d["body"].draw(canvas, t)
    for s in d["shoes"]:
        composite(canvas, s, dy=dy)
    composite(canvas, d["head"], dy=dy)
    for y in d["hair"]:
        _yarn_full(canvas, y, dy)
    for s in d["eyes"]:
        composite(canvas, s, dy=dy)
    for y in d["arms"]:
        _yarn_full(canvas, y, dy)


def _yarn_full(canvas, y, dy=0):
    if dy == 0:
        y.draw(canvas, y.length + 10)
        return
    for sp in y.chunks:
        composite(canvas, sp, dy=dy)


# --- turbinas de organza (lo que viene) ---------------------------------------------------------------
class OrganzaTurbine:
    def __init__(self, base, hgt, rng, t, u=1.0, phase=0.0, speed=1.0):
        self.base = np.asarray(base, np.float64)
        self.hgt = hgt
        self.t = t
        self.u = u
        self.phase = phase
        self.speed = speed
        self.hub = self.base + np.array([0, -hgt])
        self.blade = hgt * 0.62
        w0, w1 = hgt * 0.035, hgt * 0.018
        bx, by = self.base
        hx, hy = self.hub
        self.tower_poly = np.array([(bx - w0, by), (hx - w1, hy), (hx + w1, hy), (bx + w0, by)])
        self.rng = rng
        self.seed = int(rng.integers(1 << 30))
        # textura de organza para un recuadro que contiene todo el rotor
        R = int(self.blade * 1.1 + 20)
        self.R = R
        self.mesh, self.sheen, _ = organza((2 * R, 2 * R), np.random.default_rng(self.seed))
        tx0 = int(bx - hgt * 0.1)
        self.tbox = (tx0, int(hy - 10), int(bx + hgt * 0.1), int(by + 4))
        th, tw = self.tbox[3] - self.tbox[1], self.tbox[2] - self.tbox[0]
        self.tmesh, self.tsheen, _ = organza((th, tw), np.random.default_rng(self.seed + 1))

    def _blade_poly(self, ang):
        L = self.blade
        wmax = L * 0.085
        s = np.linspace(0, 1, 18)
        width = wmax * np.sin(np.clip(s, 0, 1) * np.pi) ** 0.7 * (1 - 0.55 * s)
        pts_l = np.stack([s * L, -width * 0.35], 1)
        pts_r = np.stack([s * L, width * 0.65], 1)
        poly = np.vstack([pts_l, pts_r[::-1]])
        c, s_ = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s_], [s_, c]])
        return poly @ R.T + self.hub

    def draw(self, canvas, t, white):
        if t < self.t:
            return
        k = (t - self.t) * UFPS
        lifted = k < 1
        rot = self.phase + (0 if t < self.t + 0.25 else (np.floor((t - self.t - 0.25) * UFPS) * np.deg2rad(9) * self.speed))
        H, W = canvas.shape[:2]
        # torre
        x0, y0, x1, y1 = self.tbox
        m = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.fillPoly(m, [np.round((self.tower_poly - [x0, y0]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        self._organza(canvas, m, x0, y0, self.tmesh, self.tsheen, white, lifted)
        # aspas
        R = self.R
        cx, cy = int(self.hub[0]), int(self.hub[1])
        bx0, by0 = cx - R, cy - R
        m = np.zeros((2 * R, 2 * R), np.float32)
        for i in range(3):
            poly = self._blade_poly(rot + i * 2 * np.pi / 3 - np.pi / 2)
            cv2.fillPoly(m, [np.round((poly - [bx0, by0]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        nac = np.array([(-10, -5), (14, -6), (16, 5), (-10, 6)]) * self.u * (self.hgt / 200) + self.hub
        cv2.fillPoly(m, [np.round((nac - [bx0, by0]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        self._organza(canvas, np.clip(m, 0, 1), bx0, by0, self.mesh, self.sheen, white, lifted)
        composite(canvas, knot(self.hub, 3.2 * self.u * (self.hgt / 200) + 1.5, "#f4f1ea", self.rng), shadow=0.25)

    def _organza(self, canvas, m, x0, y0, mesh, sheen, white, lifted):
        H, W = canvas.shape[:2]
        h, w = m.shape
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
        a = m[sl]
        # borde con puntada blanca muy fina (el ruedo de la organza)
        edge = np.clip(a - cv2.GaussianBlur(a, (0, 0), 0.9), 0, 1) * 2.2
        reg = canvas[cy0:cy1, cx0:cx1]
        sh = cv2.GaussianBlur(a, (0, 0), 3 if not lifted else 7)
        off = (3, 4) if not lifted else (9, 12)
        sh = cv2.warpAffine(sh, np.float32([[1, 0, off[0]], [0, 1, off[1]]]), (sh.shape[1], sh.shape[0]))
        reg *= (1 - 0.10 * sh)[..., None]
        veil = a * (0.26 + 0.10 * mesh[sl] + 0.22 * sheen[sl])
        veil = np.clip(veil + 0.5 * np.clip(edge, 0, 1), 0, 0.9)
        reg *= (1 - veil * 0.35)[..., None]
        reg += white[None, None, :] * (veil * 0.55)[..., None]


# --- la etiqueta bordada -------------------------------------------------------------------------------
def chalk_line(canvas, strokes, rng, width=2.2, alpha=0.55):
    """Trazo de tiza de sastre: línea seca, granulada, blanca azulada."""
    H, W = canvas.shape[:2]
    pts_all = np.vstack(strokes)
    x0, y0 = int(max(0, pts_all[:, 0].min() - 10)), int(max(0, pts_all[:, 1].min() - 10))
    x1, y1 = int(min(W, pts_all[:, 0].max() + 10)), int(min(H, pts_all[:, 1].max() + 10))
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    for s in strokes:
        p = resample(s, 1.0) - [x0, y0]
        p = p + rng.normal(0, 0.35, p.shape)
        cv2.polylines(m, [np.round(p * 8).astype(np.int32)], False, 1.0, max(1, int(width)), cv2.LINE_AA, shift=3)
    grain = rng.random(m.shape).astype(np.float32)
    m = m * smoothstep(0.35, 0.8, grain) * alpha
    reg = canvas[y0:y1, x0:x1]
    chalk = lin("#eef2f6")
    reg[:] = reg * (1 - m[..., None]) + chalk[None, None, :] * m[..., None]
