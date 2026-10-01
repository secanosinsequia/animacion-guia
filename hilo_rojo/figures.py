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
def _hershey_segs(text, font):
    f = HersheyFonts()
    f.load_default_font(font)
    f.normalize_rendering(100)
    return list(f.lines_for_text(text))


def hershey_strokes(text, font, height, x, y, anchor="left", tracking=0.0, ref=None):
    """Trazos (polilíneas) del texto, con la línea base en y. height: altura total del texto (de la
    línea base al punto más alto); con ref, la altura de ese texto de referencia (p. ej. «H», para que
    varias líneas tengan la misma escala y la misma línea base)."""
    segs = _hershey_segs(text, font)
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
    if ref is not None:
        rs = np.array([p for sg in _hershey_segs(ref, font) for p in sg], np.float64)
        base_y, top_y = float(rs[:, 1].min()), float(rs[:, 1].max())
    k = height / max(1e-6, top_y - base_y)
    xs = np.concatenate([np.asarray(s)[:, 0] for s in strokes])
    w = (xs.max() - xs.min()) * k * (1 + tracking)
    ox = x - (w / 2 if anchor == "center" else (w if anchor == "right" else 0)) - xs.min() * k
    out = []
    for s in strokes:
        s = np.asarray(s, np.float64)
        out.append(np.stack([ox + s[:, 0] * k * (1 + tracking), y - (s[:, 1] - base_y) * k], 1))
    return out, w


_ACC = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N"}


def _two_story_a(xl, xr, base, xh):
    """Una «a» de dos pisos (la de imprenta escolar), en la caja de la «a» redonda: arco arriba, palo a la
    derecha y panza abajo. A tamaño chico no se confunde con una «o»."""
    w = xr - xl
    hook = catmull_rom(np.array([(xl + 0.1 * w, base - 0.82 * xh), (xl + 0.32 * w, base - 0.99 * xh),
                                 (xl + 0.66 * w, base - 1.0 * xh), (xr - 0.04 * w, base - 0.84 * xh),
                                 (xr, base - 0.6 * xh), (xr, base - 0.3 * xh), (xr, base)]), 8)
    bowl = catmull_rom(np.array([(xr, base - 0.47 * xh), (xl + 0.42 * w, base - 0.52 * xh),
                                 (xl + 0.04 * w, base - 0.38 * xh), (xl + 0.02 * w, base - 0.14 * xh),
                                 (xl + 0.34 * w, base + 0.01 * xh), (xr - 0.16 * w, base - 0.07 * xh),
                                 (xr, base - 0.22 * xh)]), 8)
    return [hook, bowl]


def hershey_strokes_es(text, font, height, x, y, anchor="left", ref="H", two_story_a=False):
    """Como hershey_strokes, pero con tildes y eñes (las Hershey solo traen ASCII): se escribe la letra
    base y se le borda el acento encima. two_story_a: la «a» de dos pisos en vez de la redonda."""
    base = "".join(_ACC.get(c, c) for c in text)
    strokes, w = hershey_strokes(base, font, height, 0.0, y, anchor="left", ref=ref)
    ox = x - (w / 2 if anchor == "center" else (w if anchor == "right" else 0))
    out = [s_ + np.array([ox, 0.0]) for s_ in strokes]
    if two_story_a:
        for i, c in enumerate(base):
            if c != "a":
                continue
            pre = base[:i].rstrip()
            x_pre = -1e9
            if pre:
                ps, _ = hershey_strokes(pre, font, height, 0.0, y, anchor="left", ref=ref)
                x_pre = max(float(np.max(p_[:, 0])) for p_ in ps)
            cs, _ = hershey_strokes(base[:i + 1], font, height, 0.0, y, anchor="left", ref=ref)
            mine = [p_ for p_ in cs if float(np.min(p_[:, 0])) > x_pre + 0.5] or cs[-1:]
            for dt in mine:
                for j, o in enumerate(out):
                    if o.shape == dt.shape and np.allclose(o - np.array([ox, 0.0]), dt):
                        out.pop(j)
                        break
            pts = np.vstack(mine)
            out += [s_ + np.array([ox, 0.0]) for s_ in
                    _two_story_a(float(pts[:, 0].min()), float(pts[:, 0].max()), y, y - float(pts[:, 1].min()))]
    for i, c in enumerate(text):
        if c not in _ACC:
            continue
        pre = base[:i].rstrip()
        x_pre = -1e9
        if pre:
            ps, _ = hershey_strokes(pre, font, height, 0.0, y, anchor="left", ref=ref)
            x_pre = max(float(np.max(p_[:, 0])) for p_ in ps)
        cs, _ = hershey_strokes(base[:i + 1], font, height, 0.0, y, anchor="left", ref=ref)
        mine = [p_ for p_ in cs if float(np.min(p_[:, 0])) > x_pre - 0.5] or cs[-1:]
        pts = np.vstack(mine)
        if c in "íÍ":          # la i con tilde no lleva punto (y la tilde va donde iba el punto, no más arriba)
            dots = [p_ for p_ in mine if np.ptp(p_[:, 0]) < 0.15 * height and np.ptp(p_[:, 1]) < 0.15 * height]
            for dt in dots:
                for j, o in enumerate(out):
                    if o.shape == dt.shape and np.allclose(o - np.array([ox, 0.0]), dt):
                        out.pop(j)
                        break
            stem = [p_ for p_ in mine if not any(p_ is d_ for d_ in dots)]
            if stem:
                pts = np.vstack(stem)
        cx = float(pts[:, 0].mean()) + ox
        top = float(pts[:, 1].min())
        h = height
        if c.lower() == "ñ":
            xs = np.linspace(cx - 0.2 * h, cx + 0.2 * h, 9)
            out.append(np.stack([xs, top - 0.18 * h - 0.06 * h * np.sin((xs - cx) / (0.2 * h) * np.pi)], 1))
        else:
            out.append(np.array([[cx - 0.06 * h, top - 0.1 * h], [cx + 0.12 * h, top - 0.32 * h]]))
    return out, w


# --- la aguja ---------------------------------------------------------------------------------------
def _needle_shape(xx, yy, back, tip, width):
    d = tip - back
    L = float(np.hypot(*d)) + 1e-9
    t = d / L
    n = np.array([-t[1], t[0]])
    qx, qy = xx - back[0], yy - back[1]
    s = (qx * t[0] + qy * t[1]) / L
    e = qx * n[0] + qy * n[1]
    r = width / 2 * np.clip(np.where(s < 0.70, 1.0, (1 - s) / 0.30), 0, 1) ** 0.75
    r = np.where((s < -0.01) | (s > 1), 0, r)
    r = np.where(s < 0.035, width / 2 * np.clip(s / 0.035, 0, 1) ** 0.5, r)     # cabeza redondeada
    a = np.clip(r - np.abs(e) + 0.5, 0, 1) * ((s >= -0.01) & (s <= 1))           # (nada más allá de la punta)
    return a, s, e, r, L


def needle_sprite(back, tip, width, hide_from=None, lift=1.0):
    """Aguja de acero (de lana, grande): brillo a lo largo, ojo abierto y sombra que nace en la punta
    (una aguja clavada toca la tela solo ahí). hide_from: fracción desde la cual la punta ya entró en
    la tela. lift: cuánto se separa de la tela la cabeza (sombra más lejos y difusa)."""
    back, tip = np.asarray(back, np.float64), np.asarray(tip, np.float64)
    off = np.array([0.55, 0.75]) * width * 2.6 * lift
    pad = width * 3 + 6 + float(np.abs(off).max())
    x0 = int(np.floor(min(back[0], tip[0]) - pad))
    y0 = int(np.floor(min(back[1], tip[1]) - pad))
    x1 = int(np.ceil(max(back[0], tip[0]) + pad))
    y1 = int(np.ceil(max(back[1], tip[1]) + pad))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    a, s, e, r, L = _needle_shape(xx, yy, back, tip, width)
    # ojo: ranura alargada cerca de la cabeza
    eye = np.clip(width * 0.2 - np.abs(e) + 0.5, 0, 1) * np.clip(np.minimum(s - 0.035, 0.105 - s) * L + 0.5, 0, 1)
    a = a * (1 - 0.95 * eye)
    if hide_from is not None:
        a = a * np.clip((hide_from - s) * L / 1.5 + 0.5, 0, 1)
    dz = np.clip(e / np.maximum(r, 1e-3), -1, 1)
    # acero pulido: un reflejo fino y muy blanco (la ventana), uno ancho y tibio, y el canto oscuro
    streak = np.exp(-((dz + 0.42) / 0.09) ** 2) * (0.75 + 0.25 * np.sin(s * 9.0)) * 2.2
    spec = np.exp(-((dz + 0.38) / 0.22) ** 2) * 0.7 + np.exp(-((dz - 0.55) / 0.28) ** 2) * 0.16
    base = 0.18 + 0.26 * (1 - np.abs(dz))
    val = np.clip(base + spec, 0, 1.5) * (1 - 0.6 * np.clip((np.abs(dz) - 0.7) / 0.3, 0, 1))
    steel = lin("#c3c9cf")
    env = np.where(dz[..., None] < 0, np.array([1.04, 1.0, 0.94]), np.array([0.92, 0.97, 1.05]))  # sala tibia/cielo frío
    rgb = steel[None, None, :] * val[..., None] * env + np.array([0.012, 0.014, 0.018])[None, None, :]
    rgb = rgb + np.array([1.0, 0.98, 0.95])[None, None, :] * np.clip(streak, 0, 1.6)[..., None] * 0.55
    a = a.astype(np.float32)
    # sombra: la punta toca la tela (o la sombra entra con ella), la cabeza está en el aire
    sh_tip = tip if hide_from is None else back + (tip - back) * hide_from
    sa, ss_, _, _, _ = _needle_shape(xx, yy, back + off, sh_tip + off * (0.15 if hide_from is not None else 0.55),
                                    width)
    sh = cv2.GaussianBlur(sa.astype(np.float32), (0, 0), 1.2 + 1.6 * lift)
    return Sprite(x0, y0, (rgb * a[..., None]).astype(np.float32), a, sh * 0.75)


# --- casas, árboles, sol, nubes ----------------------------------------------------------------------
def _rot(poly, c, ang):
    poly = np.asarray(poly, np.float64)
    cs, sn = np.cos(ang), np.sin(ang)
    d = poly - c
    return np.stack([c[0] + d[:, 0] * cs - d[:, 1] * sn, c[1] + d[:, 0] * sn + d[:, 1] * cs], 1)


def house(cx, base, w, h, rng, t, wall="#c0392b", roof="#6e6a66", wall_kind="plain", roof_kind="cord",
          windows=1, door_side=0.0, u=1.0, vary=True, roof_style="gable", patch=None):
    """Casa de retazos. Con vary, cada casa sale distinta: proporciones, techo, puerta, ventanas y un
    leve giro (cosida a mano, no calcada). roof_style: «gable» (dos aguas) o «shed» (media agua, más alta
    de un lado); patch: color de un parche cosido sobre el techo."""
    parts = []
    if vary:
        w *= rng.uniform(0.88, 1.12)
        h *= rng.uniform(0.86, 1.14)
        door_side = float(np.clip(door_side + rng.normal(0, 0.18), -0.6, 0.6))
        windows = int(rng.choice([1, 2], p=[0.55, 0.45]))
    ang = np.deg2rad(rng.uniform(-3, 3)) if vary else 0.0
    c0 = np.array([cx, base])
    x0, x1 = cx - w / 2, cx + w / 2
    top = base - h
    jit = (lambda: rng.normal(0, 0.03 * w, 2)) if vary else (lambda: np.zeros(2))   # cortado a tijera
    wall_poly = [np.array(p_) + jit() for p_ in [(x0, base), (x0, top), (x1, top), (x1, base)]]
    parts.append(Piece(_rot(wall_poly, c0, ang), wall, rng, kind=wall_kind, t_place=t,
                       stitch=("#3d2a1e", 7 * u, 5 * u, 1.6 * u), fabric_scale=u))
    ov = w * (rng.uniform(0.06, 0.16) if vary else 0.12)
    rise = h * (rng.uniform(0.45, 0.80) if vary else 0.62)
    ridge = w * (rng.uniform(0.0, 0.14) if vary else 0.08)
    if roof_style == "shed":                        # media agua: una sola caída, alta del lado de la puerta
        ov *= 0.4                                   # (con alero corto)
        hr_ = rise * 0.62
        if door_side > 0:
            pts_ = [(x0 - ov, top + 4 * u), (x0 - ov, top - 4 * u), (x1 + ov, top - hr_), (x1 + ov, top - hr_ + 12 * u)]
        else:
            pts_ = [(x0 - ov, top - hr_ + 12 * u), (x0 - ov, top - hr_), (x1 + ov, top - 4 * u), (x1 + ov, top + 4 * u)]
        roof_poly = [np.array(p_) + jit() for p_ in pts_]
        # la pared sube hasta meterse bajo el techo (sin rendija por donde se vea lo de atrás)
        if door_side > 0:
            wall_up = [(x0, top + 2 * u), (x0, top - 3 * u), (x1, top - hr_ + 5 * u), (x1, top + 2 * u)]
        else:
            wall_up = [(x0, top + 2 * u), (x0, top - hr_ + 5 * u), (x1, top - 3 * u), (x1, top + 2 * u)]
        parts.append(Piece(_rot([np.array(p_) + jit() * 0.3 for p_ in wall_up], c0, ang), wall, rng, kind=wall_kind,
                           t_place=t, stitch=None, fabric_scale=u, margin=6, rough=0.4))
    else:
        roof_poly = [np.array(p_) + jit() for p_ in [(x0 - ov, top + 4 * u), (cx - ridge, top - rise),
                                                     (cx + ridge, top - rise), (x1 + ov, top + 4 * u)]]
    parts.append(Piece(_rot(roof_poly, c0, ang), roof, rng, kind=roof_kind, t_place=t,
                       stitch=("#2b2622", 7 * u, 5 * u, 1.6 * u), fabric_scale=u,
                       angle=(np.pi / 2 if roof_kind == "cord" else rng.uniform(-0.3, 0.3)) + ang))
    if patch is not None and roof_style != "shed":   # un parche cosido encima (se le voló una plancha)
        px_ = cx + rng.uniform(-0.25, 0.1) * w
        py_ = top - rise * 0.38
        pw_, ph_ = w * rng.uniform(0.16, 0.22), rise * rng.uniform(0.26, 0.34)
        pp_ = [np.array(p_) + rng.normal(0, 0.06 * pw_, 2) for p_ in
               [(px_ - pw_ / 2, py_ - ph_ / 2), (px_ + pw_ / 2, py_ - ph_ / 2), (px_ + pw_ / 2, py_ + ph_ / 2),
                (px_ - pw_ / 2, py_ + ph_ / 2)]]
        parts.append(Piece(_rot(pp_, c0, ang), patch, rng, kind="plain", t_place=t,
                           stitch=("#2b2622", 5 * u, 4 * u, 1.3 * u), fabric_scale=u, margin=6, inset=3))
    dw, dh = w * rng.uniform(0.2, 0.27), h * rng.uniform(0.46, 0.58)
    dx = cx + door_side * w * 0.36
    door_poly = [(dx - dw / 2, base), (dx - dw / 2, base - dh), (dx + dw / 2, base - dh), (dx + dw / 2, base)]
    parts.append(Piece(_rot(door_poly, c0, ang), "#5a3a22", rng, kind="felt", t_place=t, stitch=None, inset=3,
                       margin=6))
    wins = []
    # ventanas en el lado opuesto a la puerta (o a ambos lados si hay dos)
    slots = [cx - door_side * w * 0.30] if windows == 1 else [x0 + w * 0.22, x1 - w * 0.22]
    for wx in slots:
        if abs(wx - dx) < dw * 0.9:
            wx = dx + np.sign(wx - dx + 1e-6) * dw * 1.1
        ww = w * rng.uniform(0.15, 0.21)
        wh = ww * rng.uniform(0.9, 1.25)
        wy = top + h * rng.uniform(0.16, 0.26)
        poly = _rot([np.array(p_) + rng.normal(0, 0.07 * ww, 2) for p_ in
                     [(wx - ww / 2, wy), (wx + ww / 2, wy), (wx + ww / 2, wy + wh), (wx - ww / 2, wy + wh)]], c0, ang)
        parts.append(Piece(poly, "#2c3440", rng, kind="felt", t_place=t, stitch=None, margin=6, rough=0.6))
        # la ventana encendida: relleno de satín amarillo (se cose encima cuando llega la alerta)
        lit = satin_fill(inset_poly_simple(poly, 1.3 * u), np.deg2rad(8) + ang, 2.1 * u, 2.4 * u, "#f3c33a", rng)
        wins.append(lit)
    door = _rot([(dx, base - dh * 0.45)], c0, ang)[0]
    return parts, wins, (door[0], door[1])


def inset_poly_simple(poly, d):
    poly = np.asarray(poly, np.float64)
    c = poly.mean(axis=0)
    v = poly - c
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return c + v * np.clip((n - d) / n, 0.2, 1.0)


def round_tree(cx, base, r, rng, t, color=None, u=1.0):
    """Árbol de fieltro: cada uno con su copa (más ancha o más alta, a veces de dos lóbulos) y su verde."""
    if color is None:
        color = str(rng.choice(["#3f6b3c", "#4b7438", "#35613f", "#56793f", "#2f5a3a"]))
    r *= rng.uniform(0.85, 1.15)
    trunk = Yarn([(cx, base), (cx + rng.normal(0, 1.5), base - r * 1.2)], 4.5 * u, "#5b3b22", rng, fuzz=0.6)
    ax, ay = rng.uniform(0.85, 1.12), rng.uniform(0.8, 1.08)
    crowns = [Piece(ellipse_poly(cx, base - r * 1.55, r * ax, r * 0.92 * ay, wobble=0.08, rng=rng,
                                 rot=rng.normal(0, 0.15)), color, rng, kind="felt", t_place=t,
                    stitch=("#26401f", 6 * u, 5 * u, 1.5 * u), fabric_scale=u)]
    if rng.random() < 0.35:                         # a veces un segundo lóbulo, de otro verde
        c2 = str(rng.choice(["#46703a", "#3a6641", "#5b8043"]))
        crowns.append(Piece(ellipse_poly(cx + rng.choice([-1, 1]) * r * 0.55, base - r * 1.25, r * 0.62, r * 0.55,
                                         wobble=0.08, rng=rng), c2, rng, kind="felt", t_place=t,
                            stitch=("#26401f", 6 * u, 5 * u, 1.5 * u), fabric_scale=u))
    return [trunk], crowns


def araucaria(cx, base, hgt, rng, t, u=1.0):
    """La araucaria (pehuén): tronco alto y desnudo y, arriba, la copa de paraguas: ancha, casi plana
    encima, con las puntas de las ramas curvadas hacia arriba por debajo; bajo la copa asoman las ramas."""
    top = base - hgt
    trunk = Yarn([(cx, base), (cx + rng.normal(0, 1.2) * u, top + hgt * 0.2)], 5.0 * u, "#4a3220", rng, fuzz=0.5)
    w = hgt * rng.uniform(0.66, 0.8)
    h = hgt * rng.uniform(0.2, 0.26)
    lean = rng.normal(0, 0.04) * w
    upper = [(cx + lean * (1 - abs(c_)) + c_ * w / 2, top + h * 0.62 - h * 0.62 * (1 - c_ * c_) ** 0.35)
             for c_ in np.linspace(-1, 1, 15)]
    n_tip = int(rng.integers(5, 7))
    lower = []
    for k in range(n_tip * 2 + 1):                 # borde de abajo festoneado: puntas que suben
        c_ = 1 - 2 * k / (n_tip * 2)
        y_ = top + h * (0.62 + (0.32 if k % 2 == 0 else 0.08) * (1 - 0.35 * abs(c_)))
        lower.append((cx + c_ * w / 2 * (0.98 if k % 2 == 0 else 0.9), y_))
    crown = Piece(upper + lower, "#23452e", rng, kind="felt", t_place=t, stitch=("#132a18", 5 * u, 4 * u, 1.3 * u),
                  fabric_scale=u)
    arms = []
    for sgn in (-1, 1):                            # dos ramas que salen del tronco hacia las puntas
        p0 = (cx, top + h * 1.4)
        p1 = (cx + sgn * w * 0.28, top + h * 1.05)
        p2 = (cx + sgn * w * 0.4, top + h * 0.8)
        arms.append(Yarn(catmull_rom(np.array([p0, p1, p2]), 6), 3.2 * u, "#3b2a1c", rng, fuzz=0.4))
    return [trunk] + arms, [crown]


def sun(cx, cy, r, rng, t, u=1.0):
    disc = Piece(ellipse_poly(cx, cy, r, r, n=48), "#f1b82d", rng, kind="felt", t_place=t,
                 stitch=("#c8761b", 6 * u, 5 * u, 1.8 * u), fabric_scale=u)
    rays = []
    rs = np.random.default_rng(int(abs(cx * 7 + cy * 3)) % (2 ** 31))    # (su propio azar: la tela no cambia)
    for k in range(14):
        a = k * 2 * np.pi / 14 + rng.normal(0, 0.04) + rs.normal(0, 0.07)
        r0_ = r * rs.uniform(1.12, 1.24)                 # cosidos a ojo: cada rayo de su largo y su giro
        r1_ = r * (1.55 + 0.12 * (k % 2)) * rs.uniform(0.86, 1.14)
        tw = rs.normal(0, 0.06)
        p0 = (cx + np.cos(a) * r0_, cy + np.sin(a) * r0_)
        p1 = (cx + np.cos(a + tw) * r1_, cy + np.sin(a + tw) * r1_)
        rays.append(Stitch.render(p0, p1, 2.8 * u * rs.uniform(0.85, 1.15), "#e07b1a", rng))
    return disc, rays


def cloud(cx, cy, w, h, rng, t, u=1.0):
    """Nube de retazo: unión de lóbulos (contorno real, cortado a tijera)."""
    pad = 20
    W_, H_ = int(w * 1.3 + 2 * pad), int(h * 1.6 + 2 * pad)
    m = np.zeros((H_, W_), np.uint8)
    ox, oy = cx - W_ / 2, cy - H_ / 2
    base_y = H_ / 2 + h * 0.35
    n = int(rng.integers(3, 6))
    for k in range(n):
        fx = (k - (n - 1) / 2) / max(1, (n - 1) / 2) + rng.normal(0, 0.12)
        rx = w * (0.14 + 0.10 * (1 - abs(fx))) * rng.uniform(0.8, 1.2)
        ry = h * (0.24 + 0.34 * (1 - abs(fx))) * rng.uniform(0.75, 1.15)
        ry = min(ry, rx * 1.05)                     # lóbulos anchos: nubes, no botellas
        ccx = W_ / 2 + fx * w * 0.34
        ccy = base_y - ry * rng.uniform(0.6, 0.95)
        cv2.ellipse(m, (int(ccx), int(ccy)), (int(rx), int(ry)), float(rng.uniform(-12, 12)), 0, 360, 255, -1,
                    cv2.LINE_AA)
    cv2.rectangle(m, (int(W_ / 2 - w * 0.45), int(base_y - h * 0.25)), (int(W_ / 2 + w * 0.45), int(base_y)), 255, -1)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    c = resample(np.vstack([c, c[:1]]), 6.0)[:-1] + [ox, oy]
    return Piece(c, "#f2eee6", rng, kind="felt", t_place=t, stitch=("#b9b3a6", 6 * u, 5 * u, 1.5 * u),
                 fabric_scale=u)


# --- muñecas de tela (los «monitos» de la arpillera) -------------------------------------------------
def doll(fx, base, hgt, rng, t, dress="#2f6fa6", dress_kind="dots", dress2="#f2eadb", skin="#c68e67",
         hair="#1f1712", hand_y=None, arms="down", u=1.0, lean=0.0, hands=None, kind="woman", target=None):
    """Muñeca cosida (sin carita, como en las arpilleras). kind: woman, man (camisa, pantalón y sombrero),
    child (más chica, trenzas), elder (chal y bastón). arms: down, hip, wave, up, point (hacia target).
    hands: puntos (izq, der) para poses especiales (p. ej. tirando de un hilo)."""
    rg = np.random.default_rng(int(fx * 7 + base * 3) % (1 << 30))     # la misma muñeca en todas sus poses
    if kind == "child":
        hgt *= 0.74
    head_r = hgt * (0.17 if kind == "child" else 0.15) * rg.uniform(0.92, 1.08)
    neck_y = base - hgt * 0.68
    hip_y = base - hgt * (0.22 if kind != "man" else 0.40)
    sw = rg.uniform(0.21, 0.27)
    extra_below, extra_mid, extra_top, extra_front = [], [], [], []
    if kind == "man":
        body_poly = [(fx - hgt * 0.115, neck_y), (fx + hgt * 0.115, neck_y), (fx + hgt * 0.15, hip_y + hgt * 0.03),
                     (fx - hgt * 0.15, hip_y + hgt * 0.03)]
        pc = rg.choice(["#34384a", "#4a3a2c", "#2f3f35"])
        for sgn in (-1, 1):
            leg = [(fx + sgn * hgt * 0.015, hip_y), (fx + sgn * hgt * 0.14, hip_y), (fx + sgn * hgt * 0.12, base - 3 * u),
                   (fx + sgn * hgt * 0.035, base - 3 * u)]
            extra_below.append(Piece(leg, pc, rg, kind="cord", t_place=t, stitch=("#1c1a17", 4 * u, 3 * u, 1.1 * u),
                                     fabric_scale=u * 0.7, margin=6, inset=2.5, rough=0.4, shadow=0.7))
    else:
        body_poly = [(fx - hgt * 0.10, neck_y), (fx + hgt * 0.10, neck_y), (fx + hgt * sw, hip_y + hgt * 0.02),
                     (fx - hgt * sw, hip_y + hgt * 0.02)]
    body = Piece(body_poly, dress, rg, kind=dress_kind, color2=dress2, t_place=t,
                 stitch=("#2a2019", 5 * u, 4 * u, 1.3 * u), fabric_scale=u * 0.8, margin=8, inset=3.5, puff=1.8,
                 shadow=0.75)
    if kind == "elder":
        sh_poly = [(fx - hgt * 0.17, neck_y + hgt * 0.02), (fx + hgt * 0.17, neck_y + hgt * 0.02),
                   (fx + hgt * 0.02, neck_y + hgt * 0.30), (fx - hgt * 0.02, neck_y + hgt * 0.30)]
        extra_mid.append(Piece(sh_poly, rg.choice(["#6b3f5e", "#3f5b6b", "#7a5a2e"]), rg, kind="flannel",
                               color2="#2e2a2a", t_place=t, stitch=("#231c18", 4 * u, 3 * u, 1.1 * u),
                               fabric_scale=u * 0.7, margin=6, inset=2.5, rough=0.5, shadow=0.7))
        hair = "#b8b1a6"
    lx = rg.uniform(0.06, 0.09)
    legs = [] if kind == "man" else [
        Yarn([(fx - hgt * lx, hip_y), (fx - hgt * (lx + 0.01), base - 2)], 3.2 * u, "#2a2019", rg, fuzz=0.4),
        Yarn([(fx + hgt * lx, hip_y), (fx + hgt * (lx + 0.01), base - 2)], 3.2 * u, "#2a2019", rg, fuzz=0.4)]
    sh_l = np.array([fx - hgt * 0.10, neck_y + hgt * 0.05])
    sh_r = np.array([fx + hgt * 0.10, neck_y + hgt * 0.05])
    head_top = neck_y - head_r * 1.9
    if hands is not None:
        hand_l, hand_r = np.asarray(hands[0], np.float64), np.asarray(hands[1], np.float64)
    else:
        hy = hand_y if hand_y is not None else neck_y + hgt * 0.26
        down_l, down_r = np.array([fx - hgt * 0.27, hy]), np.array([fx + hgt * 0.27, hy])
        up_l = np.array([fx - hgt * 0.30, head_top - hgt * 0.02])
        up_r = np.array([fx + hgt * 0.30, head_top - hgt * 0.02])
        hip_l = np.array([fx - hgt * 0.20, hip_y - hgt * 0.02])
        hip_r = np.array([fx + hgt * 0.20, hip_y - hgt * 0.02])
        if arms == "point" and target is not None:
            # un brazo bien estirado hacia el cerro (la amenaza); el otro, abajo
            d_ = np.asarray(target, np.float64) - np.array([fx, neck_y])
            d_ = d_ / (np.linalg.norm(d_) + 1e-9)
            if d_[0] >= 0:
                hand_l, hand_r = down_l, sh_r + d_ * hgt * 0.5
            else:
                hand_l, hand_r = sh_l + d_ * hgt * 0.5, down_r
        else:
            hand_l, hand_r = {"down": (down_l, down_r), "hip": (hip_l, hip_r), "wave": (down_l, up_r),
                              "up": (up_l, up_r), "point": (down_l, up_r)}[arms]

    def arm(sh, hand, side):
        mid = (sh + hand) / 2 + np.array([side * 5 * u, 3 * u])
        if hand[1] < sh[1] - hgt * 0.1:           # brazo arriba: el codo hacia afuera
            mid = (sh + hand) / 2 + np.array([side * hgt * 0.06, hgt * 0.02])
        if arms == "point" and np.hypot(*(hand - sh)) > hgt * 0.42:
            mid = (sh + hand) / 2                   # apunta: brazo recto
        return Yarn(catmull_rom(np.array([sh, mid, hand]), 6), 3.4 * u, skin, rg, fuzz=0.3)

    arm_l, arm_r = arm(sh_l, hand_l, -1), arm(sh_r, hand_r, 1)
    if kind == "elder":
        hb = hand_r if hand_r[1] > neck_y else hand_l
        extra_front.append(Yarn([hb, (hb[0] + hgt * 0.04, base)], 3.0 * u, "#6b4a2a", rg, fuzz=0.3))
    hc = (fx, neck_y - head_r * 0.92)
    head = yarn_head(hc, head_r, skin, rg, u)
    hair_y = []
    nh = int(rg.integers(7, 11))
    style = {"man": 1, "child": 0, "elder": 2}.get(kind, int(rg.integers(0, 3)))
    for k in range(nh):
        a = np.pi * (0.95 + 1.1 * k / (nh - 1))
        p = (hc[0] + np.cos(a) * head_r * 0.92, hc[1] + np.sin(a) * head_r * 0.92)
        L = head_r * (1.25 if k in (0, nh - 1) else 1.12) * (0.9 if style == 1 else 1.0)
        drop = head_r * (1.25 if kind == "child" else 0.9 if style == 0 else 0.3) if k in (0, nh - 1) else 0
        q = (hc[0] + np.cos(a) * L, hc[1] + np.sin(a) * L + drop)
        hair_y.append(Yarn(catmull_rom(np.array([p, ((p[0] + q[0]) / 2 + rg.normal(0, 1), (p[1] + q[1]) / 2 - 2 * u),
                                                 q]), 5), 3.4 * u, hair, rg, fuzz=0.8))
        if kind == "child" and k in (0, nh - 1):
            extra_top.append(knot(q, 3.2 * u, "#c3241c" if k == 0 else "#2f6fa6", rg))
    if style == 2:
        hair_y.append(Yarn(np.array([(hc[0] - 4 * u, hc[1] - head_r * 1.05), (hc[0] + 4 * u, hc[1] - head_r * 1.25)]),
                           7 * u, hair, rg, fuzz=0.9))
    if kind == "man":
        brim = ellipse_poly(hc[0], hc[1] - head_r * 0.72, head_r * 1.55, head_r * 0.28, n=28)
        crown = [(hc[0] - head_r * 0.8, hc[1] - head_r * 0.72), (hc[0] - head_r * 0.62, hc[1] - head_r * 1.55),
                 (hc[0] + head_r * 0.62, hc[1] - head_r * 1.55), (hc[0] + head_r * 0.8, hc[1] - head_r * 0.72)]
        hc_ = rg.choice(["#5b4636", "#2e2b28", "#6d5a3a"])
        extra_top.append(Piece(crown, hc_, rg, kind="felt", t_place=t, stitch=None, margin=6, rough=0.3, shadow=0.7))
        extra_top.append(Piece(brim, hc_, rg, kind="felt", t_place=t, stitch=None, margin=6, rough=0.3, shadow=0.7))
    shoes = [knot((fx - hgt * (lx + 0.01), base - 1), 2.4 * u, "#1a1410", rg),
             knot((fx + hgt * (lx + 0.01), base - 1), 2.4 * u, "#1a1410", rg)]
    if kind == "man":
        shoes = [knot((fx - hgt * 0.08, base - 1), 3.0 * u, "#1a1410", rg),
                 knot((fx + hgt * 0.08, base - 1), 3.0 * u, "#1a1410", rg)]
    return dict(t=t, body=body, legs=legs, arms=[arm_l, arm_r], head=head, hair=hair_y, eyes=[], shoes=shoes,
                hands=(hand_l, hand_r), foot=np.array([fx, base]), lean=lean, below=extra_below, mid=extra_mid,
                top=extra_top, front=extra_front)


def yarn_head(c, r, color, rng, u=1.0):
    """Cabeza de muñeca de arpillera: una bola de lana enrollada (no una esfera de computador)."""
    cx, cy = c
    pad = r * 1.4 + 6
    x0, y0 = int(cx - pad), int(cy - pad)
    n = int(2 * pad) + 1
    yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n].astype(np.float32) + 0.5
    d = np.hypot(xx - cx, yy - cy)
    disc = np.clip(r - d + 0.5, 0, 1).astype(np.float32)
    layer = np.zeros((n, n, 3), np.float32)
    la = np.zeros((n, n), np.float32)
    # vueltas de lana: arcos cruzados en varias direcciones, como un ovillo
    wy = max(2.2 * u, r * 0.22)
    for k in range(9):
        ang = rng.uniform(0, np.pi)
        off = rng.uniform(-0.75, 0.75) * r
        t = np.linspace(-1, 1, 24)
        ca, sa = np.cos(ang), np.sin(ang)
        span = np.sqrt(max(r * r - off * off, 1)) * 1.02
        pts = np.stack([cx + ca * t * span - sa * off, cy + sa * t * span + ca * off], 1)
        ytmp = Yarn(pts, wy, color, rng, fuzz=0.25, chunk=64)
        for ch in ytmp.chunks:
            sh_ = ch.a.shape
            xs0, ys0 = ch.x0 - x0, ch.y0 - y0
            xa, ya = max(0, xs0), max(0, ys0)
            xb, yb = min(n, xs0 + sh_[1]), min(n, ys0 + sh_[0])
            if xb <= xa or yb <= ya:
                continue
            sl = (slice(ya - ys0, yb - ys0), slice(xa - xs0, xb - xs0))
            a_ = ch.a[sl]
            layer[ya:yb, xa:xb] = layer[ya:yb, xa:xb] * (1 - a_[..., None]) + ch.rgb[sl]
            la[ya:yb, xa:xb] = la[ya:yb, xa:xb] * (1 - a_) + a_
    base = lin(color)
    under = base[None, None, :] * 0.55                           # el relleno se ve entre vueltas
    rgb = layer + under * (1 - la)[..., None]
    # volumen suave (bola rellena, luz rasante), sin brillo de plástico
    dx, dy = (xx - cx) / r, (yy - cy) / r
    nz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
    lam = np.clip(-0.64 * dx - 0.56 * dy + 0.52 * nz, 0, 1)
    rgb *= (0.62 + 0.55 * lam)[..., None]
    rgb *= disc[..., None]
    sh = cv2.GaussianBlur(disc, (0, 0), r * 0.35)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, r * 0.3], [0, 1, r * 0.38]]), (n, n))
    return Sprite(x0, y0, rgb.astype(np.float32), disc, sh)


def doll_layer(d, lean=0.0):
    """Toda la muñeca en una capa premultiplicada (para posarla o cambiarla de pose cuadro a cuadro)."""
    sp_of = lambda x: x.sprite if isinstance(x, Piece) else x
    parts = []
    for y in d["legs"]:
        parts += y.chunks
    parts += [sp_of(x) for x in d.get("below", [])]
    parts.append(d["body"].sprite)
    parts += [sp_of(x) for x in d.get("mid", [])]
    parts += d["shoes"]
    parts.append(d["head"])
    for y in d["hair"]:
        parts += y.chunks
    parts += [sp_of(x) for x in d.get("top", [])]
    for y in d["arms"] + d.get("front", []):
        parts += y.chunks
    x0 = min(p.x0 for p in parts) - 14
    y0 = min(p.y0 for p in parts) - 14
    x1 = max(p.x0 + p.a.shape[1] for p in parts) + 18
    y1 = max(p.y0 + p.a.shape[0] for p in parts) + 18
    h, w = y1 - y0, x1 - x0
    rgb = np.zeros((h, w, 3), np.float32)
    a = np.zeros((h, w), np.float32)
    for p in parts:
        shadow = 0.75 if p is d["body"].sprite else 0.5
        _over_local(rgb, a, p, x0, y0, shadow)
    if lean:
        fx, fy = d["foot"] - np.array([x0, y0])
        M = cv2.getRotationMatrix2D((float(fx), float(fy)), float(np.rad2deg(-lean)), 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR)
    # volumen: la muñeca es rellena (más clara arriba a la izquierda) y está cosida encima: hace sombra
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx = (xx - w / 2) / max(1.0, w / 2)
    gy = (yy - h / 2) / max(1.0, h / 2)
    rgb = rgb * np.clip(1.0 - 0.11 * gx - 0.06 * gy, 0.82, 1.16)[..., None]
    k = max(1.0, h / 60)
    sh = cv2.GaussianBlur(a, (0, 0), 1.6 * k)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, 2.6 * k], [0, 1, 3.2 * k]]), (w, h))
    return Sprite(x0, y0, rgb.astype(np.float32), a, (sh * 0.85).astype(np.float32))


def _over_local(rgb, a, sp, ox, oy, shadow):
    H, W = a.shape
    h, w = sp.a.shape
    x0, y0 = sp.x0 - ox, sp.y0 - oy
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    reg_rgb, reg_a = rgb[cy0:cy1, cx0:cx1], a[cy0:cy1, cx0:cx1]
    if sp.sh is not None and shadow > 0:
        s_ = shadow * sp.sh[sl] * (1 - sp.a[sl])
        reg_rgb *= (1 - s_)[..., None]
        reg_a += s_ * (1 - reg_a)
    sa = sp.a[sl]
    reg_rgb *= (1 - sa)[..., None]
    reg_rgb += sp.rgb[sl]
    reg_a *= (1 - sa)
    reg_a += sa


def cordillera(W, H, y_base, peaks, rng, u=1.0):
    """La cordillera de los Andes al fondo: fieltro gris azulado con nieve de fieltro blanco."""
    X = lambda f: f * W
    Y = lambda f: f * H
    pts = [(X(-0.02), Y(y_base))]
    for (px, py, pw) in peaks:
        pts += [(X(px - pw), Y(y_base) - (Y(y_base) - Y(py)) * 0.35), (X(px), Y(py)),
                (X(px + pw), Y(y_base) - (Y(y_base) - Y(py)) * 0.4)]
    pts.append((X(1.02), Y(y_base)))
    line = catmull_rom(np.array(pts), 3)
    poly = np.vstack([line, [(X(1.02), Y(y_base) + 60 * u), (X(-0.02), Y(y_base) + 60 * u)]])
    parts = [Piece(poly, "#8e9db4", rng, kind="flannel", color2="#7d8ca3", stitch=("#4f5a6b", 8 * u, 6 * u, 1.6 * u),
                   fabric_scale=u)]
    for (px, py, pw) in peaks:
        top = np.array([X(px), Y(py)])
        dep = (Y(y_base) - Y(py)) * rng.uniform(0.28, 0.4)
        wl, wr = X(pw) * 0.42, X(pw) * 0.42
        snow = [top + (0, -2 * u), top + (wr, dep * 0.95), top + (wr * 0.55, dep * 0.7), top + (wr * 0.2, dep * 1.05),
                top + (-wl * 0.25, dep * 0.75), top + (-wl * 0.6, dep * 1.0), top + (-wl, dep * 0.9)]
        parts.append(Piece(snow, "#f2f0ea", rng, kind="felt", stitch=("#b8b6ae", 5 * u, 4 * u, 1.2 * u),
                           fabric_scale=u, margin=8))
    return parts


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
