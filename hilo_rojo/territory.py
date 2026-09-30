"""El territorio de la arpillera principal (un valle del centro-sur), en coordenadas normalizadas.

Se hornea una sola vez: color (con luz rasante) + transmisión (para el contraluz). Devuelve también
los puntos de las huellas, el camino del hilo por el revés y los lugares clave para la animación.
"""
import cv2
import numpy as np

from satc_intro.color import lin
from satc_intro.geometry import catmull_rom, ridge, resample
from .figures import (house, round_tree, araucaria, sun, cloud, doll, doll_layer, cordillera, hershey_strokes,
                      _yarn_full)
from .pieces import Piece, ellipse_poly
from .textile import burlap
from .thread import Sprite, Stitch, Yarn, composite, composite_T, knot, running_stitch, blanket_stitch


def stencil_mask(W, H, u, y0=0.30, mirror=False, cx=0.58):
    """Estarcido del saco harinero. La tinta quedó en la cara de adelante, bajo los retazos: a contraluz
    se lee al derecho (mirror=True: visto desde atrás)."""
    import cairo
    from satc_intro.typography import Font, Word
    surf = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    ctx = cairo.Context(surf)
    f = Font("Anton-Regular.ttf")
    lines = [("HARINA", 132), ("MOLINO LA ESPERANZA", 50), ("FLOR · 50 KG", 50)]
    y = H * y0
    ctx.save()
    if mirror:
        ctx.translate(W, 0)
        ctx.scale(-1, 1)
    for text, cap in lines:
        cap *= u
        w = Word(text, f, cap, W * cx, y + cap, 0.08, "center")
        for L in w.letters:
            L.draw(ctx)
        y += cap * 1.35
    ctx.set_source_rgba(0, 0, 0, 1)
    ctx.fill()
    ctx.restore()
    surf.flush()
    buf = np.ndarray((H, surf.get_stride()), np.uint8, buffer=surf.get_data())[:, :W].astype(np.float32) / 255
    # tinta de estarcido: bordes de plantilla (algo corrida), gastada
    buf = cv2.dilate(buf, np.ones((3, 3), np.uint8), iterations=max(1, int(round(1.5 * u))))
    rng = np.random.default_rng(9)
    wear = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 2.0)
    m = buf * (0.45 + 0.55 * (wear > 0.47))
    return cv2.GaussianBlur(m, (0, 0), 1.2)


# Composición: fracciones del ancho/alto (las medidas de objetos van en u, así no se deforman).
LAND = dict(
    sky=0.60, sun=(0.90, 0.15), clouds=[(0.14, 0.12, 0.13, 0.07), (0.30, 0.225, 0.10, 0.055), (0.405, 0.08, 0.065, 0.04)],
    birds=[(0.56, 0.30, 1.0), (0.60, 0.27, 0.8), (0.63, 0.31, 0.7)],
    hills=[("#8a6f55", "flannel", "#6f5642", 0.46, [(0.08, 0.40), (0.30, 0.45), (0.52, 0.41)]),
           ("#6f8f55", "plain", None, 0.50, [(0.35, 0.44), (0.58, 0.47), (0.70, 0.45)]),
           ("#9a8a4a", "gingham", "#7c6d35", 0.53, [(0.02, 0.50), (0.18, 0.47), (0.40, 0.52)])],
    hills_bottom=0.75,
    tower=(0.80, 0.41), tower_hill=[(0.55, 0.62), (-0.14, 0.06), (0.0, 0.0), (0.12, 0.04), (None, 0.07)],
    tower_hill_bottom=0.78,
    forest=(0.06, 0.035, 0.47),
    rows=[[(0.00, 0.56), (0.28, 0.54), (0.36, 0.64), (0.00, 0.70)],
          [(0.28, 0.54), (0.62, 0.55), (0.66, 0.66), (0.36, 0.64)],
          [(0.62, 0.55), (1.00, 0.52), (1.00, 0.70), (0.66, 0.66)],
          [(0.00, 0.70), (0.36, 0.64), (0.40, 0.82), (0.00, 0.86)],
          [(0.36, 0.64), (0.66, 0.66), (0.70, 0.84), (0.40, 0.82)],
          [(0.66, 0.66), (1.00, 0.70), (1.00, 0.86), (0.70, 0.84)],
          [(0.00, 0.86), (0.40, 0.82), (0.44, 1.00), (0.00, 1.00)],
          [(0.40, 0.82), (0.70, 0.84), (0.72, 1.00), (0.44, 1.00)],
          [(0.70, 0.84), (1.00, 0.86), (1.00, 1.00), (0.72, 1.00)]],
    river=[(0.03, 0.55), (0.15, 0.60), (0.26, 0.66), (0.30, 0.78), (0.46, 0.86), (0.58, 0.95), (0.62, 1.02)],
    fence=((0.29, 0.645), (0.48, 0.705)),
    houses=[((0.12, 0.79), 144, 108), ((0.45, 0.935), 144, 108), ((0.53, 0.625), 127, 97), ((0.84, 0.935), 144, 108)],
    apr=(0.21, 0.72),
    trees=[(0.30, 0.60, 24), (0.64, 0.66, 22), (0.95, 0.66, 26), (0.03, 0.66, 22), (0.60, 0.99, 28)],
    sheep=[(0.78, 0.76), (0.86, 0.78), (0.72, 0.79)],
    people=[0.055, 0.905, 0.375, 0.985, 0.595, 0.700, 0.93, 0.985],
    stakes=[(0.575 + 0.034 * k, 0.800 + 0.007 * k) for k in range(4)],
    intake=((0.292, 0.622), (0.262, 0.652)),
    andes=(0.47, [(0.10, 0.305, 0.09), (0.27, 0.262, 0.12), (0.47, 0.315, 0.09), (0.63, 0.27, 0.11), (0.88, 0.315, 0.1)]),
    back_line=[(0.635, 0.575), (0.465, 0.70), (0.295, 0.80), (0.125, 0.91)],
    araucarias=[(0.385, 0.505, 108), (0.425, 0.518, 88)],
)
TALL = dict(
    sky=0.36, sun=(0.80, 0.075), clouds=[(0.22, 0.042, 0.26, 0.026), (0.30, 0.20, 0.20, 0.026)],
    birds=[(0.40, 0.20, 1.0), (0.45, 0.185, 0.8), (0.49, 0.205, 0.7)],
    hills=[("#8a6f55", "flannel", "#6f5642", 0.30, [(0.10, 0.265), (0.36, 0.30), (0.60, 0.27)]),
           ("#6f8f55", "plain", None, 0.33, [(0.40, 0.29), (0.62, 0.31), (0.80, 0.30)]),
           ("#9a8a4a", "gingham", "#7c6d35", 0.35, [(0.02, 0.33), (0.20, 0.31), (0.46, 0.345)])],
    hills_bottom=0.50,
    tower=(0.74, 0.268), tower_hill=[(0.36, 0.40), (-0.22, 0.03), (0.0, 0.0), (0.16, 0.02), (None, 0.04)],
    tower_hill_bottom=0.48,
    forest=(0.06, 0.036, 0.312),
    rows=[[(0.00, 0.37), (0.52, 0.355), (0.58, 0.46), (0.00, 0.49)],
          [(0.52, 0.355), (1.00, 0.35), (1.00, 0.47), (0.58, 0.46)],
          [(0.00, 0.49), (0.58, 0.46), (0.54, 0.63), (0.00, 0.65)],
          [(0.58, 0.46), (1.00, 0.47), (1.00, 0.62), (0.54, 0.63)],
          [(0.00, 0.65), (0.54, 0.63), (0.60, 0.80), (0.00, 0.82)],
          [(0.54, 0.63), (1.00, 0.62), (1.00, 0.80), (0.60, 0.80)],
          [(0.00, 0.82), (0.30, 0.81), (0.33, 1.00), (0.00, 1.00)],
          [(0.30, 0.81), (0.60, 0.80), (0.64, 1.00), (0.33, 1.00)],
          [(0.60, 0.80), (1.00, 0.80), (1.00, 1.00), (0.64, 1.00)]],
    river=[(0.03, 0.385), (0.16, 0.44), (0.30, 0.53), (0.26, 0.64), (0.36, 0.76), (0.50, 0.88), (0.58, 1.02)],
    fence=((0.42, 0.555), (0.74, 0.585)),
    houses=[((0.17, 0.600), 144, 108), ((0.47, 0.965), 144, 108), ((0.64, 0.475), 127, 97), ((0.84, 0.855), 144, 108)],
    apr=(0.21, 0.785),
    trees=[(0.46, 0.44, 24), (0.93, 0.53, 26), (0.05, 0.49, 22), (0.70, 0.99, 28), (0.30, 0.86, 24)],
    sheep=[(0.78, 0.66), (0.88, 0.675), (0.70, 0.69)],
    people=[0.07, 0.70, 0.27, 0.985, 0.44, 0.712, 0.94, 0.975],
    stakes=[(0.545 + 0.055 * k, 0.725 + 0.005 * k) for k in range(4)],
    intake=((0.345, 0.620), (0.300, 0.635)),
    andes=(0.30, [(0.14, 0.205, 0.15), (0.46, 0.17, 0.17), (0.82, 0.195, 0.15)]),
    back_line=[(0.585, 0.43), (0.43, 0.575), (0.285, 0.715), (0.14, 0.86)],
    araucarias=[(0.37, 0.335, 96), (0.455, 0.345, 80)],
)


def build(W, H, seed=11):
    rng = np.random.default_rng(seed)
    u = min(W, H) / 1080.0
    X = lambda f: f * W
    Y = lambda f: f * H
    P = lambda fx, fy: (fx * W, fy * H)
    portrait = H > W
    Lz = TALL if portrait else LAND

    col, hgt, T = burlap(W, H, seed=seed + 1, stencil=stencil_mask(W, H, u, y0=0.095 if portrait else 0.085,
                                                                   cx=0.5 if portrait else 0.62))
    canvas = col.copy()
    T0 = T.copy()
    from .thread import composite_T_piece
    composite_T_piece.blur = cv2.GaussianBlur(T0, (0, 0), 2.4 * u)
    C = np.zeros((H, W), np.float32)
    info = {"layout": Lz, "portrait": portrait}

    seams = []

    def bake(piece):
        piece.bake(canvas, T, T0, C)
        if getattr(piece, "seam", None) is not None:
            seams.append(piece.seam)

    def bake_sprite(sp, shadow=0.5):
        composite(canvas, sp, shadow)
        composite_T(T, sp)

    def bake_yarn(y):
        y.draw(canvas, y.length + 10)
        for ch in y.chunks:
            composite_T(T, ch)

    m = 40 * u   # margen hasta el borde de lana
    # --- cielo, sol y nubes -------------------------------------------------------------------------
    bake(Piece([(m, m), (W - m, m), (W - m, Y(Lz["sky"])), (m, Y(Lz["sky"]))], "#9fc4d8", rng, kind="plain",
               stitch=None, fabric_scale=u, fray=0.6))
    sx, sy = P(*Lz["sun"])
    disc, rays = sun(sx, sy, 60 * u, rng, 0, u=u)
    bake(disc)
    for r in rays:
        bake_sprite(r)
    for (cx, cy, cw, ch) in Lz["clouds"]:
        bake(cloud(X(cx), Y(cy), X(cw), Y(ch) * 1.8, rng, 0, u=u))

    for (bx, by, bs) in Lz["birds"]:
        cxb, cyb = P(bx, by)
        w_ = 14 * u * bs
        for p0, p1 in (((cxb - w_, cyb - w_ * 0.5), (cxb, cyb)), ((cxb, cyb), (cxb + w_, cyb - w_ * 0.55))):
            bake_sprite(Stitch.render(p0, p1, 2.2 * u, "#2a2522", rng))

    # --- la cordillera, al fondo ---------------------------------------------------------------------
    yb, pk = Lz["andes"]
    for p_ in cordillera(W, H, yb, pk, rng, u=u):
        bake(p_)
    # --- cerros de fondo --------------------------------------------------------------------------------
    for colr, kind, c2, base_f, peaks in Lz["hills"]:
        pts = [(m - 10, Y(base_f))] + [P(px, py) for px, py in peaks] + [(W - m + 10, Y(base_f + 0.02))]
        line = catmull_rom(np.array(pts), 10)
        poly = np.vstack([line, [(W - m + 10, Y(Lz["hills_bottom"])), (m - 10, Y(Lz["hills_bottom"]))]])
        bake(Piece(poly, colr, rng, kind=kind, color2=c2, stitch=("#3d2a1e", 9 * u, 7 * u, 1.8 * u), fabric_scale=u))
    # cerro de la torre (derecha)
    tx, ty = P(*Lz["tower"])
    th = Lz["tower_hill"]
    hl = [P(*th[0])] + [((W - m + 10) if dx is None else tx + X(dx), ty + Y(dy)) for dx, dy in th[1:]]
    hill_line = catmull_rom(np.array(hl), 12)
    bake(Piece(np.vstack([hill_line, [(W - m + 10, Y(Lz["tower_hill_bottom"])), (hl[0][0], Y(Lz["tower_hill_bottom"]))]]),
               "#c49a45", rng, kind="cord", stitch=("#6b4b1e", 9 * u, 7 * u, 1.8 * u), fabric_scale=u, angle=0.3))
    info["tower_base"] = np.array([tx, ty + 4 * u])
    # bosque nativo: copas de fieltro apretadas sobre el cerro izquierdo
    fx0, fdx, fy0 = Lz["forest"]
    for k in range(9):
        cx = X(fx0 + fdx * k + rng.normal(0, 0.006))
        cy = Y(fy0 + 0.02 * np.sin(k * 1.7) * (0.6 if portrait else 1.0) + rng.normal(0, 0.006 * (0.6 if portrait else 1)))
        r = 26 * u * rng.uniform(0.8, 1.25)
        colr = ["#29462c", "#34553a", "#1f3a26", "#3b5e36"][k % 4]
        bake(Piece(ellipse_poly(cx, cy, r, r * 0.9, wobble=0.06, rng=rng), colr, rng, kind="felt",
                   stitch=("#15261a", 6 * u, 5 * u, 1.3 * u), fabric_scale=u))

    for (fx_, fy_, hh) in Lz["araucarias"]:
        tr_, tiers = araucaria(X(fx_), Y(fy_), hh * u, rng, 0, u=u)
        for y_ in tr_:
            bake_yarn(y_)
        for t_ in tiers:
            bake(t_)

    # --- potreros de retazos ----------------------------------------------------------------------------
    fabrics = [("#6d9a4a", "gingham", "#4f7a35"), ("#8aa84f", "plain", None), ("#5d8a54", "dots", "#dfe6c6"),
               ("#a5b35a", "cord", None), ("#4f7d3f", "flannel", "#3a5e2e"), ("#7fa35f", "print", "#e3d27a"),
               ("#5f8f3e", "cord", None), ("#91ad5c", "gingham", "#6f8d42"), ("#6a8f4c", "plain", None)]
    for poly, (c1, kind, c2) in zip(Lz["rows"], fabrics):
        if portrait:
            pp = np.array([(m - 12 + fx * (W - 2 * m + 24), fy * H) for fx, fy in poly])
        else:
            pp = np.array([(m - 12 + fx * (W - 2 * m + 24), fy * (H - m) + (m if fy < 0.5 else 0)) for fx, fy in poly])
        pp[:, 1] = np.clip(pp[:, 1], m, H - m + 12)
        bake(Piece(pp, c1, rng, kind=kind, color2=c2, stitch=("#3a2a1c", 9 * u, 7 * u, 1.8 * u), fabric_scale=u,
                   angle=rng.normal(0, 0.05)))

    # --- el río (raso azul con puntadas de corriente) ----------------------------------------------------
    river_c = catmull_rom(np.array([P(*p) for p in Lz["river"]]), 14)
    wr = np.linspace(22, 60, len(river_c)) * u
    tang = np.gradient(river_c, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nor = np.stack([-tang[:, 1], tang[:, 0]], 1)
    river_poly = np.vstack([river_c + nor * wr[:, None] / 2, (river_c - nor * wr[:, None] / 2)[::-1]])
    bake(Piece(river_poly, "#3d78ad", rng, kind="satin", stitch=("#1f3f66", 8 * u, 6 * u, 1.6 * u), fabric_scale=u))
    for k in range(5):
        off = (k - 2) / 2.6
        line = river_c + nor * (wr[:, None] / 2) * off * 0.7
        line = line[int(len(line) * 0.08 * k):]
        for _, sp in running_stitch(line, 12 * u, 16 * u, 1.8 * u, "#e9f1f5", rng):
            bake_sprite(sp)
    info["river"] = river_c

    # --- cerco (postes y alambres de lana) ---------------------------------------------------------------
    f0, f1 = np.array(P(*Lz["fence"][0])), np.array(P(*Lz["fence"][1]))
    posts = [f0 + (f1 - f0) * k / 7 for k in range(8)]
    for p in posts:
        bake_yarn(Yarn([p + (0, 6 * u), p - (0, 30 * u)], 4.4 * u, "#5b3e25", rng, fuzz=0.5))
    for off in (10, 22):
        bake_yarn(Yarn([f0 - (0, off * u), f1 - (0, off * u)], 2.2 * u, "#8a8580", rng, fuzz=0.2, kind="floss"))
    info["fence"] = (f0, f1)

    # --- casas, estanque del APR, árboles, ovejas, gente ---------------------------------------------------
    looks = [dict(wall="#b7342b", roof="#77726c", k="plain"), dict(wall="#3f6fa5", roof="#6a6560", k="stripes"),
             dict(wall="#e3b33c", roof="#7b7670", k="plain"), dict(wall="#e2d7c0", roof="#6b665f", k="gingham")]
    houses = [dict(c=P(*c), w=w_ * u, h=h_ * u, **lk) for (c, w_, h_), lk in zip(Lz["houses"], looks)]
    info["houses"] = []
    for i, hd in enumerate(houses):
        parts, wins, door = house(hd["c"][0], hd["c"][1], hd["w"], hd["h"], rng, 0, wall=hd["wall"],
                                  roof=hd["roof"], wall_kind=hd["k"], windows=1, door_side=0.5 if i % 2 else -0.5, u=u)
        for p_ in parts:
            bake(p_)
        info["houses"].append(dict(center=np.array(hd["c"]), door=np.array(door), top=hd["c"][1] - hd["h"],
                                   lit=[sp for w_ in wins for _, sp in w_]))
    # chimenea (para el humo)
    hA = houses[0]
    chx = hA["c"][0] + hA["w"] * 0.22
    chy = hA["c"][1] - hA["h"] - hA["h"] * 0.35
    bake(Piece([(chx - 7 * u, chy + 18 * u), (chx - 7 * u, chy - 8 * u), (chx + 7 * u, chy - 8 * u),
                (chx + 7 * u, chy + 18 * u)], "#6d4a35", rng, kind="felt", stitch=None, margin=6))
    info["chimney"] = np.array([chx, chy - 8 * u])
    # estanque elevado del agua potable rural (APR)
    ax, ay = P(*Lz["apr"])
    for lx in (-16, -5, 5, 16):
        bake_yarn(Yarn([(ax + lx * u, ay), (ax + lx * 0.6 * u, ay - 58 * u)], 3.2 * u, "#8b8f93", rng, fuzz=0.2,
                       kind="floss"))
    tank = [(ax - 26 * u, ay - 56 * u), (ax - 26 * u, ay - 104 * u), (ax + 26 * u, ay - 104 * u),
            (ax + 26 * u, ay - 56 * u)]
    bake(Piece(tank, "#8fb6cc", rng, kind="stripes", color2="#6f97ad", stitch=("#2f4f63", 6 * u, 4 * u, 1.4 * u),
               fabric_scale=u * 0.8, angle=np.pi / 2))
    strokes, _ = hershey_strokes("APR", "futural", 13 * u, ax, ay - 74 * u, anchor="center")
    for st in strokes:
        for _, sp in running_stitch(st, 3.2 * u, 0.8 * u, 1.3 * u, "#1f2f3a", rng, jitter=0.2):
            bake_sprite(sp)
    # árboles sueltos
    for (fx, fy, r) in Lz["trees"]:
        trunks, crowns = round_tree(X(fx), Y(fy), r * u, rng, 0, u=u)
        for y_ in trunks:
            bake_yarn(y_)
        for c_ in crowns:
            bake(c_)
    # ovejas de bouclé
    for i_s, (fx, fy) in enumerate(Lz["sheep"]):
        cx, cy = P(fx, fy)
        k_s = rng.uniform(0.85, 1.18)                  # cada oveja de su tamaño
        side = -1 if i_s % 2 == 0 else 1               # y mirando hacia su lado
        for leg in (-10, -4, 5, 11):
            bake_yarn(Yarn([(cx + leg * u * k_s, cy), (cx + leg * u * k_s + rng.normal(0, 1), cy + 15 * u * k_s)],
                           2.6 * u, "#2a2320", rng, fuzz=0.2))
        for k in range(int(26 * k_s)):
            a = rng.uniform(0, 2 * np.pi)
            rr = np.sqrt(rng.uniform(0, 1))
            bake_sprite(knot((cx + np.cos(a) * rr * 17 * u * k_s, cy - 6 * u + np.sin(a) * rr * 10 * u * k_s),
                             rng.uniform(3.2, 4.4) * u, rng.choice(["#f1ede4", "#ebe4d6", "#f4f0e8"]), rng), shadow=0.35)
        hy_ = cy - (10 + rng.uniform(-4, 5)) * u        # una pastando, otra mirando
        bake_sprite(knot((cx + side * 20 * u * k_s, hy_), 5.5 * u, "#2b2522", rng))
    # vecinas y vecinos de lana (sin carita): se animan en la escena; aquí solo su silueta a contraluz
    pp_ = Lz["people"]
    looks = [("#7d3c6a", "dots", "#e8dcc8", "#c68e67", "#3a2a20"), ("#2f6f4f", "gingham", "#e9e1cf", "#9a6a4b", "#1d1612"),
             ("#c0582f", "print", "#f0d98a", "#d7a47e", "#5a3a22"), ("#3b4f8f", "stripes", "#e8e2d2", "#b98363", "#2a1d15")]
    info["dolls"] = []
    for i, (dc, dk, d2, skin, hair) in enumerate(looks):
        fx, fy = pp_[2 * i], pp_[2 * i + 1]
        spec = dict(fx=X(fx), base=Y(fy) - 6 * u, hgt=150 * u * rng.uniform(0.88, 1.12), dress=dc, dress_kind=dk,
                    dress2=d2, skin=skin, hair=hair, u=u)
        spec["arms"] = ["down", "down", "down", "hip"][i]
        spec["kind"] = ["elder", "man", "woman", "child"][i]
        info["dolls"].append(spec)
        # (la silueta a contraluz se calcula en cada imagen, según la pose)

    # --- borde de punto festón y orilla del saco --------------------------------------------------------------
    b = 26 * u
    border = [(b, b), (W - b, b), (W - b, H - b), (b, H - b), (b, b)]
    for _, sp in blanket_stitch(border, 14 * u, 16 * u, 3.6 * u, "#c98f2c", rng, inward=-1):
        bake_sprite(sp)
    # --- el revés, a contraluz: la puntada corrida es un hilo continuo y hay nudos -----------------------
    back = np.zeros((H * 2, W * 2), np.float32)
    for poly_, sw_ in seams:
        pts_ = np.vstack([poly_, poly_[:1]]) * 2
        cv2.polylines(back, [np.round(pts_ * 8).astype(np.int32)], False, 1.0, max(1, int(sw_ * 1.6)), cv2.LINE_AA,
                      shift=3)
        for _ in range(int(rng.integers(1, 3))):          # nudos de comienzo y fin
            q_ = poly_[int(rng.integers(0, len(poly_)))] * 2
            cv2.circle(back, (int(q_[0]), int(q_[1])), max(2, int(sw_ * 2.6)), 1.0, -1, cv2.LINE_AA)
    back = cv2.resize(back, (W, H), interpolation=cv2.INTER_AREA)
    back = np.clip(cv2.GaussianBlur(back, (0, 0), 0.9 * u) * 1.3, 0, 1)
    T *= (1 - 0.72 * back)[..., None]
    info["u"] = u
    info["W"], info["H"] = W, H
    info["stakes"] = [np.array(P(*p)) for p in Lz["stakes"]]
    info["intake"] = (np.array(P(*Lz["intake"][0])), np.array(P(*Lz["intake"][1])))
    info["back_line"] = [np.array(P(*p)) for p in Lz["back_line"]]
    return canvas, T, info


def _bake_doll(d, canvas, T):
    for y in d["legs"]:
        _yarn_full(canvas, y)
        for ch in y.chunks:
            composite_T(T, ch)
    d["body"].bake(canvas, T)   # la muñeca es gruesa: silueta
    for s in d["shoes"]:
        composite(canvas, s)
    composite(canvas, d["head"])
    composite_T(T, d["head"])
    for y in d["hair"] + d["arms"]:
        _yarn_full(canvas, y)
        for ch in y.chunks:
            composite_T(T, ch)


# ------------------------------------------------------------------------------------------------------
# Otros territorios de la red (se ven colgados al final, junto a la arpillera principal)
# ------------------------------------------------------------------------------------------------------
def _common(W, H, seed, sky="#a9cadb"):
    rng = np.random.default_rng(seed)
    u = min(W, H) / 1080.0
    col, hgt, T = burlap(W, H, seed=seed + 1)
    canvas = col.copy()
    m = 40 * u

    def bake(piece):
        piece.bake(canvas, T)

    def spr(sp, shadow=0.5):
        composite(canvas, sp, shadow)

    bake(Piece([(m, m), (W - m, m), (W - m, H * 0.62), (m, H * 0.62)], sky, rng, kind="plain", stitch=None,
               fabric_scale=u, fray=0.6))
    return rng, u, canvas, T, m, bake, spr


def _border(canvas, W, H, u, rng, color="#c98f2c"):
    b = 26 * u
    for _, sp in blanket_stitch([(b, b), (W - b, b), (W - b, H - b), (b, H - b), (b, b)], 14 * u, 16 * u, 3.6 * u,
                                color, rng, inward=-1):
        composite(canvas, sp)


def _lattice_tower(spr, base, hgt, u, rng, col="#141516"):
    """Torre de alta tensión hilvanada (puntadas largas con huecos): también es un proyecto."""
    _spr = spr

    def spr(sp_or_seg):
        _spr(sp_or_seg)
    bx, by = base
    wb, wt = hgt * 0.22, hgt * 0.06
    top = by - hgt
    L = [(bx - wb / 2, by), (bx - wt / 2, top + hgt * 0.18)]
    R = [(bx + wb / 2, by), (bx + wt / 2, top + hgt * 0.18)]
    for a, b in (L, R):
        _hilvan_seg(_spr, a, b, 2.4 * u, col, rng, u)
    n = 5
    for i in range(n):
        f0, f1 = i / n, (i + 1) / n
        yl0 = by - (hgt * 0.82) * f0
        yl1 = by - (hgt * 0.82) * f1
        w0 = wb / 2 + (wt / 2 - wb / 2) * f0
        w1 = wb / 2 + (wt / 2 - wb / 2) * f1
        _hilvan_seg(_spr, (bx - w0, yl0), (bx + w1, yl1), 1.5 * u, col, rng, u)
        _hilvan_seg(_spr, (bx + w0, yl0), (bx - w1, yl1), 1.5 * u, col, rng, u)
    for fy, fw in ((0.82, 0.30), (0.95, 0.22)):
        y = by - hgt * fy
        _hilvan_seg(_spr, (bx - hgt * fw, y), (bx + hgt * fw, y), 2.2 * u, col, rng, u)
    _hilvan_seg(_spr, (bx - wt / 2, top + hgt * 0.18), (bx, top), 2.0 * u, col, rng, u)
    _hilvan_seg(_spr, (bx + wt / 2, top + hgt * 0.18), (bx, top), 2.0 * u, col, rng, u)
    return [(bx - hgt * 0.30, by - hgt * 0.82), (bx + hgt * 0.30, by - hgt * 0.82)]


def _hilvan_seg(spr, a, b, w, col, rng, u):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    L = float(np.hypot(*(b - a)))
    n = max(1, int(round(L / (16 * u))))
    for k in range(n):
        f0, f1 = k / n, (k + 0.7) / n
        spr(Stitch.render(a + (b - a) * f0, a + (b - a) * f1, w, col, rng, kind="synthetic"))


def _cable(spr, a, b, sag, u, rng):
    pts = catmull_rom(np.array([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + sag), b]), 12)
    for _, sp in running_stitch(pts, 18 * u, 8 * u, 1.3 * u, "#141516", rng, jitter=0.2, kind="synthetic"):
        spr(sp)


def _red_line(canvas, pts, u, rng):
    """La lana roja de una arpillera vecina: NO se hornea; llega cuando le llega la alerta por el cordel.
    Se arma al revés (del borde de arriba a la puerta) para poder bajarla de a poco."""
    path = catmull_rom(np.array(pts), 16)[::-1]
    return Yarn(path, 8.0 * u, "#c3241c", rng, couch_every=19 * u, couch_color="#8e160f")


def exit_points(path, u, depth=26):
    """Dónde cruza la lana el borde de arriba (y = 0) y la guarda de festón (y = depth·u), buscando
    desde el final del camino: [(x, depth·u), (x, 0)] en coordenadas de la imagen."""
    pts = np.asarray(path, np.float64)
    out = []
    for yc in (depth * u, 0.0):
        x = pts[-1, 0]
        for i in range(len(pts) - 1, 0, -1):
            a, b = pts[i - 1], pts[i]
            if (a[1] - yc) * (b[1] - yc) <= 0 and a[1] != b[1]:
                x = a[0] + (b[0] - a[0]) * (yc - a[1]) / (b[1] - a[1])
                break
        out.append((float(x), float(yc)))
    return out


def build_desert(W, H, seed=31):
    """Desierto con un parque solar y una línea de alta tensión."""
    rng, u, canvas, T, m, bake, spr = _common(W, H, seed, sky="#b4d3e2")
    X = lambda f: f * W
    Y = lambda f: f * H
    d, rays = sun(X(0.16), Y(0.16), 64 * u, rng, 0, u=u)
    bake(d)
    for r in rays:
        spr(r)
    for (c1, kind, top, peaks) in [("#c89a64", "cord", 0.50, [(0.2, 0.45), (0.55, 0.49), (0.85, 0.44)]),
                                   ("#d9b27a", "plain", 0.58, [(0.1, 0.56), (0.45, 0.60), (0.8, 0.55)]),
                                   ("#e3c08a", "flannel", 0.70, [(0.3, 0.68), (0.7, 0.72)])]:
        pts = [(m - 10, Y(top))] + [(X(px), Y(py)) for px, py in peaks] + [(W - m + 10, Y(top))]
        line = catmull_rom(np.array(pts), 10)
        bake(Piece(np.vstack([line, [(W - m + 10, H - m + 10), (m - 10, H - m + 10)]]), c1, rng, kind=kind,
                   color2="#b98a55", stitch=("#6b4b2a", 9 * u, 7 * u, 1.8 * u), fabric_scale=u))
    # parque solar: filas de paneles
    for row in range(4):
        y = Y(0.66 + row * 0.075)
        pw, ph = X(0.07) * (1 + 0.15 * row), Y(0.04) * (1 + 0.15 * row)
        x = X(0.30) - row * X(0.02)
        while x + pw < X(0.96):
            if row >= 2 and x + pw > X(0.53) and x < X(0.81):     # ahí vive una familia: no hay paneles
                x += pw * 1.25
                continue
            poly = [(x, y), (x + pw, y), (x + pw * 1.08, y + ph), (x + pw * 0.08, y + ph)]
            bake(Piece(poly, "#2c3a52", rng, kind="plain", stitch=None, fabric_scale=u * 0.7, margin=6))
            for k in (1, 2):
                a_ = (x + pw * k / 3, y)
                b_ = (x + pw * 0.08 + pw * k / 3, y + ph)
                spr(Stitch.render(a_, b_, 1.3 * u, "#c9ced4", rng))
            spr(Stitch.render((x + pw * 0.04, y + ph / 2), (x + pw * 1.04, y + ph / 2), 1.3 * u, "#c9ced4", rng))
            x += pw * 1.25
    arms = [_lattice_tower(spr, (X(0.14), Y(0.80)), Y(0.30), u, rng),
            _lattice_tower(spr, (X(0.36), Y(0.60)), Y(0.20), u, rng)]
    _cable(spr, arms[0][1], arms[1][0], 14 * u, u, rng)
    _cable(spr, arms[1][1], (W - m, Y(0.40)), 18 * u, u, rng)
    _cable(spr, (m, Y(0.44)), arms[0][0], 18 * u, u, rng)
    parts, wins, door = house(X(0.62), Y(0.93), 110 * u, 96 * u, rng, 0, wall="#d9a96c", roof="#8b5e3c",
                              wall_kind="plain", roof_kind="cord", u=u)
    for p_ in parts:
        bake(p_)
    lit = [sp for w_ in wins for _, sp in w_]
    dd = doll(X(0.74), Y(0.97), 150 * u, rng, 0, dress="#6d3b7a", dress_kind="dots", dress2="#efe2c6",
              skin="#b07a55", hair="#1d1612", u=u)
    _bake_doll(dd, canvas, T)
    y = _red_line(canvas, [door, (X(0.70), Y(0.80)), (X(0.78), Y(0.58)), (X(0.74), Y(0.30)), (X(0.76), -20)], u, rng)
    _border(canvas, W, H, u, rng, color="#2f5a8a")
    return dict(img=canvas, exit=exit_points(y.pts[::-1], u), red=y, lit=lit)


def build_towers(W, H, seed=41):
    """Campo cruzado por una línea de alta tensión."""
    rng, u, canvas, T, m, bake, spr = _common(W, H, seed, sky="#9dc0d6")
    X = lambda f: f * W
    Y = lambda f: f * H
    for (c1, kind, c2, top, peaks) in [("#6f8f55", "flannel", "#557340", 0.48, [(0.25, 0.42), (0.7, 0.46)]),
                                       ("#8aa84f", "gingham", "#6d8b3a", 0.58, [(0.15, 0.56), (0.5, 0.60), (0.85, 0.55)]),
                                       ("#5f8f3e", "cord", None, 0.72, [(0.35, 0.70), (0.75, 0.74)])]:
        pts = [(m - 10, Y(top))] + [(X(px), Y(py)) for px, py in peaks] + [(W - m + 10, Y(top))]
        line = catmull_rom(np.array(pts), 10)
        bake(Piece(np.vstack([line, [(W - m + 10, H - m + 10), (m - 10, H - m + 10)]]), c1, rng, kind=kind,
                   color2=c2, stitch=("#2e3d1f", 9 * u, 7 * u, 1.8 * u), fabric_scale=u))
    for (fx, fy, r) in [(0.10, 0.50, 28), (0.16, 0.52, 24), (0.88, 0.50, 26)]:
        trunks, crowns = round_tree(X(fx), Y(fy), r * u, rng, 0, u=u)
        for y_ in trunks:
            y_.draw(canvas, 1e9)
        for c_ in crowns:
            bake(c_)
    arms = []
    for (fx, fy, hh) in [(0.30, 0.78, 0.36), (0.55, 0.64, 0.26), (0.76, 0.56, 0.18)]:
        arms.append(_lattice_tower(spr, (X(fx), Y(fy)), Y(hh), u, rng))
    for i in range(len(arms) - 1):
        _cable(spr, arms[i][0], arms[i + 1][0], 16 * u, u, rng)
        _cable(spr, arms[i][1], arms[i + 1][1], 16 * u, u, rng)
    _cable(spr, (m, Y(0.40)), arms[0][0], 20 * u, u, rng)
    _cable(spr, arms[-1][1], (W - m, Y(0.40)), 12 * u, u, rng)
    lit = []
    for (fx, wall) in [(0.12, "#3f6fa5"), (0.84, "#b7342b")]:
        parts, wins, door = house(X(fx), Y(0.92), 110 * u, 96 * u, rng, 0, wall=wall, roof="#6e6a66", u=u)
        for p_ in parts:
            bake(p_)
        if fx < 0.5:
            lit = [sp for w_ in wins for _, sp in w_]
    dd = doll(X(0.20), Y(0.97), 150 * u, rng, 0, dress="#c0582f", dress_kind="gingham", dress2="#f0e3c8",
              skin="#c68e67", hair="#3a2a20", u=u)
    _bake_doll(dd, canvas, T)
    y = _red_line(canvas, [(X(0.13), Y(0.88)), (X(0.30), Y(0.70)), (X(0.48), Y(0.76)), (X(0.60), Y(0.40)),
                           (X(0.40), -20)], u, rng)
    _border(canvas, W, H, u, rng, color="#7a2e5a")
    return dict(img=canvas, exit=exit_points(y.pts[::-1], u), red=y, lit=lit)
