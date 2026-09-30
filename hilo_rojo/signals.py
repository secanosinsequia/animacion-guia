"""Las huellas del megaproyecto, cosidas con el mismo hilo negro (el hilván):

* estacas de topógrafo con su cinta naranja, de a una;
* un AVISO de papel prendido al cerco con un alfiler de gancho;
* una toma de agua: una cañería gruesa hilvanada que entra al río, con su caseta;
* y, al final, la torre de alta tensión: solo un contorno de puntadas largas, a medio hacer (lo que
  vendría, todavía hilvanado).

Cada puntada del hilván es (t, sprite, p0, p1): sus dos extremos son agujeros en la tela, por donde el
hilo pasa al revés y por donde, a contraluz, se cuela la luz.
"""
import cv2
import numpy as np

from satc_intro.color import lin
from satc_intro.geometry import catmull_rom
from .figures import hershey_strokes
from .pieces import Piece
from .thread import Sprite, Stitch, Yarn, knot, backstitch

BLK = "#141516"


def safety_pin(a, b, w, rng):
    """Alfiler de gancho de a (resorte) a b (cabeza): dos alambres, el resorte en espiral y la cabeza."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = b - a
    L = float(np.hypot(*d)) + 1e-9
    t = d / L
    n = np.array([-t[1], t[0]])
    steel = "#b8bec4"
    sprites = []
    gap = w * 1.9
    # alambre de abajo (pasa por la tela) y de arriba (el que abre)
    sprites.append(Stitch.render(a + n * gap * 0.5, b - t * w * 1.2 + n * gap * 0.5, w, steel, rng, kind="synthetic",
                                 hole=False, bow=0))
    sprites.append(Stitch.render(a - n * gap * 0.5, b - t * w * 1.2 - n * gap * 0.45, w, steel, rng, kind="synthetic",
                                 hole=False, bow=0))
    # resorte: dos vueltas
    sprites.append(knot(a - t * w * 0.8, w * 1.45, steel, rng))
    # cabeza (el gancho que cierra)
    c = b - t * w * 0.6
    head = [c + n * gap * 0.95, c + n * gap * 0.95 + t * w * 2.2, c - n * gap * 0.95 + t * w * 2.2, c - n * gap * 0.95]
    hs = Stitch.render(head[0], head[1], w * 1.25, steel, rng, kind="synthetic", hole=False, bow=0)
    sprites.append(hs)
    sprites.append(Stitch.render(head[1], head[2], w * 1.25, steel, rng, kind="synthetic", hole=False, bow=0))
    sprites.append(Stitch.render(head[2], head[3], w * 1.25, steel, rng, kind="synthetic", hole=False, bow=0))
    return sprites


def build_signals(info, u, rng, T0):
    """Devuelve dict con: puntadas del hilván (t, sprite, p0, p1, huella), retazos que se posan (aviso,
    caseta), sprites de adorno (alfiler, cañería) y los puntos clave de cada huella. T0: tiempos."""
    g = []                    # (t, sprite, p0, p1, huella)
    extra = []                # (t, sprite, huella): cosas que se van con el hilván pero no son puntadas
    pieces = []               # (t, Piece, huella)

    def st(tt, a, b, w, col=BLK, kind="synthetic", h=0, bow=None):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        g.append((tt, Stitch.render(a, b, w, col, rng, kind=kind, bow=bow), a, b, h))

    # --- 1. estacas de topógrafo (H2) ---------------------------------------------------------------
    t0, t1 = T0["stakes"]
    stakes = info["stakes"]
    for i, sp_ in enumerate(stakes):
        tt = t0 + (t1 - t0) * (i + 0.3) / len(stakes)
        top = sp_ - (0, 44 * u)
        st(tt, sp_, top, 4.2 * u, "#3b2a1e", kind="floss", h=0, bow=0)          # la estaca (palo)
        # la cinta naranja: un retacito cosido con una puntada negra
        flag = [top + (0, 1 * u), top + (19 * u, 5 * u), top + (17 * u, 14 * u), top + (0, 11 * u)]
        pieces.append((tt, Piece(flag, "#f06a14", rng, kind="plain", stitch=None, fabric_scale=u * 0.6, margin=6,
                                 rough=0.5, shadow=0.6), 0))
        st(tt + 0.02, top + (2 * u, 4 * u), top + (2 * u, 10 * u), 2.4 * u, h=0, bow=0)
    H2 = stakes[1] - (0, 20 * u)

    # --- 2. el AVISO prendido al cerco (H3) ----------------------------------------------------------
    t0, t1 = T0["notice"]
    f0, f1 = info["fence"]
    nc = f0 + (f1 - f0) * 0.56 - (0, 40 * u)
    nw, nh = 42 * u, 30 * u
    ang = np.deg2rad(-4)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    corners = [nc + R @ np.array(v) for v in [(-nw, -nh), (nw, -nh), (nw, nh), (-nw, nh)]]
    notice = Piece(corners, "#f3efe3", rng, kind="plain", stitch=None, margin=8, fabric_scale=u * 0.8, rough=0.35,
                   wear=0.4, puff=0.6)
    pieces.append((t0, notice, 1))
    # AVISO en letras de imprenta bordadas en rojo, y tres renglones de letra chica
    strokes, _ = hershey_strokes("AVISO", "futuram", 15 * u, nc[0], nc[1] - 6 * u, anchor="center")
    ta = t0 + 0.2
    for k, s_ in enumerate(strokes):
        s_ = (np.asarray(s_) - nc) @ R.T + nc
        for _, sp in backstitch(s_, 4.2 * u, 2.3 * u, "#c0261c", rng, jitter=0.15):
            g.append((ta + 0.012 * k, sp, s_[0], s_[-1], 1))
    for r in range(3):
        y_ = nc[1] + 5 * u + r * 7 * u
        x_ = nc[0] - nw * 0.7
        xe = nc[0] + nw * (0.7 if r < 2 else 0.2)
        while x_ < xe:
            a = (np.array([x_, y_]) - nc) @ R.T + nc
            b = (np.array([x_ + 4 * u, y_]) - nc) @ R.T + nc
            st(ta + 0.18 + 0.03 * r, a, b, 1.3 * u, BLK, "floss", h=1, bow=0)
            x_ += 6 * u * rng.uniform(0.85, 1.2)
    # el alfiler de gancho que lo prende al alambre del cerco
    pa = nc + R @ np.array([-nw * 0.5, -nh - 4 * u])
    pb = nc + R @ np.array([nw * 0.45, -nh - 7 * u])
    for sp in safety_pin(pa, pb, 2.3 * u, rng):
        extra.append((t0 + 0.08, sp, 1))
    # dos puntadas de hilván en las esquinas de abajo
    for cx_ in (-1, 1):
        c = nc + R @ np.array([cx_ * (nw - 6 * u), nh - 6 * u])
        st(t0 + 0.45 + 0.05 * (cx_ + 1), c - (4 * u, 3 * u), c + (4 * u, 3 * u), 2.4 * u, h=1)
    H3 = nc.copy()

    # --- 3. la toma de agua: una cañería que sale del río, con su llave de paso (H4) -------------------
    t0, t1 = T0["intake"]
    bank, into = info["intake"]
    d = into - bank
    L = float(np.hypot(*d))
    tv = d / L
    nv = np.array([-tv[1], tv[0]])
    # la cañería: nace dentro del agua (boca), cruza la orilla y se va potrero adentro
    mouth = into + tv * 16 * u
    out_end = bank - tv * 150 * u + nv * 10 * u
    pipe_pts = catmull_rom(np.array([mouth, (mouth + bank) / 2 + nv * 2 * u, bank - tv * 20 * u,
                                     bank - tv * 90 * u + nv * 6 * u, out_end]), 10)
    pipe = Yarn(pipe_pts, 11.0 * u, "#8e9498", rng, fuzz=0.1, kind="floss")
    for ch in pipe.chunks:
        extra.append((t0 + 0.06, ch, 2))
    # boca de la cañería en el agua: un anillo oscuro
    extra.append((t0 + 0.06, knot(mouth, 6.5 * u, "#3a3f43", rng), 2))
    # la llave de paso: un volante rojo con cuatro rayos, sobre la cañería
    vc = bank - tv * 60 * u + nv * 3 * u
    R_ = 15 * u
    ring = catmull_rom(np.array([vc + R_ * np.array([np.cos(a_), np.sin(a_)]) for a_ in
                                 np.linspace(0, 2 * np.pi, 13)]), 4)
    for ch in Yarn(ring, 4.4 * u, "#b42a1e", rng, fuzz=0.2, kind="floss").chunks:
        extra.append((t0 + 0.12, ch, 2))
    for k in range(4):
        a_ = np.pi / 4 + k * np.pi / 2
        extra.append((t0 + 0.12, Stitch.render(vc, vc + R_ * np.array([np.cos(a_), np.sin(a_)]), 2.8 * u, "#8e1f16", rng,
                                               kind="floss", hole=False, bow=0), 2))
    extra.append((t0 + 0.12, knot(vc, 4.2 * u, "#6e1a12", rng), 2))
    # vástago de la llave (une el volante con la cañería)
    extra.append((t0 + 0.1, Stitch.render(vc + nv * 1 * u, vc - nv * 0 * u + tv * 0, 4 * u, "#5d6266", rng,
                                          kind="floss", hole=False, bow=0), 2))
    # puntadas largas de hilván que sujetan la cañería (la tomaron «de paso»)
    for k in range(4):
        f = (k + 0.5) / 4
        idx = int(f * (len(pipe_pts) - 1))
        p = pipe_pts[idx]
        tg = pipe_pts[min(idx + 1, len(pipe_pts) - 1)] - pipe_pts[max(idx - 1, 0)]
        tg = tg / (np.linalg.norm(tg) + 1e-9)
        ng = np.array([-tg[1], tg[0]])
        st(t0 + 0.2 + 0.07 * k, p - ng * 10 * u - tg * 3 * u, p + ng * 10 * u + tg * 3 * u, 2.6 * u, h=2)
    kp = out_end + nv * 12 * u
    g.append((t0 + 0.5, knot(kp, 5.0 * u, BLK, rng), kp, kp, 2))
    H4 = bank.copy()

    # --- 4. la torre de alta tensión: solo un contorno de puntadas largas, a medio hacer (H1) ---------
    t0, t1 = T0["pylon"]
    tx, ty = info["tower_base"]
    hgt = 230 * u
    wb, wt = hgt * 0.26, hgt * 0.07
    waist = ty - hgt * 0.72
    top = ty - hgt
    segs = []
    # patas (hasta la cintura) y cuerpo superior
    segs += [((tx - wb / 2, ty), (tx - wt / 2, waist)), ((tx + wb / 2, ty), (tx + wt / 2, waist))]
    segs += [((tx - wt / 2, waist), (tx - wt / 2, top + hgt * 0.1)), ((tx + wt / 2, waist), (tx + wt / 2, top + hgt * 0.1))]
    # crucetas (dos pisos) y la punta
    for fy, fw in ((0.74, 0.34), (0.88, 0.26)):
        y = ty - hgt * fy
        segs.append(((tx - hgt * fw, y), (tx + hgt * fw, y)))
        segs.append(((tx - hgt * fw, y), (tx - wt / 2, y + hgt * 0.07)))
        segs.append(((tx + hgt * fw, y), (tx + wt / 2, y + hgt * 0.07)))
    segs.append(((tx - wt / 2, top + hgt * 0.1), (tx, top)))
    segs.append(((tx + wt / 2, top + hgt * 0.1), (tx, top)))
    # celosía: dos diagonales en las patas (una queda sin terminar: todavía es un proyecto)
    segs.append(((tx - wb / 2, ty), (tx + wb * 0.18, ty - hgt * 0.36)))
    segs.append(((tx + wb / 2, ty), (tx - wb * 0.05, ty - hgt * 0.25)))
    # hilván: puntadas largas con huecos
    items = []
    for (a, b) in segs:
        a, b = np.array(a), np.array(b)
        Ls = float(np.hypot(*(b - a)))
        n = max(1, int(round(Ls / (30 * u))))
        for k in range(n):
            f0, f1_ = k / n, (k + 0.72) / n
            items.append((a + (b - a) * f0, a + (b - a) * f1_))
    for i, (a, b) in enumerate(items):
        st(t0 + (t1 - t0) * i / len(items), a, b, 2.5 * u, h=3)
    # aisladores: puntaditas colgando de las crucetas
    for fy, fw in ((0.74, 0.34), (0.88, 0.26)):
        y = ty - hgt * fy
        for sx in (-1, 1):
            x = tx + sx * hgt * fw * 0.92
            st(t1 - 0.02, (x, y), (x, y + 9 * u), 2.2 * u, h=3, bow=0)
    H1 = np.array([tx, ty])
    g.sort(key=lambda it: it[0])
    return dict(stitches=g, extra=extra, pieces=pieces, H=dict(H1=H1, H2=H2, H3=H3, H4=H4), knot=kp,
                pylon_top=np.array([tx, top]), notice_c=nc)
