"""El mundo alrededor de la arpillera: pared de adobe encalada, el cordel de lana roja, perritos de
ropa, otras arpilleras colgadas (otros territorios) y la tira de tela con el título.

La cámara se aleja por pasos (stop-motion) con paralaje: la pared está más lejos que las arpilleras,
así que se agranda menos. Todo se compone en raster a la resolución de salida.
"""
import cv2
import numpy as np

from satc_intro.color import lin
from satc_intro.geometry import catmull_rom, resample
from satc_intro.noise import fbm, smooth_noise, smoothstep
from .figures import hershey_strokes
from .textile import fabric, fray_mask, shade
from .thread import Sprite, Stitch, Yarn, composite, knot, running_stitch

LIGHT2 = np.array([-0.64, -0.56])


def plaster(W, H, seed=5):
    """Pared de adobe encalada: cal irregular, llana, grietas finas, luz rasante."""
    rng = np.random.default_rng(seed)
    s = max(W, H) / 1920
    h = (0.55 * fbm((H, W), 260 * s, rng, octaves=4) + 0.30 * fbm((H, W), 40 * s, rng, octaves=3)
         + 0.15 * fbm((H, W), 6 * s, rng, octaves=2)).astype(np.float32)
    # marcas de llana: ruido estirado en diagonal
    n = rng.standard_normal((H // 4, W // 4)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=18, sigmaY=2)
    M = cv2.getRotationMatrix2D((W / 8, H / 8), 24, 1.0)
    n = cv2.warpAffine(n, M, (W // 4, H // 4), borderMode=cv2.BORDER_REFLECT)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    h += 0.05 * n / (n.std() + 1e-6)
    gy, gx = np.gradient(cv2.GaussianBlur(h, (0, 0), 1.2 * s))
    lam = 1 + 9.0 * s * (-gx * LIGHT2[0] - gy * LIGHT2[1])
    base = lin("#e7ddcb")
    tone = 1 + 0.07 * (fbm((H, W), 500 * s, rng, octaves=3) - 0.5)
    col = base[None, None, :] * (tone * np.clip(lam, 0.75, 1.25))[..., None]
    # grietas
    cr = np.zeros((H, W), np.float32)
    for _ in range(7):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        a = rng.uniform(0, 2 * np.pi)
        for k in range(int(rng.uniform(40, 120))):
            a += rng.normal(0, 0.35)
            nx, ny = x + np.cos(a) * 4 * s, y + np.sin(a) * 4 * s
            cv2.line(cr, (int(x * 4), int(y * 4)), (int(nx * 4), int(ny * 4)), 1.0, 1, cv2.LINE_AA, shift=2)
            x, y = nx, ny
    col *= (1 - 0.28 * cv2.GaussianBlur(cr, (0, 0), 0.6 * s))[..., None]
    # una mancha de humedad cerca del zócalo
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    damp = smoothstep(0.82, 1.0, yy + 0.08 * (fbm((H, W), 120 * s, rng, octaves=3) - 0.5))
    col *= (1 - 0.10 * damp)[..., None]
    return np.clip(col, 0, 1).astype(np.float32)


def clothespin(p, ang, L, rng):
    """Perrito de ropa de madera con su resorte (sprite en coordenadas de pantalla)."""
    wood = lin("#cfa36a")
    w = L * 0.22
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    pad = int(L * 0.8) + 6
    x0, y0 = int(p[0] - pad), int(p[1] - pad)
    n = 2 * pad
    yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n].astype(np.float32) + 0.5
    qx, qy = xx - p[0], yy - p[1]
    lx = qx * c + qy * s          # a lo largo del perrito
    ly = -qx * s + qy * c
    a = np.zeros(xx.shape, np.float32)
    shade_ = np.zeros(xx.shape, np.float32)
    for side in (-1, 1):
        cx = side * w * 0.26
        hw = w * 0.24 * (1 - 0.35 * np.clip((lx + L * 0.5) / L, 0, 1))
        inside = (np.abs(ly - cx) < hw) & (lx > -L * 0.5) & (lx < L * 0.5)
        m = np.clip(hw - np.abs(ly - cx) + 0.5, 0, 1) * np.clip(L * 0.5 - np.abs(lx) + 0.5, 0, 1)
        dz = np.clip((ly - cx) / np.maximum(hw, 1e-3), -1, 1)
        nz = np.sqrt(np.clip(1 - dz * dz, 0, 1))
        lam = np.clip(-dz * (s * LIGHT2[0] * -1 + c * LIGHT2[1] * 0) * 0.6 + nz * 0.8, 0, 1.2)
        shade_ = np.where(m > a, lam, shade_)
        a = np.maximum(a, m)
    grain = 1 + 0.10 * np.sin(lx * 0.9 / max(1.0, L / 70) + ly * 0.2) + 0.05 * (rng.random(a.shape) - 0.5)
    rgb = wood[None, None, :] * (0.45 + 0.7 * shade_)[..., None] * grain[..., None]
    # resorte de alambre
    spring = (np.abs(lx - L * 0.02) < L * 0.09) & (np.abs(ly) < w * 0.62)
    coil = 0.5 + 0.5 * np.cos(lx * 1.6 / max(1.0, L / 70) * 6)
    steel = lin("#9aa0a6")
    sm = spring.astype(np.float32) * (0.6 + 0.4 * coil)
    rgb = rgb * (1 - sm[..., None]) + steel[None, None, :] * (0.5 + 0.7 * coil)[..., None] * sm[..., None]
    a = np.maximum(a, sm * 0.95)
    sh = cv2.GaussianBlur(a, (0, 0), L * 0.06)
    sh = cv2.warpAffine(sh, np.float32([[1, 0, L * 0.12], [0, 1, L * 0.11]]), (n, n))
    return Sprite(x0, y0, (rgb * a[..., None]).astype(np.float32), a.astype(np.float32), sh)


class World:
    """Escena final (coordenadas de «mundo» = el encuadre final)."""

    def __init__(self, W, H, main, neighbors, rng, title_lines, portrait=False):
        self.W, self.H = W, H
        self.main = main                 # dict(rect=(x0,y0,x1,y1)) en mundo
        self.neighbors = neighbors       # [(img_rgb, alpha, rect)]
        self.rng = rng
        self.portrait = portrait
        self.wall = plaster(W * 2, H * 2)
        # tira con el título (sobre la pared), a 2x
        self.title = self._title_strip(title_lines)
        # cordel: pasa por los perritos de cada arpillera
        pins = []
        for rect in [n[2] for n in neighbors] + [main["rect"]]:
            x0, y0, x1, y1 = rect
            pins += [(x0 + (x1 - x0) * 0.05, y0), (x1 - (x1 - x0) * 0.05, y0)]
        pins.sort()
        self.pins = pins
        pts = [(-0.3 * W, pins[0][1] - 0.06 * H)]
        for i, p in enumerate(pins):
            pts.append((p[0], p[1] - 2))
            if i + 1 < len(pins):
                q_ = pins[i + 1]
                sag = 0.012 * H + 0.03 * abs(q_[0] - p[0]) * (H / W)
                pts.append(((p[0] + q_[0]) / 2, (p[1] + q_[1]) / 2 + sag * 0.35))
        pts.append((1.3 * W, pins[-1][1] - 0.06 * H))
        self.line_pts = catmull_rom(np.array(pts), 20)

    def _title_strip(self, lines):
        """Tira de tocuyo con el título bordado en punto atrás (a 2x para el alejamiento)."""
        rng = self.rng
        W, H = self.W * 2, self.H * 2
        (x0, y0, x1, y1), texts = lines
        x0, y0, x1, y1 = [2 * v for v in (x0, y0, x1, y1)]
        shape = (int(y1 - y0) + 40, int(x1 - x0) + 40)
        ox, oy = x0 - 20, y0 - 20
        fab, _ = fabric(shape, rng, "#efe6d0", kind="plain", scale=2.0)
        m = fray_mask([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], shape, rng, offset=(ox, oy), fray=2.0)
        pf = shade(cv2.GaussianBlur(m, (0, 0), 4) * 3.2, strength=1.0, ambient=0.75)
        rgb = fab * np.clip(pf, 0.6, 1.2)[..., None] * m[..., None]
        a = m.copy()
        spr = Sprite(int(ox), int(oy), rgb.astype(np.float32), a.astype(np.float32),
                     cv2.warpAffine(cv2.GaussianBlur(m, (0, 0), 6), np.float32([[1, 0, 10], [0, 1, 9]]),
                                    (shape[1], shape[0])))
        stitches = []
        for (txt, font, hgt, cx, base, col, wd) in texts:
            strokes, _ = hershey_strokes(txt, font, hgt * 2, cx * 2, base * 2, anchor="center")
            for st in strokes:
                for _, sp in running_stitch(st, 5.5 * 2 * hgt / 30, 1.2, wd * 2, col, rng, jitter=0.25):
                    stitches.append(sp)
        nails = [knot((x0 + 26, y0 + 22), 7, "#3b3632", rng), knot((x1 - 26, y0 + 22), 7, "#3b3632", rng)]
        return spr, stitches, nails

    # ------------------------------------------------------------------------------------------------
    def render(self, main_rgb, cam_z, cam_c, t_wool_done=True):
        """cam_z: acercamiento del plano de las arpilleras (1 = encuadre final); cam_c: centro (mundo)."""
        W, H = self.W, self.H
        zw = 1 + (cam_z - 1) * 0.72             # paralaje: la pared está más lejos
        # pared (a 2x): pantalla = (mundo - c) * z + centro
        M = np.float32([[zw / 2, 0, W / 2 - cam_c[0] * zw], [0, zw / 2, H / 2 - cam_c[1] * zw]])
        out = cv2.warpAffine(self.wall, M, (W, H), flags=cv2.INTER_AREA if zw < 2 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        # tira del título en la pared
        spr, stitches, nails = self.title
        wall_layer = np.zeros((int(H * 2), int(W * 2), 3), np.float32) if False else None
        self._draw_on_plane(out, [spr] + stitches + nails, M, shadow=0.35)
        # arpilleras: sombra en la pared y luego la tela
        Mn = np.float32([[cam_z, 0, W / 2 - cam_c[0] * cam_z], [0, cam_z, H / 2 - cam_c[1] * cam_z]])
        items = [(img, a, rect) for (img, a, rect) in self.neighbors] + [(main_rgb, None, self.main["rect"])]
        for img, a, rect in items:
            self._hang(out, img, a, rect, Mn, cam_z)
        # cordel de lana roja y perritos
        u = cam_z * min(W, H) / 1080
        line = self.line_pts * cam_z + [W / 2 - cam_c[0] * cam_z, H / 2 - cam_c[1] * cam_z]
        vis = (line[:, 0] > -60) & (line[:, 0] < W + 60) & (line[:, 1] > -60) & (line[:, 1] < H + 60)
        if vis.any():
            i0, i1 = max(0, np.argmax(vis) - 2), min(len(line), len(vis) - np.argmax(vis[::-1]) + 2)
            seg = line[i0:i1]
            if len(seg) > 2:
                y = Yarn(seg, 8.0 * u * 0.42, "#c3241c", np.random.default_rng(7), fuzz=0.8, step=max(1.0, 2 * u))
                y.draw(out, 1e9, shadow=0.45)
        for (px, py) in self.pins:
            sp = np.array([px, py]) * cam_z + [W / 2 - cam_c[0] * cam_z, H / 2 - cam_c[1] * cam_z]
            if -200 < sp[0] < W + 200 and -200 < sp[1] < H + 200:
                composite(out, clothespin(sp + (0, 6 * u * 0.42), np.deg2rad(88 + (px % 7) - 3), 70 * u * 0.42,
                                          np.random.default_rng(int(px))), shadow=0.45)
        return out

    def _draw_on_plane(self, out, sprites, M, shadow=0.4):
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
            rgb = cv2.warpAffine(sp.rgb, Ml, size, flags=cv2.INTER_AREA)
            a = cv2.warpAffine(sp.a, Ml, size, flags=cv2.INTER_AREA)
            sh = cv2.warpAffine(sp.sh, Ml, size, flags=cv2.INTER_AREA) if sp.sh is not None else None
            composite(out, Sprite(x0, y0, rgb, a, sh), shadow=shadow)

    def _hang(self, out, img, alpha, rect, Mn, z):
        """Una arpillera colgada: borde de saco deshilachado, leve comba y sombra en la pared."""
        W, H = self.W, self.H
        x0, y0, x1, y1 = rect
        ih, iw = img.shape[:2]
        sx, sy = (x1 - x0) / iw * z, (y1 - y0) / ih * z
        ox = Mn[0, 0] * x0 + Mn[0, 2]
        oy = Mn[1, 1] * y0 + Mn[1, 2]
        if ox > W or oy > H or ox + iw * sx < 0 or oy + ih * sy < 0:
            return
        if alpha is None:
            alpha = self._edge_alpha(iw, ih)
        M = np.float32([[sx, 0, ox], [0, sy, oy]])
        flags = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
        rgb = cv2.warpAffine(img, M, (W, H), flags=flags)
        a = cv2.warpAffine(alpha, M, (W, H), flags=flags)
        # comba: el borde superior cuelga entre los perritos
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u_ = np.clip((xx - ox) / max(1.0, iw * sx), 0, 1)
        v_ = np.clip((yy - oy) / max(1.0, ih * sy), 0, 1)
        sag = (ih * sy) * 0.012 * (1 - (2 * u_ - 1) ** 2) * (1 - v_) ** 2
        mapy = (yy - sag).astype(np.float32)
        rgb = cv2.remap(rgb, xx, mapy, cv2.INTER_LINEAR)
        a = cv2.remap(a, xx, mapy, cv2.INTER_LINEAR)
        # pliegues suaves de tela colgada
        folds = 1 + 0.035 * np.sin(u_ * np.pi * 5.0 + 0.6) * (1 - v_) * np.sin(u_ * np.pi)
        rgb *= folds[..., None]
        # sombra en la pared
        sh = cv2.GaussianBlur(a, (0, 0), max(1.0, 9 * z * min(W, H) / 1080 * 0.42 * 2.4))
        off = 14 * z * min(W, H) / 1080 * 0.42 * 2.4
        sh = cv2.warpAffine(sh, np.float32([[1, 0, off], [0, 1, off * 0.9]]), (W, H))
        out *= (1 - 0.42 * sh * (1 - a))[..., None]
        out *= (1 - a)[..., None]
        out += rgb * a[..., None]

    def _edge_alpha(self, iw, ih):
        key = (iw, ih)
        if not hasattr(self, "_ea"):
            self._ea = {}
        if key not in self._ea:
            rng = np.random.default_rng(3)
            m = np.ones((ih, iw), np.float32)
            b = int(6 * max(iw, ih) / 1920) + 2
            m[:b, :] = 0
            m[-b:, :] = 0
            m[:, :b] = 0
            m[:, -b:] = 0
            dx = (smooth_noise((ih, iw), 5, rng) - 0.5) * 6
            dy = (smooth_noise((ih, iw), 5, rng) - 0.5) * 6
            yy, xx = np.mgrid[0:ih, 0:iw].astype(np.float32)
            m = cv2.remap(m, xx + dx, yy + dy, cv2.INTER_LINEAR)
            self._ea[key] = cv2.GaussianBlur(m, (0, 0), 0.8)
        return self._ea[key]
