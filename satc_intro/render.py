"""Render de la animación: cuadros PNG en paralelo y codificación MP4 (H.264) con ffmpeg.

Uso:
    python -m satc_intro.render --size 1920x1080 --out output/cuando_una_ve_1920x1080.mp4
    python -m satc_intro.render --size 1080x1350 --out output/cuando_una_ve_1080x1350.mp4
    python -m satc_intro.render --size 1920x1080 --frames-dir web/frames/desktop --web
"""
import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from .scene import Scene, FPS, DURATION

_SCENE = None


def _init(scene):
    global _SCENE
    _SCENE = scene


def _render_one(args):
    i, t, path = args
    img = _SCENE.frame_srgb8(t)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return i


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render_frames(W, H, frames_dir, workers=None, seed=7, only=None):
    os.makedirs(frames_dir, exist_ok=True)
    t0 = time.time()
    scene = Scene(W, H, seed=seed)
    print(f"escena construida en {time.time() - t0:.1f}s", flush=True)
    n = int(round(DURATION * FPS))
    jobs = [(i, i / FPS, os.path.join(frames_dir, f"f{i:04d}.png")) for i in range(n)]
    if only is not None:
        jobs = [jobs[i] for i in only if i < n]
    workers = workers or max(1, (os.cpu_count() or 2))
    t1 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(scene,)) as pool:
        for k, _ in enumerate(pool.imap_unordered(_render_one, jobs, chunksize=2)):
            if k % 15 == 0:
                print(f"  cuadro {k + 1}/{len(jobs)}  ({time.time() - t1:.0f}s)", flush=True)
    print(f"{len(jobs)} cuadros en {time.time() - t1:.1f}s", flush=True)
    return n


def encode(frames_dir, out, crf=16, all_intra=False, preset="slow", width=None):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
           os.path.join(frames_dir, "f%04d.png")]
    if width:
        cmd += ["-vf", f"scale={int(width)}:-2:flags=lanczos"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-tune", "grain", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-profile:v", "high"]
    if all_intra:
        # Cada cuadro es clave: ideal para controlar la animación con el scroll (video.currentTime)
        cmd += ["-g", "1", "-keyint_min", "1", "-sc_threshold", "0"]
    else:
        cmd += ["-g", str(FPS)]
    cmd += [out]
    subprocess.run(cmd, check=True)
    print("MP4:", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


def export_web_frames(frames_dir, web_dir, width, quality=82):
    """Secuencia WebP para scroll-scrubbing en <canvas>."""
    os.makedirs(web_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    total = 0
    for k, name in enumerate(names):
        img = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if w != width:
            img = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        p = os.path.join(web_dir, f"f{k:03d}.webp")
        cv2.imwrite(p, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
        total += os.path.getsize(p)
    print(f"WebP: {len(names)} cuadros en {web_dir} ({total / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render de «Cuando una ve» (Guía SATC)")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", default=None, help="MP4 de salida")
    ap.add_argument("--scroll-out", default=None, help="MP4 con todos los cuadros clave (scroll)")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--web-dir", default=None, help="exportar secuencia WebP para la web")
    ap.add_argument("--web-width", type=int, default=1600)
    ap.add_argument("--scroll-width", type=int, default=None, help="ancho del MP4 para scroll (reescala)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args(argv)
    W, H = [int(v) for v in a.size.lower().split("x")]
    frames_dir = a.frames_dir or os.path.join("build", f"frames_{W}x{H}")
    if not a.skip_render:
        render_frames(W, H, frames_dir, workers=a.workers, seed=a.seed)
    if a.out:
        encode(frames_dir, a.out, crf=a.crf)
    if a.scroll_out:
        encode(frames_dir, a.scroll_out, crf=a.crf + 4, all_intra=True, preset="medium", width=a.scroll_width)
    if a.web_dir:
        export_web_frames(frames_dir, a.web_dir, a.web_width)


if __name__ == "__main__":
    sys.exit(main())
