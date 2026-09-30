"""Render de «Hilván»: imágenes únicas en paralelo (12 por segundo, stop-motion «en dos») y MP4.

Uso:
    python -m hilo_rojo.render --size 1920x1080 --out output/hilvan_1920x1080.mp4
    python -m hilo_rojo.render --size 1080x1920 --out output/hilvan_1080x1920.mp4 \\
        --scroll-out output/hilvan_1080x1920_scroll.mp4 --web-dir web/hilvan/frames/tall
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2

from satc_intro.render import ffmpeg_bin
from .scene import Scene, UFPS, FPS, DURATION

_SCENE = None


def _render_one(args):
    k, path = args
    img = _SCENE.frame_srgb8(k / UFPS)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return k


def n_unique():
    return int(round(DURATION * UFPS))


def render_frames(W, H, frames_dir, workers=None, only=None):
    global _SCENE
    os.makedirs(frames_dir, exist_ok=True)
    t0 = time.time()
    _SCENE = Scene(W, H)
    print(f"escena construida en {time.time() - t0:.1f}s", flush=True)
    jobs = [(k, os.path.join(frames_dir, f"u{k:04d}.png")) for k in range(n_unique())]
    if only is not None:
        jobs = [jobs[k] for k in only if k < len(jobs)]
    workers = workers or max(1, (os.cpu_count() or 2))
    t1 = time.time()
    # fork: los procesos heredan la escena ya construida (sin copiarla)
    with mp.get_context("fork").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render_one, jobs, chunksize=1)):
            if i % 12 == 0:
                print(f"  imagen {i + 1}/{len(jobs)}  ({time.time() - t1:.0f}s)", flush=True)
    print(f"{len(jobs)} imágenes en {time.time() - t1:.1f}s", flush=True)


def encode(frames_dir, out, crf=17, all_intra=False, preset="slow", width=None):
    """Cada imagen única dura dos cuadros a 24 fps (animación «en dos»)."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    vf = [f"fps={FPS}"]
    if width:
        vf.append(f"scale={int(width)}:-2:flags=lanczos")
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-framerate", str(UFPS), "-i",
           os.path.join(frames_dir, "u%04d.png"), "-vf", ",".join(vf),
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-tune", "grain",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-profile:v", "high"]
    if all_intra:
        cmd += ["-g", "1", "-keyint_min", "1", "-sc_threshold", "0"]
    else:
        cmd += ["-g", str(FPS)]
    cmd += [out]
    subprocess.run(cmd, check=True)
    print("MP4:", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


def export_web_frames(frames_dir, web_dir, width, quality=74):
    """Secuencia WebP (una por imagen única) para mover con el scroll en un <canvas>."""
    os.makedirs(web_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(frames_dir) if f.startswith("u") and f.endswith(".png"))
    total = 0
    for k, name in enumerate(names):
        img = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if w != width:
            img = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        p = os.path.join(web_dir, f"f{k:03d}.webp")
        cv2.imwrite(p, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
        total += os.path.getsize(p)
    print(f"WebP: {len(names)} imágenes en {web_dir} ({total / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render de «Hilván» (Guía SATC)")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scroll-out", default=None, help="MP4 con todos los cuadros clave (scroll)")
    ap.add_argument("--scroll-width", type=int, default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--web-dir", default=None)
    ap.add_argument("--web-width", type=int, default=1600)
    ap.add_argument("--web-quality", type=int, default=74)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args(argv)
    W, H = [int(v) for v in a.size.lower().split("x")]
    frames_dir = a.frames_dir or os.path.join("build", f"hilvan_{W}x{H}")
    if not a.skip_render:
        render_frames(W, H, frames_dir, workers=a.workers)
    if a.out:
        encode(frames_dir, a.out, crf=a.crf)
    if a.scroll_out:
        encode(frames_dir, a.scroll_out, crf=a.crf + 7, all_intra=True, preset="medium", width=a.scroll_width)
    if a.web_dir:
        export_web_frames(frames_dir, a.web_dir, a.web_width, quality=a.web_quality)


if __name__ == "__main__":
    sys.exit(main())
