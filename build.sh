#!/usr/bin/env bash
# Genera todos los entregables de «Cuando una ve» (Guía SATC).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output
# 16:9 escritorio
python -m satc_intro.render --size 1920x1080 --frames-dir build/frames_1920x1080 \
  --out output/cuando_una_ve_1920x1080.mp4 \
  --scroll-out output/cuando_una_ve_1920x1080_scroll.mp4 --scroll-width 1280 \
  --web-dir web/frames/landscape --web-width 1600
# 4:5 móvil
python -m satc_intro.render --size 1080x1350 --frames-dir build/frames_1080x1350 \
  --out output/cuando_una_ve_1080x1350.mp4 \
  --scroll-out output/cuando_una_ve_1080x1350_scroll.mp4 --scroll-width 720 \
  --web-dir web/frames/portrait --web-width 960
# Pósters (cuadro 0 y cuadro del título)
cp build/frames_1920x1080/f0000.png output/poster_1920x1080_inicio.png
cp build/frames_1920x1080/f0080.png output/poster_1920x1080_titulo.png
cp build/frames_1920x1080/f0149.png output/poster_1920x1080_final.png
cp build/frames_1080x1350/f0080.png output/poster_1080x1350_titulo.png
# Vista previa liviana en GIF
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -y -loglevel error -i output/cuando_una_ve_1920x1080.mp4 \
  -vf "fps=15,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  output/vista_previa.gif
echo "Listo: output/ y web/frames/"
