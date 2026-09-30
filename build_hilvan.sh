#!/usr/bin/env bash
# Genera todos los entregables de «Hilván» (segunda intro de la Guía SATC).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output
# 16:9 escritorio: imágenes únicas + MP4 + MP4 para scroll + secuencia WebP de la web
python -m hilo_rojo.render --size 1920x1080 --frames-dir build/hilvan_1920x1080 \
  --out output/hilvan_1920x1080.mp4 \
  --scroll-out output/hilvan_1920x1080_scroll.mp4 --scroll-width 1280 \
  --web-dir web/hilvan/frames/landscape --web-width 1280 --web-quality 62
# 9:16 teléfono (compuesta aparte)
python -m hilo_rojo.render --size 1080x1920 --frames-dir build/hilvan_1080x1920 \
  --out output/hilvan_1080x1920.mp4 \
  --scroll-out output/hilvan_1080x1920_scroll.mp4 --scroll-width 720 \
  --web-dir web/hilvan/frames/tall --web-width 720 --web-quality 60
# El revés con el bolsillo y la carta (llamado a leer la guía)
python -m hilo_rojo.bolsillo web/hilvan/bolsillo.webp
# Pósters: cuadro 0 (arpillera colgada), el contraluz y el cuadro final con el título
cp build/hilvan_1920x1080/u0000.png output/hilvan_poster_1920x1080_inicio.png
cp build/hilvan_1920x1080/u0068.png output/hilvan_poster_1920x1080_contraluz.png
cp build/hilvan_1920x1080/u0167.png output/hilvan_poster_1920x1080_final.png
cp build/hilvan_1080x1920/u0000.png output/hilvan_poster_1080x1920_inicio.png
cp build/hilvan_1080x1920/u0167.png output/hilvan_poster_1080x1920_final.png
# Vista previa liviana en GIF
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -y -loglevel error -i output/hilvan_1920x1080.mp4 \
  -vf "fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  output/hilvan_vista_previa.gif
echo "Listo: output/hilvan_* y web/hilvan/"
