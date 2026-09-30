# «Cuando una ve» — intro animada de la Guía SATC

Animación de presentación de **5 segundos** para la *Guía del Sistema de Alerta Temprana
Comunitario* (Red Comunitaria de Alerta Energética), generada **100 % con código Python**:
tinta, aguadas de acuarela y gouache sobre **papel kraft**, con el lenguaje de una **lámina de guía
de campo**. Pensada también como *hero* con efecto de scroll para la web.

![Vista previa de «Cuando una ve»](output/vista_previa.gif)

> Vista previa en GIF (liviana). El video en alta calidad está en
> [`output/cuando_una_ve_1920x1080.mp4`](output/cuando_una_ve_1920x1080.mp4).
> Concepto completo y guion: [`CONCEPTO.md`](CONCEPTO.md)

## La metáfora, en una línea

Un **queltehue** (*Vanellus chilensis*), el centinela del campo chileno, despierta cuando en el cerro
aparece una **torre de medición de viento** —la primera huella de un proyecto eólico—; su **ojo rojo**
se vuelve **ALERTA**; las motas del propio papel (el «murmullo» de señales débiles) forman el título;
todo **encaja de un golpe**; y cada letra se vuelve **un queltehue** que baja al potrero. La «A» de
ALERTA vuela con su plancha roja de imprenta, que al posarse se contrae hasta su ojo: **la vigía de
turno**, mientras las demás duermen. *Cuando una ve, todas vuelan; duermen por turnos.*

## Archivos entregados

| Archivo | Qué es |
|---|---|
| `output/cuando_una_ve_1920x1080.mp4` | Versión escritorio 16:9, H.264, 5 s, 30 fps |
| `output/cuando_una_ve_1080x1350.mp4` | Versión móvil 4:5 |
| `output/*_scroll.mp4` | Mismas versiones con **todos los cuadros clave** (para controlar con `video.currentTime`) |
| `output/poster_*.png` | Cuadro 0 (póster), cuadro del título y cuadro final, en PNG |
| `output/vista_previa.gif` | Vista previa liviana (720 px, 15 fps) |
| `web/index.html` | Demo del *hero* con scroll (secuencia de cuadros en `<canvas>`) |
| `web/frames/landscape`, `web/frames/portrait` | Cuadros WebP para la demo (≈ 8 MB y 7 MB) |
| `satc_intro/` | El código que genera todo |

## Cómo generar la animación

Todo de una vez (MP4 16:9 y 4:5, versiones para scroll, cuadros WebP de la web y pósters):

```bash
pip install -r requirements.txt
./build.sh
```

O por partes:

```bash
# 16:9 (escritorio): cuadros + MP4 + MP4 para scroll + secuencia WebP
python -m satc_intro.render --size 1920x1080 \
    --out output/cuando_una_ve_1920x1080.mp4 \
    --scroll-out output/cuando_una_ve_1920x1080_scroll.mp4 --scroll-width 1280 \
    --web-dir web/frames/landscape --web-width 1600
# 4:5 (móvil)
python -m satc_intro.render --size 1080x1350 \
    --out output/cuando_una_ve_1080x1350.mp4 \
    --scroll-out output/cuando_una_ve_1080x1350_scroll.mp4 --scroll-width 720 \
    --web-dir web/frames/portrait --web-width 960
```

Con 4 núcleos, cada versión tarda menos de un minuto (150 cuadros). Se puede renderizar a cualquier
tamaño: la maquetación se adapta a formatos horizontales (≥ 1,2:1) o verticales.

## Cómo usarla en la web (efecto scroll)

`web/index.html` implementa el comportamiento recomendado por la verificación:

1. al cargar, **los actos I–II corren solos** hasta el título quieto (cuadros 0–84);
2. **el scroll maneja el acto III** (la suelta y el aterrizaje, cuadros 84–149);
3. al volver hacia arriba, **la bandada regresa y vuelve a formar el título**;
4. hay un `<h1>` real (accesible e indexable), versión 16:9 o 4:5 según la pantalla y respeto de
   `prefers-reduced-motion`.

Para probarla localmente: `python -m http.server` en la raíz del repositorio y abrir
`http://localhost:8000/web/`.

Alternativa con video: usar `output/*_scroll.mp4` (todos sus cuadros son clave) y asignar
`video.currentTime = progreso * 5` en el evento de scroll.

## Cómo está hecho

| Módulo | Qué hace |
|---|---|
| `paper.py` | Kraft procedural: formación nubosa, fibras, motas, grano, viñeta. Las motas que se levantarán no se hornean. |
| `watercolor.py` | Aguadas por polígonos deformados y apilados (técnica de Tyler Hobbs), oscurecimiento de bordes, granulación sobre el relieve del papel, *backruns*, gouache opaco. |
| `landscape.py` | El territorio: cordillera en neblina, cerro de la torre, cercos vivos en tres filas (con álamos y pincel seco), potrero. |
| `queltehue.py` | La vigía: ilustración con aguadas, plumilla «perdida y encontrada», cresta que se eriza, ojo rojo, grito. |
| `tower.py` | La torre de medición: única línea de regla, grosor constante y velocidad mecánica. |
| `murmur.py` | ~1300 motas del kraft (visibles desde el cuadro 0) se despegan, giran juntas y forman las palabras como puntillado. |
| `title.py` | Título riso: planchas rígidas descuadradas que tiemblan «en dos» y encajan en un clic con sobreimpulso. |
| `flock.py` | Letras → queltehues: onda desde el ojo, letras que se parten y se abren como alas, vuelo en 4 poses, aterrizaje, sueño en grupos y la posta de la plancha roja. |
| `plate.py` | Aparato de lámina: marco, cabecera, leyenda, huellas, «No confundir con», distribución. |
| `scene.py` | Maquetación 16:9 / 4:5 y línea de tiempo de los tres actos. |
| `render.py` | Render en paralelo y codificación con ffmpeg. |

## Proceso de verificación

La pieza pasó por **agentes verificadores** cuyo único criterio de aprobación era quedar
genuinamente sorprendidos:

* **Concepto y guion:** rechazado dos veces (5/10 y 7/10) y aprobado en la tercera ronda (8/10).
* **Estilo, dibujo y movimiento** (sobre renders reales): rechazado tres veces (5, 7 y 8/10; en la
  última, porque el rojo que goteaba «se leía como sangre») y aprobado en la cuarta (8,5/10).
* **Entrega final**, con ojos frescos: ver `VERIFICACION.md`.

## Créditos y licencias de recursos de diseño

* Tipografías de [Google Fonts](https://github.com/google/fonts) (SIL Open Font License, incluidas en
  `assets/fonts` con sus licencias): **Anton** (Vernon Adams), **IM Fell English** (Igino Marini).
* Paleta a partir de las combinaciones n.º 241, 243 y 126 del *A Dictionary of Color Combinations* de
  **Sanzo Wada**, vía [mattdesl/dictionary-of-colour-combinations](https://github.com/mattdesl/dictionary-of-colour-combinations).
* Técnica de acuarela generativa inspirada en el ensayo de **Tyler Hobbs** «A Generative Approach to
  Simulating Watercolor Paints».
* Textos, conceptos y citas: *Propuesta de Guía SATC* (Red Comunitaria de Alerta Energética, 2026).
