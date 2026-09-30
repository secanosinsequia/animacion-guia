# «Hilván» — segunda intro animada de la Guía SATC

Una **arpillera chilena** animada en *stop-motion*: retazos de ropa usada cosidos sobre un saco
harinero, lana, hilo, una aguja y luz. **15 segundos**, generada **100 % con Python y en raster**
(nada vectorial: cada tela es un campo de altura iluminado con luz rasante, cada puntada un tubo de
hilo con torsión, cada lana tiene pelusa). Pensada como *hero* con scroll para la web.

![Vista previa de «Hilván»](output/hilvan_vista_previa.gif)

> Video en alta calidad: [`output/hilvan_1920x1080.mp4`](output/hilvan_1920x1080.mp4) (16:9) y
> [`output/hilvan_1080x1920.mp4`](output/hilvan_1080x1920.mp4) (9:16, compuesto aparte para teléfono).
> Concepto y guion: [`hilo_rojo/CONCEPTO.md`](hilo_rojo/CONCEPTO.md).

## La metáfora, en una línea

Todo megaproyecto **empieza como un hilván**: puntadas provisorias, que por delante parecen sueltas
—unas estacas, un aviso en el cerco, una toma en el río, el contorno de una torre que todavía no
existe—. Cuando se tensa el hilo, **la tela se frunce entre las casas**. A contraluz se ve el revés:
**es un solo hilo**. La comunidad lo marca con **lana roja**, avisa casa por casa, **saca el hilván
antes de que sea costura**, y la lana sube al cordel: de él cuelgan otras arpilleras, otros
territorios, y el rojo corre de una a otra. **Una red de alerta.**

La forma viene de las **arpilleras de los talleres de la Vicaría de la Solidaridad**, que cosieron con
retazos sobre sacos harineros lo que la prensa callaba y que llevaban, al revés, un bolsillo con la
carta de quien las hizo. La pieza toma esa gramática con respeto y le da su crédito (en la web, el
llamado a leer la guía es ese bolsillo con su carta).

## Guion (15 s, 24 fps animados «en dos»: 12 imágenes únicas por segundo)

| Tiempo | Pilar | Qué pasa |
|---|---|---|
| 0,0–1,0 | **Conocer** | Póster: la arpillera cuelga de un cordel de cáñamo con perritos de ropa. Un valle del centro-sur con la cordillera nevada, araucarias, río, potreros, el estanque del APR, casas y vecinas de lana. Una aguja enhebrada cuelga de su hilo desde fuera del cuadro y se mece: algo está por coserse. |
| 1,0–4,4 | **Vigilar** | La aguja hilvana, en orden, las señales tempranas: cuatro estacas de topógrafo con cinta naranja, un **AVISO** prendido al cerco con alfiler de gancho, una toma de agua (cañería gruesa hilvanada y su caseta) y, al final, la **torre de alta tensión solo como contorno de puntadas largas**, a futuro. La aguja tira del hilo: la tela se frunce en pliegues a lo largo del hilo y las casas de los extremos se acercan. |
| 4,4–7,5 | **Alertar** | Se apaga la sala. Una lámpara recorre el revés y se detiene en cada huella; luego toda la tela es linterna: el yute brilla, los retazos son vitrales y aparece **un solo hilo** uniendo las señales; cada agujero del hilván es una estrellita (la torre proyectada, una constelación). En el cielo se lee el estarcido del saco: HARINERA. La luz del día vuelve de a poco. |
| 7,5–11,0 | **Responder** | La lana roja nace donde empieza el hilo y lo recorre por delante; de ella salen ramales a cada casa, cada vez más rápido: se encienden las ventanas y la gente levanta los brazos. La lana sube y se anuda al cordel. Una vecina tira del hilván y lo junta en un ovillo a sus pies: salen las señales y, al último, puntada a puntada, la torre. Quedan los agujeros. |
| 11,2–15,0 | **Red** | La cámara se aleja por pasos, con paralaje: el cordel sostiene otras arpilleras (un desierto con paneles solares, un campo cruzado por torres). Desde el nudo, **el rojo recorre el cordel** de arpillera en arpillera y cada una se mece al recibirlo. Abajo, la tira de tocuyo con el título bordado. |

## Archivos

| Archivo | Qué es |
|---|---|
| `output/hilvan_1920x1080.mp4` | 16:9, H.264, 15 s, 24 fps («en dos») |
| `output/hilvan_1080x1920.mp4` | 9:16 para teléfono, compuesto aparte |
| `output/hilvan_*_scroll.mp4` | Las mismas, con todos los cuadros clave (para `video.currentTime`) |
| `output/hilvan_poster_*.png` | Cuadro 0, el contraluz y el cuadro final |
| `output/hilvan_vista_previa.gif` | Vista previa liviana |
| `web/hilvan/index.html` | La intro con scroll: 7 paradas con texto en HTML, `<h1>` real, bolsillo con la carta |
| `web/hilvan/frames/landscape`, `tall` | Imágenes WebP de la web (16:9 y 9:16) |
| `web/hilvan/bolsillo.webp` | El revés de la arpillera con el bolsillo y la carta (llamado a leer la guía) |
| `hilo_rojo/` | El código |

## Cómo generarla

```bash
pip install -r requirements.txt
./build_hilvan.sh          # MP4 16:9 y 9:16, versiones para scroll, WebP de la web, pósters, GIF
```

Con 4 núcleos, cada formato tarda ≈ 1,5 min (180 imágenes únicas). Para probar la web:
`python -m http.server` en la raíz y abrir `http://localhost:8000/web/hilvan/`.

## Cómo está hecho (`hilo_rojo/`)

| Módulo | Qué hace |
|---|---|
| `textile.py` | Tejidos procedurales con relieve y luz rasante: yute grueso con hebras de grosor lognormal y nudos (calculado al doble de resolución, sin moiré), algodón, cuadrillé, lunares, rayas, estampado, franela, pana, fieltro y raso; transmisión de luz para el contraluz. |
| `thread.py` | Hilos como tubos con torsión, cabos y brillo; puntadas con comba; lana continua con pelusa y puntadas de fijación (*couching*); nudos franceses, punto atrás, satén y festón. |
| `pieces.py` | Retazos cortados a tijera (contorno ondulado, muescas, hebras sueltas), gastados (desteñido, *pilling*, manchas), acolchados y fijados con puntada corrida; se posan en *stop-motion*. |
| `figures.py` | Casas distintas entre sí, árboles de fieltro, araucarias, cordillera nevada, sol, nubes, muñecas con cabeza de lana enrollada y poses, la aguja, texto bordado con tildes (fuentes Hershey de un solo trazo). |
| `signals.py` | Las huellas del megaproyecto: estacas, AVISO con alfiler de gancho, toma de agua con caseta y la torre hilvanada. |
| `territory.py` | La arpillera (16:9 y 9:16 compuestas por separado), el estarcido del saco y las arpilleras vecinas. |
| `scene.py` | La línea de tiempo: aguja, frunce, lámpara y linterna, lana y ramales, reacciones de la gente, el descosido. |
| `world.py` | La pared encalada, el cordel, los perritos, las telas colgadas (comba y pliegues), la tira del título, la cámara por pasos con paralaje por capas, el frente rojo y los péndulos, el grano y el bamboleo de película. |
| `bolsillo.py` | El revés con el bolsillo y la carta, para la web. |
| `render.py` | Render en paralelo y codificación con ffmpeg. |

## Verificación

(ver [`hilo_rojo/VERIFICACION.md`](hilo_rojo/VERIFICACION.md))
