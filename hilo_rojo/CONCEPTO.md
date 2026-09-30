# «Hilván» — segunda intro animada para la Guía SATC (v2)

**Pieza:** 13 segundos, 24 fps animados «en dos» (stop-motion), para el *hero* de la web con *scroll*.
Todo generado con Python y **nada vectorial**: arpillera de yute, retazos de ropa usada, lana, hilo y
luz, como una foto de un textil real.

> v2 tras el rechazo del verificador de concepto (6/10): se abandona «unir puntos con hilo rojo y
> bordar ALERTA» (tablero de conspiración y mismo remate que la pieza 1). El centro ahora es el
> **hilván a contraluz** y el remate es la **red de arpilleras colgadas del mismo hilo rojo**.

## 1. Por qué una arpillera

Porque la arpillera **ya fue una red de alerta**. Durante la dictadura, las arpilleristas de los
talleres de la **Vicaría de la Solidaridad** cosían con retazos de ropa sobre sacos harineros lo que la
prensa callaba, y esas telas salían del país a contar lo que pasaba: cada arpillera llevaba, cosido al
revés, un bolsillo con la carta de quien la hizo. Esta pieza toma esa gramática —saco harinero, retazos,
muñecas de lana, borde de punto festón— con respeto y dándole su crédito.

## 2. La metáfora: todo megaproyecto empieza como un hilván

Un **hilván** es la costura provisoria que se hace antes de coser en serio: se ve poco, se hace rápido y
**todavía se puede sacar**. Así llegan los megaproyectos: una torre en el cerro, unas estacas en el
potrero, un aviso en el cerco, una toma en el río. Por delante parecen puntadas sueltas.

Pero cuando se **tensa** el hilo, la tela se **frunce** entre dos casas: el tejido comunitario se
divide antes de que empiece la obra. Y cuando la luz pasa **por detrás** —la arpillera se vuelve
linterna— aparece lo que el revés escondía: **es un solo hilo**. Torre, estacas, aviso y toma de agua son
el mismo titular. Es el «murmullo» de la guía (3.1.3), hecho física: señales débiles que juntas son un
patrón.

La comunidad responde con **lana roja**: la tiende sobre el hilván y la fija con puntaditas (punto de
*couching*) para **hacerlo visible**, y así **se puede sacar el hilván** antes de que sea costura. La
lana sigue de casa en casa, sube y sale de la tela… y al alejarnos se ve que **es el cordel del que
cuelga esta arpillera**, junto a otras: un desierto con paneles solares, un campo cruzado por torres de
alta tensión. Es la **red de alerta**: los territorios colgados del mismo hilo.

## 3. Guion en cuatro paradas (una por pilar del SATC)

| Tiempo | Pilar | Qué pasa |
|---|---|---|
| 0,0–1,0 | **Conocer** | Póster: la arpillera terminada, con luz rasante. Un solo territorio del centro-sur: cerros, bosque nativo, el río, potreros de retazos, casas, el estanque del agua potable rural, ovejas de bouclé, vecinas y vecinos de lana. Solo se mueve el humo de una chimenea (lana). |
| 1,0–4,2 | **Vigilar** | Una aguja hilvana con hilo sintético negro y brillante: la torre de medición en el cerro, estacas en el potrero, un aviso en el cerco, una toma en el río. Luego el hilo se **tensa** y la tela se **frunce** entre dos casas. |
| 4,2–6,8 | **Alertar** | La luz pasa por detrás: el yute brilla como una linterna, los retazos se vuelven vitrales y aparece **el revés**: un solo hilo negro une todas las huellas (y se lee, en espejo, el estarcido del saco harinero). Vuelve la luz de frente. |
| 6,8–9,8 | **Responder** | La lana roja se tiende sobre el hilván, fijada con puntaditas, y sigue de casa en casa; el hilván **se saca** y el frunce se relaja (quedan los agujeros, como memoria). La lana sube y sale por el borde. |
| 9,8–13,0 | **Red** | La cámara se aleja: la lana roja es el cordel del que cuelga la arpillera con perritos de ropa, junto a otras (desierto con paneles solares; torres de alta tensión) y una tira de tela con el título. Reposo de 1,8 s. |

## 4. Para que no se note que es código

* Luz rasante (~25°) con sombras entre retazos; nada de luz plana.
* Lana de dos cabos con torsión y pelusa; puntadas con ±20 % de variación; hilo sintético con brillo.
* Bordes cortados a tijera y deshilachados; estampados que siguen la trama; retazos de ropa usada
  (cuadrillé, franela, pana, lunares).
* Stop-motion: solo se mueve lo que se toca; parpadeo leve de exposición; nada de desenfoque ni temblor
  global.
* Trama de al menos 6 px y yute calculado al doble de resolución para evitar moiré.

## 5. En la web

* El cuadro 0 es el póster. El **scroll** maneja la historia, con **cuatro paradas** (Conocer, Vigilar,
  Alertar, Responder) y su texto en HTML; al subir, la historia se deshace (el hilván vuelve, la luz se
  apaga, la cámara se acerca).
* El título real va en un `<h1>`. Versiones 16:9 y 9:16 compuestas por separado.
* Al pie, como en las arpilleras de la Vicaría, un **bolsillo** con «la carta» (llamado a leer la guía).
