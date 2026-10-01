# «Hilván» — guion de la voz en off

Voz en off para la intro animada «Hilván» (16,92 s). Narra lo que va pasando en la arpillera, frase por
frase, con el tiempo exacto de cada una. Es una voz de mujer en español latinoamericano, con música original
debajo.

- **Videos con voz y música:** [`output/hilvan_1920x1080_voz.mp4`](output/hilvan_1920x1080_voz.mp4) (16:9) y
  [`output/hilvan_1080x1920_voz.mp4`](output/hilvan_1080x1920_voz.mp4) (9:16).
- **Subtítulos:** [`output/hilvan_voz.srt`](output/hilvan_voz.srt) y [`output/hilvan_voz.vtt`](output/hilvan_voz.vtt).
- **Audio suelto:** `output/audio/hilvan_mezcla.wav` (y `.m4a`), más las pistas separadas
  `hilvan_voz.wav` y `hilvan_musica.wav`.

## El texto completo

> Todo megaproyecto empieza como un hilván: puntadas sueltas que parecen nada.
> Pero es un solo hilo, y llega hasta tu puerta.
> Lo marcamos de mano en mano, y lo sacamos antes de que sea costura.
> Cada comunidad, un hilo de la red.
> Sistema de Alerta Temprana Comunitario.

Son 47 palabras en 16,9 s: la voz va casi sin pausas, así que no queda espacio para agregar texto.

## Frase por frase

Los tiempos son los de la voz ya ubicada en el video (inicio y fin de lo hablado).

| # | Tiempo | Voz en off | En pantalla | Cómo decirla |
|---|---|---|---|---|
| 1 | 0,15–4,59 s | «Todo megaproyecto empieza como un hilván: puntadas sueltas que parecen nada.» | La aguja cuelga sobre la arpillera, baja y hilvana estacas, un AVISO, una cañería y la torre del cerro; al tensarse el hilo, la tela se frunce. | Cálida y cercana, como quien cuenta algo que vio. Una pausa breve en los dos puntos; «nada» cae justo cuando la tela se frunce. |
| 2 | 4,75–5,84 s | «Pero es un solo hilo,» | Se apaga la sala. A contraluz, una lámpara de mano sigue la hebra por las señales. | Baja la voz: es la revelación. Sin cerrar la frase. |
| 3 | 6,05–7,32 s | «y llega hasta tu puerta.» | La lámpara baja por las torres escondidas hasta la casa roja; después, la línea completa. | Íntima y firme. «Tu puerta» con peso, mirando a quien escucha. |
| 4 | 7,88–9,39 s | «Lo marcamos de mano en mano,» | Clic: vuelve la luz. La lana roja calca la ruta, con un nudo en cada torre, y pasa de mano en mano; se encienden las ventanas. | Más luz y energía: es la comunidad que responde. |
| 5 | 9,69–11,81 s | «y lo sacamos antes de que sea costura.» | La vigía tira del hilván y la torre se descose; toda la ruta se frunce y se suelta. A contraluz, el revés queda vacío: solo los pinchazos. | Decidida, sin dramatismo. «Costura» llega sobre el revés vacío. |
| 6 | 11,98–14,21 s | «Cada comunidad, un hilo de la red.» | La cámara se aleja: del cordel cuelgan otras arpilleras y el rojo corre de una a otra. | Abierta y esperanzada. |
| 7 | 14,42–16,83 s | «Sistema de Alerta Temprana Comunitario.» | La aguja del comienzo baja por el hilo rojo y borda la última «a» de «Alerta». | Clara y pausada, como una firma. |

Los respiros sin voz también cuentan:
- **7,32–7,88 s:** el clic de la lámpara y el tubo de la sala que titila.
- Entre las frases 5 y 6: el revés vacío y la vuelta al día.

## La música

Es una tonada original en 6/8 (♩. = 90), compuesta y sintetizada para este video. No usa grabaciones ni
muestras de nadie, así que es libre de derechos. Los instrumentos son guitarra de nylon y charango pulsados,
bombo legüero, un bordón grave y campanitas de vidrio. Va unos 12 dB debajo de la voz y sube en los respiros.

| Tiempo | Qué hace la música |
|---|---|
| 0,0–4,0 s | Guitarra arpegiada en re menor (re menor, si bemol, sol menor, la). Se oyen apenas las puntadas de la aguja. |
| 4,0–4,62 s | El hilo se tensa: un acorde tenso y suave. Justo después de «nada», un golpe de bombo cuando se apaga la sala. |
| 4,7–7,67 s | Un bordón grave (re y la) en la sala a oscuras. Cada torre escondida que encuentra la lámpara enciende una campanita, cada vez más aguda; la casa roja trae un latido de bombo. La línea completa crece en un acorde. |
| 7,67–8,0 s | Se corta todo con el clic de la lámpara. En la penumbra se oye el tubo de la sala, que hace tic-tic y prende. |
| 8,0–9,33 s | La lana: la guitarra pasa a re mayor y cada ventana que se enciende suena como una campanada de charango. |
| 9,33–10,9 s | El tirón: si menor y mi menor, con bombo en los tirones. Se oye el hilo que se descose, puntada por puntada, y un rasgón largo cuando la ruta se suelta. |
| 10,9–11,83 s | El revés vacío: un acorde de vidrio y puntitos de luz, entre dos clics de la lámpara. |
| 11,83–16,0 s | La red: guitarra, charango rasgueado y bombo legüero (re, sol, la), con dos campanadas cuando el rojo llega a las vecinas. Las puntadas de la aguja sobre la «a», muy bajito. |
| 16,0–16,92 s | Acorde final en re mayor, que suena hasta el último cuadro. |

## La voz

La voz es una síntesis neuronal hecha con [Piper](https://github.com/OHF-Voice/piper1-gpl), con la voz
`es_MX-claude-high`: mujer, español de México. Su fonética latinoamericana pronuncia la «c» y la «z» como «s».
- Cada frase se ajustó a su ventana con una velocidad natural, entre 1,04 y 1,19 veces la normal.
- De varias tomas por frase se eligió la mejor con medidas objetivas: duración, entonación y nitidez.
- Las tomas elegidas quedaron en `hilo_rojo/audio/voz_01.wav` a `voz_07.wav`.

**Antes de publicar:** revisar la licencia de la voz `es_MX-claude-high` en su `MODEL_CARD`
(huggingface.co/rhasspy/piper-voices). Piper pide revisar la licencia de cada voz, porque algunas tienen
condiciones propias.

### Para grabar con otra voz (una locutora o tu Voice Studio)

1. Graba cada frase de la tabla por separado, con la intención indicada. Las frases 2 y 3, y 4 y 5, son una
   sola oración partida: la primera mitad queda abierta, sin cerrar el tono.
2. Guárdalas como `voz_01.wav` … `voz_07.wav` (WAV, mono o estéreo, a cualquier frecuencia de muestreo) en
   `hilo_rojo/audio/`, reemplazando las actuales.
3. Corre `python -m hilo_rojo.sonido`. Cada frase se ubica sola en su tiempo, se ecualiza, se mezcla con la
   música y se rehacen los MP4 con voz, el audio suelto y los subtítulos.

Si una frase grabada no cabe en su ventana, el programa avisa. La voz sintetizada actual sirve como pista
guía de ritmo.

Para rehacer la voz sintética: `REHACER_VOZ=1 python -m hilo_rojo.sonido --voz ruta/es_MX-claude-high.onnx`.
