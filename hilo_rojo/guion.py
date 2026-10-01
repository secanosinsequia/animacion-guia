"""Guion de la voz en off de «Hilván» (16,92 s): cada frase con su ventana de tiempo, lo que se ve en pantalla
y cómo decirla. Es la única fuente: de aquí salen la voz sintetizada, los subtítulos (SRT y VTT) y la tabla
del guion escrito.

Los tiempos siguen la línea de tiempo de `scene.T` (imagen k = k/12 s).
"""

# (inicio, fin, texto, en pantalla, intención)
LINEAS = [
    (0.12, 4.58, "Todo megaproyecto empieza como un hilván: puntadas sueltas que parecen nada.",
     "La aguja cuelga sobre la arpillera, baja y hilvana estacas, un AVISO, una cañería y la torre del cerro; "
     "al tensarse el hilo, la tela se frunce.",
     "Cálida y cercana, como quien cuenta algo que vio. Una pausa breve en los dos puntos; «nada» cae justo "
     "cuando la tela se frunce."),
    (4.72, 5.92, "Pero es un solo hilo,",
     "Se apaga la sala. A contraluz, una lámpara de mano sigue la hebra por las señales.",
     "Baja la voz: es la revelación. Sin cerrar la frase."),
    (6.02, 7.40, "y llega hasta tu puerta.",
     "La lámpara baja por las torres escondidas hasta la casa roja; después, la línea completa.",
     "Íntima y firme. «Tu puerta» con peso, mirando a quien escucha."),
    (7.85, 9.52, "Lo marcamos de mano en mano,",
     "Clic: vuelve la luz. La lana roja calca la ruta, con un nudo en cada torre, y pasa de mano en mano; "
     "se encienden las ventanas.",
     "Más luz y energía: es la comunidad que responde."),
    (9.66, 11.80, "y lo sacamos antes de que sea costura.",
     "La vigía tira del hilván y la torre se descose; toda la ruta se frunce y se suelta. A contraluz, el "
     "revés queda vacío: solo los pinchazos.",
     "Decidida, sin dramatismo. «Costura» llega sobre el revés vacío."),
    (11.95, 14.20, "Cada comunidad, un hilo de la red.",
     "La cámara se aleja: del cordel cuelgan otras arpilleras y el rojo corre de una a otra.",
     "Abierta y esperanzada."),
    (14.40, 16.85, "Sistema de Alerta Temprana Comunitario.",
     "La aguja del comienzo baja por el hilo rojo y borda la última «a» de «Alerta».",
     "Clara y pausada, como una firma."),
]


def _ts(t, sep):
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def _envolver(txt, ancho=40):
    """Hasta dos renglones de ~40 caracteres (lo cómodo para leer un subtítulo)."""
    if len(txt) <= ancho:
        return txt
    pal = txt.split()
    mejor, corte = None, 1
    for k in range(1, len(pal)):
        a, b = " ".join(pal[:k]), " ".join(pal[k:])
        costo = abs(len(a) - len(b)) - (8 if a.endswith((":", ",")) else 0)
        if mejor is None or costo < mejor:
            mejor, corte = costo, k
    return " ".join(pal[:corte]) + "\n" + " ".join(pal[corte:])


def srt(lineas):
    """Subtítulos SRT. lineas: [(inicio, fin, texto)]."""
    out = []
    for i, (a, b, txt) in enumerate(lineas, 1):
        out.append(f"{i}\n{_ts(a, ',')} --> {_ts(b, ',')}\n{_envolver(txt)}\n")
    return "\n".join(out)


def vtt(lineas):
    out = ["WEBVTT", ""]
    for (a, b, txt) in lineas:
        out.append(f"{_ts(a, '.')} --> {_ts(b, '.')}\n{_envolver(txt)}\n")
    return "\n".join(out)


def _seg(t):
    return f"{t:.2f}".replace(".", ",")


def tabla_md(lineas_reales):
    """Tabla del guion con los tiempos reales de la voz (inicio y fin de cada frase ya ubicada)."""
    filas = ["| # | Tiempo | Voz en off | En pantalla | Cómo decirla |", "|---|---|---|---|---|"]
    for i, ((a, b, txt), (_, _, _, pantalla, intencion)) in enumerate(zip(lineas_reales, LINEAS), 1):
        filas.append(f"| {i} | {_seg(a)}–{_seg(b)} s | «{txt}» | {pantalla} | {intencion} |")
    return "\n".join(filas)
