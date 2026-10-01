"""Música original de «Hilván» (16,92 s), sintetizada aquí mismo con numpy: libre de derechos.

Una tonada en 6/8 (negra con punto = 90; un compás = 1,333 s; corchea = 0,222 s): a ese pulso caen los
momentos del video (la torre a 3,33 s, el tirón del hilo a 4,0, la lana a 8,0, el tirón de la vigía a 9,33,
la red a 12,0, el final a 16,0).
- Conocer y Vigilar (0–4,6 s): guitarra de nylon arpegiada en re menor; al tensarse el hilo, un acorde
  tenso con bombo legüero.
- Alertar (4,65–7,9 s): se apaga la sala y queda un bordón grave; cada torre escondida que encuentra la
  lámpara enciende una campanita de vidrio, cada vez más aguda; la línea completa crece en un acorde que
  se corta con el clic de la lámpara; titila el tubo de la sala.
- Responder (8,0–11,8 s): la guitarra pasa a re mayor y el charango da una campanada por cada ventana que
  se enciende; el tirón con bombo y el hilo que se descose; el revés vacío en un acorde de vidrio con
  puntitos de luz.
- Red (11,9–16,9 s): guitarra, charango rasgueado y bombo en re mayor (re, sol, la, re) y un acorde final.

Instrumentos: cuerdas pulsadas por Karplus-Strong (con afinación fraccional y cuerpo de madera), bombo
legüero (parche y aro), bordón aditivo, campanas de vidrio y ruidos de sala.
"""
import numpy as np
from scipy import signal

BPM = 90.0
BAR = 2 * 60.0 / BPM            # compás de 6/8: dos negras con punto
CORCHEA = BAR / 6


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _biquad(kind, f0, sr, q=0.707, gain_db=0.0):
    from .sonido import biquad
    return biquad(kind, f0, sr, q, gain_db)


def _filt(x, kind, f0, sr, q=0.707, gain_db=0.0):
    b, a = _biquad(kind, f0, sr, q, gain_db)
    return signal.lfilter(b, a, x, axis=0)


class Pista:
    """Una pista estéreo donde se suman eventos (mono con paneo o estéreo)."""

    def __init__(self, dur, sr):
        self.sr = sr
        self.x = np.zeros((int(np.ceil(dur * sr)) + sr * 3, 2))

    def add(self, t, y, gain=1.0, pan=0.0):
        i0 = int(round(t * self.sr))
        if i0 >= len(self.x):
            return
        if y.ndim == 1:
            l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            y = np.stack([y * l, y * r], 1) * np.sqrt(2)
        n = min(len(y), len(self.x) - i0)
        if i0 < 0:
            y, n, i0 = y[-i0:], n + i0, 0
        self.x[i0:i0 + n] += y[:n] * gain


# --- cuerdas pulsadas -----------------------------------------------------------------------------------
def cuerda(freq, dur, sr, decay=2.5, brillo=0.6, pos=0.17, s=0.5, seed=0):
    """Karplus-Strong con afinación fraccional (interpolación lineal en el lazo), procesada por bloques del
    largo del lazo (cada bloque depende solo del anterior: se calcula de una vez con numpy)."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    P = sr / freq
    N = int(np.floor(P - s))
    d = P - s - N
    g = 10 ** (-3.0 / (freq * decay))                 # pierde 60 dB en `decay` segundos
    hN, hN1, hN2 = g * (1 - s) * (1 - d), g * ((1 - s) * d + s * (1 - d)), g * s * d
    exc = rng.uniform(-1, 1, N)
    a1 = float(np.clip(1 - brillo, 0.0, 0.95))            # excitación más oscura cuanto más suave
    exc = signal.lfilter([1 - a1], [1, -a1], exc)
    k = max(1, int(pos * N))                              # posición del punteo (filtro peine)
    exc = exc - np.concatenate([np.zeros(k), exc[:-k]])
    exc -= exc.mean()
    y = np.zeros(n + N + 2)
    off = N + 2
    x = np.zeros(n)
    x[:min(n, N)] = exc[:min(n, N)]
    for i0 in range(0, n, N):
        i1 = min(n, i0 + N)
        j = np.arange(i0, i1) + off
        y[j] = x[i0:i1] + hN * y[j - N] + hN1 * y[j - N - 1] + hN2 * y[j - N - 2]
    out = y[off:]
    # un toque de ataque (la uña/yema contra la cuerda) y apagado suave al final
    fade = np.ones(n)
    nf = min(n, int(0.04 * sr))
    fade[-nf:] = np.linspace(1, 0, nf)
    return out * fade / (np.abs(out[: int(0.05 * sr)]).max() + 1e-9)


def guitarra(freq, vel, sr, dur=None, seed=0):
    dec = float(np.interp(np.log2(freq), np.log2([80, 160, 330, 660]), [3.4, 2.8, 2.0, 1.3]))
    dur = dur or min(4.0, dec * 1.1)
    return cuerda(freq, dur, sr, decay=dec, brillo=0.35 + 0.45 * vel, pos=0.16, s=0.5, seed=seed) * vel


def charango(freq, vel, sr, dur=None, seed=0):
    """Doble cuerda (cada orden, apenas desafinada entre sí), brillante y corta."""
    dur = dur or 1.6
    a = cuerda(freq * 2 ** (3 / 1200), dur, sr, decay=1.3, brillo=0.6 + 0.35 * vel, pos=0.12, s=0.35, seed=seed)
    b = cuerda(freq * 2 ** (-3 / 1200), dur, sr, decay=1.2, brillo=0.6 + 0.35 * vel, pos=0.14, s=0.35,
               seed=seed + 1)
    return (a + b) * 0.5 * vel


def cuerpo_guitarra(x, sr):
    """Caja de madera: aire (~98 Hz), tapa (~205 Hz) y la madera (~410 Hz); agudos tibios."""
    x = _filt(x, "peak", 98, sr, 2.2, 5.0)
    x = _filt(x, "peak", 205, sr, 1.6, 3.0)
    x = _filt(x, "peak", 410, sr, 1.2, 2.0)
    x = _filt(x, "highshelf", 3500, sr, 0.7, -4.0)
    return _filt(x, "hp", 60, sr)


def cuerpo_charango(x, sr):
    x = _filt(x, "peak", 480, sr, 1.4, 3.0)
    x = _filt(x, "peak", 1250, sr, 1.2, 2.0)
    x = _filt(x, "highshelf", 6000, sr, 0.7, -3.0)
    return _filt(x, "hp", 200, sr)


# --- percusión, bordón, vidrio y sala --------------------------------------------------------------------
def bombo(vel, sr, seed=0):
    """Bombo legüero: parche grave con caída de altura, un segundo modo y el golpe del cuero."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(1.0 * sr)) / sr
    f = 50 + 42 * np.exp(-t / 0.05)
    ph = 2 * np.pi * np.cumsum(f) / sr
    cuerpo = np.sin(ph) * np.exp(-t / 0.33) + 0.35 * np.sin(1.53 * ph) * np.exp(-t / 0.16)
    golpe = _filt(rng.normal(0, 1, len(t)), "bp", 650, sr, 0.8) * np.exp(-t / 0.022)
    y = cuerpo + 0.5 * golpe
    y[: int(0.002 * sr)] *= np.linspace(0, 1, int(0.002 * sr))
    return y * vel


def aro(vel, sr, seed=0):
    """El golpe en el aro de madera del bombo."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.12 * sr)) / sr
    y = _filt(rng.normal(0, 1, len(t)), "bp", 2100, sr, 2.5) * np.exp(-t / 0.018)
    y += 0.6 * np.sin(2 * np.pi * 920 * t) * np.exp(-t / 0.03)
    return y * vel * 0.5


def bordon(notas, dur, sr, seed=0, ataque=0.6):
    """Bordón grave y vivo (suma de parciales con respiración lenta): la sala a oscuras."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros(len(t))
    for f0 in notas:
        for k in range(1, 13):
            amp = (1 / k ** 1.15) * np.exp(-k / 7)
            vib = 1 + 0.0015 * np.sin(2 * np.pi * rng.uniform(0.15, 0.4) * t + rng.uniform(0, 6))
            mod = 0.75 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.1, 0.35) * t + rng.uniform(0, 6))
            y += amp * mod * np.sin(2 * np.pi * f0 * k * np.cumsum(vib) / sr + rng.uniform(0, 6))
    env = np.minimum(1, t / ataque)
    return y * env / (len(notas) * 2.2)


def acorde_cuerdas(notas, dur, sr, ataque, seed=0):
    """Un acorde que crece (cuerdas aditivas con coro): la línea completa."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros((len(t), 2))
    for f0 in notas:
        for v in range(3):
            det = 2 ** (rng.uniform(-6, 6) / 1200)
            pan = rng.uniform(-0.6, 0.6)
            vib = 1 + 0.002 * np.sin(2 * np.pi * rng.uniform(4.5, 5.5) * t + rng.uniform(0, 6))
            ph = 2 * np.pi * f0 * det * np.cumsum(vib) / sr + rng.uniform(0, 6)
            s_ = sum(np.sin(k * ph) / k ** 1.4 * np.exp(-k / 5) for k in range(1, 9))
            y[:, 0] += s_ * np.cos((pan + 1) * np.pi / 4)
            y[:, 1] += s_ * np.sin((pan + 1) * np.pi / 4)
    env = (np.minimum(1, t / ataque) ** 2)[:, None]
    return y * env / (len(notas) * 3)


def vidrio(freq, vel, sr, dur=2.6, seed=0):
    """Campanita de vidrio (parciales inarmónicos de copa)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros(len(t))
    for r, a, d in ((1.0, 1.0, 2.0), (2.32, 0.22, 1.1), (4.25, 0.07, 0.55), (6.63, 0.025, 0.3)):
        for dd in (0.0, 0.7):                                       # batido suave
            y += a * 0.5 * np.sin(2 * np.pi * (freq * r + dd) * t + rng.uniform(0, 6)) * np.exp(-t / d)
    y *= np.minimum(1, t / 0.004)
    return y * vel


def clic(vel, sr, seed=0, tono=3200):
    """El interruptor de la lámpara (dos chasquidos muy juntos y un golpecito grave)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.08 * sr)) / sr
    y = np.zeros(len(t))
    for d0 in (0.0, 0.006):
        i = int(d0 * sr)
        n = rng.normal(0, 1, len(t) - i)
        y[i:] += _filt(n, "bp", tono, sr, 1.8) * np.exp(-t[: len(t) - i] / 0.004)
    y += 0.6 * np.sin(2 * np.pi * 140 * t) * np.exp(-t / 0.012)
    return y * vel


def tubo(dur, sr, seed=0):
    """El tubo de la sala: el arrancador hace tic-tic, prende con un zumbido y titila una vez."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    hum = sum(np.sin(2 * np.pi * 100 * k * t) / k ** 1.5 for k in range(1, 7))
    hum += 0.3 * _filt(rng.normal(0, 1, len(t)), "bp", 6000, sr, 0.7)
    env = np.interp(t, [0, 0.083, 0.09, 0.166, 0.17, 0.25, 0.6, dur], [0, 0, 1, 1, 0.55, 1, 0.7, 0])
    y = hum * env * 0.25
    for t0 in (0.0, 0.083):
        i = int(t0 * sr)
        c = clic(0.6, sr, seed + int(t0 * 1000), tono=2600)
        y[i:i + len(c)] += c[: len(y) - i]
    return y


def descosido(dur, sr, n, acel=1.6, seed=0):
    """Hilo que sale de la tela, puntada por puntada: chasquidos de fibra cada vez más seguidos, sobre un
    roce continuo."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    y = 0.25 * _filt(rng.normal(0, 1, len(t)), "bp", 2400, sr, 0.9) * (0.4 + 0.6 * np.sin(np.pi * t / dur))
    for k in range(n):
        tk = dur * (k / n) ** (1 / acel)
        i = int(tk * sr)
        L = int(0.012 * sr)
        if i + L >= len(y):
            break
        g = _filt(rng.normal(0, 1, L), "bp", rng.uniform(1800, 4200), sr, 2.0) * np.exp(-np.arange(L) / (0.003 * sr))
        y[i:i + L] += g * rng.uniform(0.6, 1.0)
    return y


def reverb(x, sr, rt60=1.5, wet=0.22, pre=0.022, seed=11):
    rng = np.random.default_rng(seed)
    L = int(rt60 * 1.2 * sr)
    t = np.arange(L) / sr
    ir = rng.normal(0, 1, (L, 2)) * np.exp(-t * 6.91 / rt60)[:, None]
    ir[: int(pre * sr)] = 0
    ir = _filt(ir, "lp", 4500, sr)
    ir /= np.sqrt(np.sum(ir ** 2, axis=0))
    w = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)
    return x + wet * w


# --- la partitura ----------------------------------------------------------------------------------------
ACORDES = {   # voces de guitarra (MIDI): bajo, quinta, octava, tercera
    "Dm": [38, 45, 50, 53], "Bbmaj7": [46, 53, 57, 62], "Gm6": [43, 50, 58, 64], "A7sus4": [45, 52, 55, 62],
    "A7b9": [45, 52, 55, 61, 70], "D": [38, 45, 50, 54], "G": [43, 50, 55, 59], "A": [45, 52, 57, 61],
    "Bm": [47, 54, 59, 62], "Em7": [40, 47, 50, 55], "Asus4": [45, 52, 57, 62],
}


def arpegio(t0, acorde, vel, sr, rng, compases=1.0, gtr_bus=None):
    """Arpegio de 6/8 (bajo, quinta, octava, tercera, octava, quinta) con un poco de mano."""
    notas = ACORDES[acorde]
    patron = [0, 1, 2, 3, 2, 1]
    vels = [0.82, 0.55, 0.62, 0.7, 0.55, 0.5]
    n = int(round(6 * compases))
    for j in range(n):
        idx = patron[j % 6]
        nota = notas[idx] + (12 if idx > 0 and j % 6 == 3 and len(notas) > 4 else 0)
        t = t0 + j * CORCHEA + rng.normal(0, 0.006)
        v = vel * vels[j % 6] * rng.uniform(0.9, 1.05)
        gtr_bus.add(t, guitarra(hz(nota), v, sr, seed=int(rng.integers(1e6))), pan=-0.15)


def rasgueo(bus, t, acorde, vel, sr, rng, abajo=True, inst="charango"):
    """Rasgueo: las cuerdas una tras otra en ~25 ms (abajo: de grave a agudo)."""
    if inst == "charango":
        notas = [n_ + 24 for n_ in ACORDES[acorde][1:]] + [ACORDES[acorde][2] + 24]
    else:
        notas = ACORDES[acorde]
    orden = notas if abajo else notas[::-1]
    for k, nota in enumerate(orden):
        f = charango if inst == "charango" else guitarra
        bus.add(t + k * 0.007 + rng.normal(0, 0.002), f(hz(nota), vel * rng.uniform(0.85, 1.0), sr,
                                                        seed=int(rng.integers(1e6))), pan=0.2 if inst == "charango" else -0.15)


def componer(dur, sr, seed=5):
    rng = np.random.default_rng(seed)
    gtr = Pista(dur, sr)       # guitarra (luego su caja de madera)
    chg = Pista(dur, sr)       # charango
    mus = Pista(dur, sr)       # todo lo demás (ya con su forma)
    sfx = Pista(dur, sr)       # ruidos de sala (secos, con poca sala)
    B = lambda k: k * BAR

    # Conocer: la aguja cuelga (re menor, muy suave) ----------------------------------------------
    arpegio(0.05, "Dm", 0.55, sr, rng, gtr_bus=gtr)
    # Vigilar: las señales (si bemol, sol menor, la con cuarta: algo se tensa) ----------------------
    arpegio(B(1), "Bbmaj7", 0.6, sr, rng, gtr_bus=gtr)
    arpegio(B(2), "Gm6", 0.62, sr, rng, compases=0.5, gtr_bus=gtr)
    arpegio(B(2) + BAR / 2, "A7sus4", 0.66, sr, rng, compases=0.5, gtr_bus=gtr)
    # el hilo se tensa (4,0 s): un acorde tenso, suave (la voz dice «nada»); el golpe de bombo llega justo
    # después de la palabra, cuando se apaga la sala (4,62)
    rasgueo(gtr, 4.0, "A7b9", 0.5, sr, rng, inst="guitarra")
    mus.add(4.62, bombo(0.9, sr, 1), 0.55)
    # las puntadas del hilván (agujita en la tela, muy bajito): estacas, aviso, cañería, torre
    for (a, b) in ((1.17, 1.8), (1.85, 2.4), (2.45, 2.98), (3.3, 3.95)):
        for k in range(int(np.ceil(a * 12)), int(b * 12) + 1):
            if k % 2 == 0:
                sfx.add(k / 12, clic(0.18, sr, 300 + k, tono=4200), 1.0, pan=0.1)
    # Alertar: se apaga la sala (4,65) y queda el bordón; la lámpara (4,92) busca --------------------
    mus.add(4.7, bordon([hz(38), hz(45)], 7.667 - 4.7 + 0.02, sr, 3, ataque=0.7), 0.5)
    sfx.add(4.917, clic(0.35, sr, 1), 1.0, pan=0.3)
    # cada torre escondida que encuentra la lámpara: una campanita, cada vez más aguda (re menor 7)
    for t, m, v in ((5.0, 69, 0.25), (5.417, 74, 0.5), (5.667, 77, 0.45), (5.917, 81, 0.45), (6.167, 84, 0.45)):
        mus.add(t, vidrio(hz(m), v, sr, seed=int(t * 100)), 0.32, pan=float(rng.uniform(-0.4, 0.4)))
    # la casa roja (6,42): el golpe más hondo, un latido
    mus.add(6.417, vidrio(hz(62), 0.7, sr, seed=7) + vidrio(hz(69), 0.4, sr, seed=8), 0.32)
    mus.add(6.417, bombo(0.6, sr, 4), 0.45)
    # la línea completa (6,75–7,58): un acorde que crece sobre el re (si bemol / re) y se corta con el clic
    sw = acorde_cuerdas([hz(50), hz(53), hz(58), hz(62), hz(65)], 7.667 - 6.7, sr, ataque=0.9, seed=9)
    sw[-int(0.02 * sr):] *= np.linspace(1, 0, int(0.02 * sr))[:, None]
    mus.add(6.7, sw, 0.55)
    # clic: se apaga la lámpara (7,67); penumbra; el tubo hace tic-tic y prende (7,83), titila (7,92)
    sfx.add(7.667, clic(0.8, sr, 2), 1.0, pan=0.25)
    sfx.add(7.75, tubo(1.4, sr, 5), 0.5, pan=-0.2)
    # Responder: la lana (8,0) en re mayor; una campanada de charango por cada ventana que se enciende ----
    arpegio(B(6), "D", 0.6, sr, rng, gtr_bus=gtr)
    for t, m, v in ((8.0, 74, 0.35), (8.833, 78, 0.5), (9.0, 81, 0.5), (9.083, 86, 0.5)):
        chg.add(t, charango(hz(m), v, sr, seed=int(t * 100)), 1.0, pan=float(rng.uniform(-0.3, 0.5)))
    # el tirón (9,42): si menor; bombo en el corte y en los tirones; el hilo se descose
    arpegio(B(7), "Bm", 0.62, sr, rng, compases=0.5, gtr_bus=gtr)
    arpegio(B(7) + BAR / 2, "Em7", 0.6, sr, rng, compases=0.5, gtr_bus=gtr)
    for t, v in ((9.417, 0.65), (9.75, 0.4), (10.083, 0.42), (10.25, 0.55)):
        mus.add(t, bombo(v, sr, int(t * 10)), 0.5)
    sfx.add(9.62, descosido(0.55, sr, 26, seed=3), 0.35, pan=0.15)
    # toda la ruta se frunce y se suelta desde la punta (10,33–10,75): un rasgón largo que cruza
    rr = descosido(0.45, sr, 30, acel=0.8, seed=4)
    pan = np.linspace(-0.6, 0.6, len(rr))
    sfx.add(10.33, np.stack([rr * np.cos((pan + 1) * np.pi / 4), rr * np.sin((pan + 1) * np.pi / 4)], 1) * 1.4, 0.3)
    rasgueo(gtr, B(8), "A", 0.55, sr, rng, inst="guitarra")
    # el revés vacío (11,0–11,75): un acorde de vidrio y puntitos de luz
    sfx.add(10.917, clic(0.25, sr, 6), 1.0, pan=0.3)
    sfx.add(11.0, clic(0.3, sr, 7), 1.0, pan=0.3)
    for m in (74, 76, 78, 81):
        mus.add(11.0, vidrio(hz(m), 0.35, sr, dur=1.2, seed=m), 0.22, pan=float(rng.uniform(-0.5, 0.5)))
    for k, (t, m) in enumerate(((11.12, 93), (11.26, 90), (11.37, 97), (11.5, 88), (11.6, 95))):
        mus.add(t, vidrio(hz(m), 0.18, sr, dur=0.8, seed=50 + k), 0.2, pan=float(rng.uniform(-0.7, 0.7)))
    sfx.add(11.833, clic(0.3, sr, 8), 1.0, pan=0.3)
    # Red: vuelve el día y la cámara se aleja (12,0): re mayor con charango rasgueado y bombo ---------
    rasgueo(chg, 11.833, "D", 0.5, sr, rng, abajo=False)
    for k, ac in ((9, "D"), (10, "G"), (11, "Asus4")):
        arpegio(B(k), ac, 0.78, sr, rng, gtr_bus=gtr)
        for j, (dt, abajo, v) in enumerate(((0, True, 0.6), (3 * CORCHEA, True, 0.45), (5 * CORCHEA, False, 0.35))):
            rasgueo(chg, B(k) + dt, ac, v, sr, rng, abajo=abajo)
        mus.add(B(k), bombo(0.7, sr, 20 + k), 0.45)
        mus.add(B(k) + 3 * CORCHEA, aro(0.7, sr, 30 + k), 0.5)
        mus.add(B(k) + 5 * CORCHEA, bombo(0.4, sr, 40 + k), 0.4)
    # el rojo baja a las vecinas y enciende sus ventanas: dos campanadas
    for t, m in ((14.28, 81), (14.5, 86)):
        chg.add(t, charango(hz(m), 0.5, sr, seed=int(t * 100)), 1.0, pan=0.4)
    # la aguja borda la «a» (14,87–15,92): sus puntadas, muy bajito
    for k in range(int(np.ceil(14.87 * 12)), int(15.92 * 12) + 1):
        if k % 2 == 0:
            sfx.add(k / 12, clic(0.16, sr, 600 + k, tono=4500), 1.0, pan=-0.1)
    # final (16,0): re mayor que suena hasta el último cuadro
    rasgueo(gtr, B(12), "D", 0.85, sr, rng, inst="guitarra")
    rasgueo(chg, B(12) + 0.01, "D", 0.6, sr, rng)
    mus.add(B(12), bombo(0.6, sr, 99), 0.45)
    mus.add(B(12), vidrio(hz(86), 0.4, sr, seed=86), 0.25)

    n = int(np.ceil(dur * sr))
    x = (cuerpo_guitarra(gtr.x, sr) * 0.9 + cuerpo_charango(chg.x, sr) * 0.55 + mus.x)[:n]
    x = reverb(_filt(x, "hp", 40, sr), sr, rt60=1.6, wet=0.24)        # (sin retumbo bajo 40 Hz)
    # silencio de verdad en la penumbra del clic (7,67–7,98): solo el clic y el tubo
    t = np.arange(n) / sr
    hush = np.interp(t, [7.66, 7.68, 7.98, 8.02], [1, 0.0, 0.0, 1.0])
    x = x * hush[:, None] + reverb(sfx.x[:n], sr, rt60=0.5, wet=0.12)
    # cola: el último cuadro llega con el acorde aún vivo; 0,25 s de salida
    x[-int(0.25 * sr):] *= np.linspace(1, 0, int(0.25 * sr))[:, None] ** 1.5
    return x / (np.abs(x).max() + 1e-9) * 0.5
