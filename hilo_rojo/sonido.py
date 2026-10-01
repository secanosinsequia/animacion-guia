"""Sonido de «Hilván»: voz en off (mujer, español latinoamericano), música original y mezcla con el video.

La voz se sintetiza con Piper (voz es_MX-claude-high), frase por frase, dentro de la ventana que le da el
guion (`guion.LINEAS`). Como la síntesis tiene algo de azar, se hacen varias tomas por frase y se elige la
mejor con medidas objetivas (duración, entonación, nitidez); las tomas elegidas se guardan en
hilo_rojo/audio/ y se reutilizan, así la mezcla es reproducible sin el modelo.

Uso:
    python -m hilo_rojo.sonido --voz ruta/es_MX-claude-high.onnx      # rehace la voz (si no está guardada)
    python -m hilo_rojo.sonido                                          # mezcla con la voz ya guardada
"""
import argparse
import os
import subprocess
import wave

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

from satc_intro.render import ffmpeg_bin
from .guion import LINEAS, srt, vtt, tabla_md
from .scene import DURATION, UFPS, FPS

SR = 48000
N_UNIQUE = int(round(DURATION * UFPS))
DUR = N_UNIQUE * (FPS // UFPS) / FPS             # duración exacta del MP4 (406 cuadros a 24 fps)
AQUI = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(AQUI, "audio")


# --- utilidades -----------------------------------------------------------------------------------------
def leer_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch) if ch > 1 else x, sr


def escribir_wav(path, x, sr=SR):
    x = np.asarray(x, np.float32)
    ch = 1 if x.ndim == 1 else x.shape[1]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def biquad(kind, f0, sr=SR, q=0.707, gain_db=0.0):
    """Coeficientes RBJ (b, a) para filtros de dos polos."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / (2 * q)
    cw = np.cos(w)
    if kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == "bp":
        b = [al, 0, -al]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == "peak":
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "highshelf":
        sa = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    a = np.asarray(a, np.float64)
    return np.asarray(b, np.float64) / a[0], a / a[0]


def filtrar(x, kind, f0, sr=SR, q=0.707, gain_db=0.0):
    b, a = biquad(kind, f0, sr, q, gain_db)
    return signal.lfilter(b, a, x, axis=0)


def loudness(x, sr=SR):
    """Sonoridad integrada aproximada (BS.1770: ponderación K y compuerta absoluta y relativa), en LUFS."""
    x = np.atleast_2d(np.asarray(x, np.float64).T).T if np.asarray(x).ndim == 1 else np.asarray(x, np.float64)
    y = filtrar(filtrar(x, "highshelf", 1681.97, sr, 0.7072, 3.9998), "hp", 38.135, sr, 0.5003)
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    zs = []
    for i in range(0, len(y) - blk, hop):
        zs.append(float(np.sum(np.mean(y[i:i + blk] ** 2, axis=0))))
    zs = np.array(zs) + 1e-12
    lk = -0.691 + 10 * np.log10(zs)
    g = zs[lk > -70]
    if not len(g):
        return -70.0
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = zs[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(g2.mean()))


def limitador(x, techo_db=-1.0, sr=SR, look=0.003, rel=0.08):
    """Limitador de picos con anticipación: nada pasa del techo (en muestras; con sobremuestreo x4 para los
    picos entre muestras)."""
    techo = 10 ** (techo_db / 20)
    up = signal.resample_poly(x, 4, 1, axis=0)
    pk = np.abs(up).reshape(len(x), 4, -1).max(axis=(1, 2)) if x.ndim > 1 else np.abs(up).reshape(len(x), 4).max(axis=1)
    g = np.minimum(1.0, techo / np.maximum(pk, 1e-9))
    g = minimum_filter1d(g, size=2 * int(look * sr) + 1)                  # anticipa la bajada
    a = np.exp(-1 / (rel * sr))
    gs = signal.lfilter([1 - a], [1, -a], g)                              # sube despacio
    gs = np.minimum(gs, g)
    return x * (gs[:, None] if x.ndim > 1 else gs)


# --- voz -----------------------------------------------------------------------------------------------
def _recortar(a, sr, umbral=0.012, margen=0.03):
    env = np.abs(a) > umbral
    if not env.any():
        return a
    i0 = max(0, int(np.argmax(env)) - int(margen * sr))
    i1 = min(len(a), len(a) - int(np.argmax(env[::-1])) + int(margen * sr))
    return a[i0:i1]


def _f0(x, sr):
    """Altura por autocorrelación en ventanas de 40 ms (Hz; 0 en lo sordo)."""
    w, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 400), int(sr / 90)
    out = []
    for i in range(0, len(x) - w, hop):
        seg = x[i:i + w] - x[i:i + w].mean()
        e = float(np.dot(seg, seg))
        if e < w * 2e-4:
            out.append(0.0)
            continue
        ac = signal.fftconvolve(seg, seg[::-1])[w - 1:]
        k = lo + int(np.argmax(ac[lo:hi]))
        out.append(sr / k if ac[k] > 0.35 * ac[0] else 0.0)
    return np.array(out)


def _puntaje(a, sr, objetivo):
    """Menor es mejor: duración cerca de la ventana, entonación viva pero sin saltos raros, voz nítida."""
    dur = len(a) / sr
    f0 = _f0(a, sr)
    v = f0[f0 > 0]
    if len(v) < 5:
        return 1e9
    st = 12 * np.log2(v / np.median(v))
    rango = float(np.percentile(st, 95) - np.percentile(st, 5))
    saltos = float(np.mean(np.abs(np.diff(st)) > 3.5))                  # quiebres de altura (artefactos)
    f, t, S = signal.stft(a, sr, nperseg=1024)
    P = np.abs(S) ** 2 + 1e-12
    plano = np.exp(np.mean(np.log(P), axis=0)) / np.mean(P, axis=0)   # planitud espectral (soplido)
    e = P.sum(axis=0)
    plano_v = float(np.median(plano[e > np.percentile(e, 40)]))
    return (abs(dur - objetivo) * 3 + max(0.0, 4.0 - rango) * 0.25 + max(0.0, rango - 9.0) * 0.25
            + saltos * 6 + plano_v * 8)


def _cadena_voz(a, sr_in):
    """De 22,05 kHz a 48 kHz y una cadena de locución: paso alto, cuerpo, presencia y compresión suave."""
    x = signal.resample_poly(a.astype(np.float64), 320, 147) if sr_in == 22050 else \
        signal.resample_poly(a.astype(np.float64), SR, sr_in)
    x = filtrar(x, "hp", 85, q=0.71)
    x = filtrar(x, "peak", 210, q=0.9, gain_db=1.5)        # cuerpo
    x = filtrar(x, "peak", 3400, q=1.0, gain_db=2.0)       # presencia
    x = filtrar(x, "highshelf", 9000, gain_db=-1.5)        # sin aspereza
    # compresión RMS (2,5:1 sobre -24 dBFS)
    w = int(0.012 * SR)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="same") + 1e-12)
    lv = 20 * np.log10(rms)
    gr = np.where(lv > -24, (lv + 24) * (1 - 1 / 2.5), 0.0)
    a_, r_ = np.exp(-1 / (0.005 * SR)), np.exp(-1 / (0.12 * SR))
    g = np.zeros_like(gr)
    acc = 0.0
    for i in range(len(gr)):                              # (frases cortas: el lazo explícito alcanza)
        c = a_ if gr[i] > acc else r_
        acc = c * acc + (1 - c) * gr[i]
        g[i] = acc
    return x * 10 ** (-g / 20)


def sintetizar(modelo, tomas=6, semilla=7, solo=None):
    """Sintetiza cada frase del guion en su ventana; elige la mejor de `tomas`. solo: números de frase
    (desde 1) para rehacer solo esas. Devuelve [(inicio, audio, sr)]."""
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    voz = PiperVoice.load(modelo)
    sr = voz.config.sample_rate
    out = []
    np.random.seed(semilla)
    for i, (a, b, txt, _, _) in enumerate(LINEAS):
        if solo and (i + 1) not in solo:
            continue
        nat = np.mean([len(_recortar(np.concatenate([c.audio_float_array for c in voz.synthesize(
            txt, syn_config=SynthesisConfig(length_scale=1.0))]), sr)) / sr for _ in range(2)])
        ls = float(np.clip((b - a) / nat, 0.84, 1.02))
        cands = []
        for k in range(tomas):
            au = np.concatenate([c.audio_float_array for c in voz.synthesize(
                txt, syn_config=SynthesisConfig(length_scale=ls))])
            au = _recortar(au, sr)
            cands.append((_puntaje(au, sr, b - a), k, au))
        cands.sort(key=lambda c: c[0])
        best = cands[0][2]
        print(f"  frase {i + 1}: velocidad x{1 / ls:.2f}, {len(best) / sr:.2f} s (ventana {b - a:.2f} s), "
              f"puntajes {[round(c[0], 2) for c in cands]}", flush=True)
        escribir_wav(os.path.join(AUDIO, f"voz_{i + 1:02d}.wav"), best / max(1e-6, np.abs(best).max()) * 0.9, sr)
        out.append((a, best, sr))
    return out


def pista_voz():
    """La voz ubicada en el tiempo (estéreo 48 kHz) y los tiempos reales de cada frase."""
    n = int(np.ceil(DUR * SR))
    pista = np.zeros(n)
    reales = []
    for i, (a, b, txt, _, _) in enumerate(LINEAS):
        au, sr = leer_wav(os.path.join(AUDIO, f"voz_{i + 1:02d}.wav"))
        if au.ndim > 1:                                   # (una grabación propia puede venir en estéreo)
            au = au.mean(axis=1)
        au = _recortar(au, sr, umbral=0.04 * float(np.abs(au).max()))   # empieza donde empieza la voz
        if len(au) / sr > (b - a) + 0.25:
            print(f"  ojo: la frase {i + 1} dura {len(au) / sr:.2f} s y su ventana es de {b - a:.2f} s")
        x = _cadena_voz(au, sr)
        x = x / (np.sqrt(np.mean(x[np.abs(x) > 0.02] ** 2)) + 1e-9) * 0.12      # mismo nivel en todas
        i0 = int(a * SR)
        i1 = min(n, i0 + len(x))
        pista[i0:i1] += x[:i1 - i0]
        # tiempos del texto (sin el margen de silencio de los extremos)
        env = np.abs(x) > 0.02
        t0 = a + np.argmax(env) / SR
        t1 = a + (len(x) - np.argmax(env[::-1])) / SR
        reales.append((round(t0, 2), round(min(t1, DUR), 2), txt))
    # una sala chica y cálida, apenas (la voz queda adelante)
    rng = np.random.default_rng(3)
    L = int(0.45 * SR)
    t = np.arange(L) / SR
    ir = rng.normal(0, 1, (L, 2)) * np.exp(-t * 6.91 / 0.38)[:, None]
    ir[: int(0.012 * SR)] = 0
    ir = filtrar(ir, "lp", 5200)
    ir /= np.sqrt(np.sum(ir ** 2, axis=0))
    wet = np.stack([signal.fftconvolve(pista, ir[:, c])[:n] for c in range(2)], 1)
    est = np.stack([pista, pista], 1) + 0.16 * wet
    return est, reales


# --- mezcla --------------------------------------------------------------------------------------------
def mezclar(voz, musica, reales, bajo_voz_db=-12.0, respiro_db=4.0):
    """La música queda `bajo_voz_db` debajo de la voz en cada frase (medido frase por frase, sin aplastar
    su dinámica) y sube `respiro_db` en los respiros entre frases; transiciones de 150 ms."""
    n = min(len(voz), len(musica))
    voz, musica = voz[:n], musica[:n]
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    pts_t, pts_g = [], []
    gs = []
    for (t0, t1, _) in reales:
        i0, i1 = int(t0 * SR), int(t1 * SR)
        g = 20 * np.log10(rms(voz[i0:i1]) / rms(musica[i0:i1])) + bajo_voz_db
        gs.append((t0, t1, float(np.clip(g, -30, 18))))
    rampa = 0.15
    for k, (t0, t1, g) in enumerate(gs):
        g_prev = gs[k - 1][2] if k else g
        libre = min(g, g_prev) + respiro_db                     # el respiro antes de esta frase
        ini = gs[k - 1][1] if k else 0.0
        if t0 - ini > 2 * rampa:
            pts_t += [ini + rampa * 0.5, t0 - rampa]
            pts_g += [libre, libre]
        pts_t += [t0 - 0.02, t1]
        pts_g += [g, g]
    pts_t += [gs[-1][1] + rampa, n / SR]
    pts_g += [gs[-1][2] + respiro_db, gs[-1][2] + respiro_db]
    curva = np.interp(np.arange(n) / SR, pts_t, pts_g)
    m = musica * (10 ** (curva / 20))[:, None]
    mix = voz + m
    # sonoridad de entrega web: -16 LUFS integrados, picos bajo -1 dBTP
    g = 10 ** ((-16.0 - loudness(mix)) / 20)
    mix = limitador(mix * g, -1.0)
    return mix, voz * g, m * g


def ffmpeg(*args):
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voz", help="modelo Piper (.onnx) para rehacer la voz")
    ap.add_argument("--tomas", type=int, default=6)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    if args.voz and (not os.path.exists(os.path.join(AUDIO, "voz_01.wav")) or os.environ.get("REHACER_VOZ")):
        print("voz: sintetizando con", os.path.basename(args.voz))
        sintetizar(args.voz, args.tomas)
    voz, reales = pista_voz()
    from .musica import componer
    musica = componer(DUR, SR)
    mix, voz_n, mus_n = mezclar(voz, musica, reales)
    aud = os.path.join(args.out, "audio")
    os.makedirs(aud, exist_ok=True)
    escribir_wav(os.path.join(aud, "hilvan_mezcla.wav"), mix)
    escribir_wav(os.path.join(aud, "hilvan_voz.wav"), voz_n)
    escribir_wav(os.path.join(aud, "hilvan_musica.wav"), mus_n)
    ffmpeg("-i", os.path.join(aud, "hilvan_mezcla.wav"), "-c:a", "aac", "-b:a", "192k", os.path.join(aud, "hilvan_mezcla.m4a"))
    print(f"mezcla: {loudness(mix):.1f} LUFS, pico {20 * np.log10(np.abs(mix).max()):.1f} dBFS")
    with open(os.path.join(args.out, "hilvan_voz.srt"), "w", encoding="utf-8") as f:
        f.write(srt(reales))
    with open(os.path.join(args.out, "hilvan_voz.vtt"), "w", encoding="utf-8") as f:
        f.write(vtt(reales))
    with open(os.path.join(AUDIO, "tiempos.md"), "w", encoding="utf-8") as f:
        f.write(tabla_md(reales) + "\n")
    for tam in ("1920x1080", "1080x1920"):
        src = os.path.join(args.out, f"hilvan_{tam}.mp4")
        if os.path.exists(src):
            dst = os.path.join(args.out, f"hilvan_{tam}_voz.mp4")
            ffmpeg("-i", src, "-i", os.path.join(aud, "hilvan_mezcla.wav"), "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-shortest",
                   "-movflags", "+faststart", dst)
            print("MP4 con voz y música:", dst, f"{os.path.getsize(dst) / 1e6:.1f} MB")
    for t0, t1, txt in reales:
        print(f"  {t0:5.2f}–{t1:5.2f}  {txt}")


if __name__ == "__main__":
    main()
