import sys, os, wave
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from scipy import signal
from PIL import Image, ImageDraw
import timeline as TL
path, out = sys.argv[1], sys.argv[2]
w = wave.open(path); sr = w.getframerate(); n = w.getnframes()
x = np.frombuffer(w.readframes(n), '<i2').reshape(-1, 2).astype(np.float32) / 32768
m = x.mean(1)
f, t, S = signal.spectrogram(m, sr, nperseg=2048, noverlap=2048 - 480)
S = 10 * np.log10(S + 1e-10)
fmax_i = np.searchsorted(f, 12000)
S = S[:fmax_i][::-1]
# log-frequency remap
H = 300
fl = np.geomspace(30, 12000, H)[::-1]
idx = np.clip(np.searchsorted(f[:fmax_i], fl), 0, fmax_i - 1)
S2 = S[::-1][idx]
v = np.clip((S2 + 100) / 70, 0, 1)
img = (np.stack([v ** 0.6 * 255, v ** 1.5 * 200, (1 - v) * 80 + v * 60], -1)).astype(np.uint8)
Wd = 1600
img = np.array(Image.fromarray(img).resize((Wd, H)))
# rms envelope
hop = sr // 100
rms = np.sqrt(np.convolve(m ** 2, np.ones(hop) / hop, 'same'))[::hop]
canvas = Image.new('RGB', (Wd, H + 140), (15, 15, 20))
canvas.paste(Image.fromarray(img), (0, 0))
d = ImageDraw.Draw(canvas)
dur = len(m) / sr
pts = [(i / len(rms) * Wd, H + 130 - min(1, r * 3) * 120) for i, r in enumerate(rms)]
d.line(pts, fill=(120, 220, 255), width=1)
marks = [(L['impact'], (255, 60, 60)) for L in TL.LV] + [(L['start'], (255, 220, 60)) for L in TL.LV] + \
        [(TL.MUSIC_CUT, (255, 255, 255)), (TL.LAND_T, (120, 255, 120)), (TL.FREEZE_T, (200, 120, 255)), (TL.END_CARD, (200, 120, 255))]
for (tm, c) in marks:
    xx = tm / dur * Wd
    d.line([(xx, 0), (xx, H + 140)], fill=c, width=1)
for s in range(0, 32, 2):
    d.text((s / dur * Wd + 2, H + 2), f'{s}s', fill=(200, 200, 200))
canvas.save(out)
print('peak', np.abs(x).max(), 'saved', out)
