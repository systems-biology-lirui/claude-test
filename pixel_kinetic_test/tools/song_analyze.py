"""Analyse a song for re-timing: beats, downbeat phase, sections, plus a diagnostic image."""
import sys, json
import numpy as np, librosa
from scipy import signal
from PIL import Image, ImageDraw, ImageFont

path, out_png, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
y, sr = librosa.load(path, sr=22050, mono=True)
dur = len(y) / sr
hop = 256
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, units='time', tightness=100)
period = float(np.median(np.diff(beats)))
# refine beats onto a straight grid (AI tracks are machine-tight): least-squares fit
idx = np.arange(len(beats))
A = np.vstack([idx, np.ones_like(idx)]).T
p, t0 = np.linalg.lstsq(A, beats, rcond=None)[0]
grid = t0 + p * idx
print(f'grid: t0={t0:.4f}s period={p:.5f}s ({60/p:.3f} BPM); max deviation {np.abs(beats-grid).max()*1000:.1f} ms')
# band energies at each beat
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
def band(lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return S[m].sum(0)
low = band(30, 150); mid = band(150, 2000); high = band(4000, 11000)
vocal = band(250, 3500)
def at(arr, t, w=0.06):
    m = (times >= t - 0.01) & (times < t + w)
    return float(arr[m].mean()) if m.any() else 0.0
# low-band onset (kick) strength per beat, choose downbeat phase by chroma change + kick
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
def chroma_at(t, w):
    m = (times >= t) & (times < t + w)
    return chroma[:, m].mean(1) if m.any() else np.zeros(12)
kick = np.array([at(low, b) for b in grid])
chg = np.array([np.linalg.norm(chroma_at(b, p) - chroma_at(b - p, p)) for b in grid])
scores = []
for ph in range(4):
    sel = idx % 4 == ph
    scores.append((kick[sel].mean() / kick.mean()) + (chg[sel].mean() / chg.mean()))
ph = int(np.argmax(scores))
print('downbeat phase scores', np.round(scores, 3), '-> phase', ph)
downbeats = grid[idx % 4 == ph]
# also allow a pickup: bars before the first detected downbeat
first = downbeats[0]
while first - 4 * p > -1e-3:
    first -= 4 * p
bars = np.arange(first, dur, 4 * p)
print('bar starts:', ' '.join(f'{b:.3f}' for b in bars))
# section novelty (self-similarity on chroma+mfcc, per bar)
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
feat = np.vstack([chroma, (mfcc - mfcc.mean(1, keepdims=True)) / (mfcc.std(1, keepdims=True) + 1e-6)])
def feat_bar(a, b):
    m = (times >= a) & (times < b)
    return feat[:, m].mean(1)
bf = np.array([feat_bar(b, b + 4 * p) for b in bars])
nov = [0.0] + [float(np.linalg.norm(bf[i] - bf[i - 1])) for i in range(1, len(bf))]
loud = [20 * np.log10(np.sqrt(np.mean(y[int(max(0, b) * sr):int(min(dur, b + 4 * p) * sr)] ** 2)) + 1e-9) for b in bars]
lowb = [float(np.mean(low[(times >= b) & (times < b + 4 * p)])) for b in bars]
highb = [float(np.mean(high[(times >= b) & (times < b + 4 * p)])) for b in bars]
vocb = [float(np.mean(vocal[(times >= b) & (times < b + 4 * p)])) for b in bars]
print(' bar  start   loud  low   high  novelty')
for i, b in enumerate(bars):
    print(f'{i+1:4d} {b:6.2f}  {loud[i]:5.1f}  {lowb[i]/max(lowb):.2f}  {highb[i]/max(highb):.2f}  {nov[i]:.2f}')
json.dump({'t0': float(t0), 'period': float(p), 'bpm': float(60 / p), 'phase': ph, 'bars': [float(b) for b in bars],
           'beats': [float(g) for g in grid], 'duration': float(dur), 'loud': [float(v) for v in loud], 'novelty': [float(v) for v in nov]}, open(out_json, 'w'), indent=1)

# ---------- diagnostic image
W, H = 1800, 620
img = Image.new('RGB', (W, H), (14, 12, 22))
d = ImageDraw.Draw(img)
Sdb = librosa.amplitude_to_db(S, ref=np.max)
fl = np.geomspace(40, 10000, 260)[::-1]
fi = np.clip(np.searchsorted(freqs, fl), 0, len(freqs) - 1)
spec = np.clip((Sdb[fi] + 70) / 70, 0, 1)
col = np.stack([spec ** 0.7 * 255, spec ** 1.6 * 210, (1 - spec) * 60 + spec * 70], -1).astype(np.uint8)
spec_img = Image.fromarray(col).resize((W, 260))
img.paste(spec_img, (0, 0))
def plot(arr, y0, h, colr, label):
    a = arr / (arr.max() + 1e-9)
    pts = [(i / len(a) * W, y0 + h - a[i] * h) for i in range(len(a))]
    d.line(pts, fill=colr, width=1)
    d.text((4, y0 + 2), label, fill=colr)
sm = lambda a, k=8: np.convolve(a, np.ones(k) / k, 'same')
plot(sm(low), 270, 70, (255, 120, 90), 'low 30-150Hz (kick/bass)')
plot(sm(vocal), 350, 70, (120, 230, 255), 'mid 250-3500Hz (vocal/lead)')
plot(sm(high), 430, 70, (200, 200, 120), 'high 4-11kHz (hats/cymbals)')
plot(sm(oenv, 4), 510, 70, (180, 255, 160), 'onset strength')
for g in grid:
    x = g / dur * W
    d.line([(x, 262), (x, 266)], fill=(150, 150, 150))
for i, b in enumerate(bars):
    x = b / dur * W
    d.line([(x, 0), (x, H)], fill=(255, 210, 63), width=1)
    d.text((x + 2, 590), f'bar{i+1}\n{b:.2f}', fill=(255, 210, 63))
img.save(out_png)
print('saved', out_png)
