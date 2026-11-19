"""Original chiptune score + SFX, synthesised from scratch and locked to timeline.py.

Channels modelled on 8-bit consoles (two pulse waves, a 4-bit triangle, an LFSR noise
channel) plus a modern sine sub layer that only appears on impacts.
usage: python3 src/audio.py out.wav
"""
import os
import sys
import math
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(__file__))
import timeline as TL

SR = 48000
DUR = TL.DURATION + 0.5
N = int(DUR * SR)
rng = np.random.default_rng(1234)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
        'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def n2m(s):
    """'A4' -> 69"""
    name, octv = s[:-1], int(s[-1])
    return 12 * (octv + 1) + NOTE[name]


# ---------------------------------------------------------------- oscillators
def _phase(freq, n):
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    return np.cumsum(f / SR) % 1.0, f / SR


def _polyblep(t, dt):
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1.0
    m2 = t > 1.0 - dt
    x = (t[m2] - 1.0) / dt[m2]
    out[m2] = x * x + x + x + 1.0
    return out


def pulse(freq, dur, duty=0.5):
    n = int(dur * SR)
    if n <= 0:
        return np.zeros(0)
    ph, dt = _phase(freq, n)
    dt = np.maximum(dt, 1e-6)
    sq = np.where(ph < duty, 1.0, -1.0)
    sq += _polyblep(ph, dt)
    sq -= _polyblep((ph - duty) % 1.0, dt)
    return sq * 0.5


def tri(freq, dur, steps=16):
    n = int(dur * SR)
    ph, _ = _phase(freq, n)
    v = 4 * np.abs(ph - 0.5) - 1
    if steps:
        v = np.round((v + 1) / 2 * (steps - 1)) / (steps - 1) * 2 - 1
    return v * 0.6


def sine(freq, dur):
    n = int(dur * SR)
    ph, _ = _phase(freq, n)
    return np.sin(2 * np.pi * ph)


def noise(dur, rate=SR, seed=None):
    """Sample-and-hold white noise (rate lowers the 'pitch' like an LFSR period)."""
    n = int(dur * SR)
    r = np.random.default_rng(seed) if seed is not None else rng
    k = max(1, int(SR / rate))
    v = r.uniform(-1, 1, n // k + 2)
    return np.repeat(v, k)[:n]


def env(n, a=0.002, d=0.05, s=0.0, r=0.05, hold=None):
    """ADSR envelope over n samples. hold: seconds before release (default: fills n)."""
    e = np.zeros(n)
    A, D, R = int(a * SR), int(d * SR), int(r * SR)
    Hs = n - R if hold is None else int(hold * SR)
    i = 0
    seg = min(A, n)
    e[:seg] = np.linspace(0, 1, seg, endpoint=False) if seg else e[:seg]
    i = seg
    seg = max(0, min(D, Hs - i))
    if seg:
        e[i:i + seg] = np.linspace(1, s, seg, endpoint=False)
    i += seg
    if Hs > i:
        e[i:Hs] = s
        i = Hs
    rem = n - i
    if rem > 0:
        start = e[i - 1] if i > 0 else s
        e[i:] = np.linspace(start, 0, rem)
    return e


def expdec(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def lp(x, fc, order=2):
    sos = signal.butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc, 'high', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, min(hi, SR * 0.45)], 'band', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def sweep_bp(x, f0, f1, q=3.0, steps=48):
    """Time-varying band-pass by crossfading blocks (cheap and good enough for whooshes)."""
    n = len(x)
    out = np.zeros(n)
    bl = n // steps + 1
    for k in range(steps):
        a, b = k * bl, min(n, (k + 1) * bl + bl // 2)
        if a >= n:
            break
        f = f0 * (f1 / f0) ** (k / max(1, steps - 1))
        lo, hi = f / (1 + 1 / q), f * (1 + 1 / q)
        seg = bp(x[max(0, a - 2048):b], max(20, lo), min(hi, SR * 0.45))[-(b - a):]
        w = np.ones(b - a)
        out[a:b] += seg * w
    return out / 1.4


def bitcrush(x, bits=6, hold=3):
    q = 2 ** (bits - 1)
    y = np.round(x * q) / q
    return np.repeat(y[::hold], hold)[:len(x)]


# ---------------------------------------------------------------- mixer
class Bus:
    def __init__(self, n=N):
        self.L = np.zeros(n)
        self.R = np.zeros(n)

    def add(self, t, sig, gain=1.0, pan=0.0):
        if sig is None or len(sig) == 0:
            return
        i = int(round(t * SR))
        if i < 0:
            sig = sig[-i:]
            i = 0
        j = min(len(self.L), i + len(sig))
        if j <= i:
            return
        s = sig[:j - i] * gain
        gl = math.cos((pan + 1) * math.pi / 4)
        gr = math.sin((pan + 1) * math.pi / 4)
        self.L[i:j] += s * gl * 1.4142
        self.R[i:j] += s * gr * 1.4142


music = Bus()
sfx = Bus()
amb = Bus()


# ---------------------------------------------------------------- instruments
def inst_lead(m, dur, vol=0.22, duty=0.5, vib=True):
    f = mtof(m)
    n = int(dur * SR)
    if vib and dur > 0.18:
        tt = np.arange(n) / SR
        f = f * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * tt) * np.clip((tt - 0.1) / 0.1, 0, 1))
    return pulse(f, dur, duty) * env(n, 0.003, 0.08, 0.6, 0.04) * vol


def inst_arp(m, dur, vol=0.1, duty=0.125):
    n = int(dur * SR)
    return pulse(mtof(m), dur, duty) * env(n, 0.002, 0.05, 0.3, 0.02) * vol


def inst_bass(m, dur, vol=0.42):
    n = int(dur * SR)
    return tri(mtof(m), dur) * env(n, 0.002, 0.02, 0.9, 0.015) * vol


def kick(vol=0.75):
    d = 0.22
    n = int(d * SR)
    tt = np.arange(n) / SR
    f = 45 + 110 * np.exp(-tt / 0.025)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.08)
    click = noise(0.004, seed=1) * 0.5
    s[:len(click)] += click
    return s * vol


def snare(vol=0.42):
    d = 0.18
    n = int(d * SR)
    nz = hp(noise(d, 22000, seed=2), 900) * expdec(n, 0.05)
    tn = tri(190, d, steps=16) * expdec(n, 0.03)
    return (nz * 0.9 + tn * 0.6) * vol


def hat(vol=0.12, open_=False):
    d = 0.12 if open_ else 0.035
    n = int(d * SR)
    return hp(noise(d, SR, seed=3), 7000) * expdec(n, 0.04 if open_ else 0.01) * vol


def crash(vol=0.35, d=1.4):
    n = int(d * SR)
    return hp(noise(d, SR, seed=4), 3500) * expdec(n, 0.35) * vol


# ---------------------------------------------------------------- score
def seq_notes(bus, t0, beat, pattern, fn, slot=0.5, gain=1.0, pan=0.0, **kw):
    """pattern: list of note names / None per slot (slot in beats)."""
    for i, nm in enumerate(pattern):
        if nm is None or nm == '-':
            continue
        # sustain over following '-' slots
        L = 1
        while i + L < len(pattern) and pattern[i + L] == '-':
            L += 1
        dur = L * slot * beat * 0.92
        m = n2m(nm) if isinstance(nm, str) else nm
        bus.add(t0 + i * slot * beat, fn(m, dur, **kw), gain, pan)


def build_music():
    b120 = 0.5
    # ---------------- intro (0-4 s), A minor, calm lab theme
    # soft sustained bass + arpeggio with echo
    music.add(0.0, inst_bass(n2m('A2'), 2.0, 0.32) * np.linspace(0, 1, int(2.0 * SR)) ** 0.5)
    arp = ['A4', 'C5', 'E5', 'B4', 'A4', 'E5', 'C5', 'B4'] * 2
    for i, nm in enumerate(arp):
        t = 0.25 + i * 0.25
        if t >= 4.0:
            break
        s = inst_arp(n2m(nm), 0.2, 0.09)
        music.add(t, s, 1.0, -0.3)
        music.add(t + 0.1875, s, 0.35, 0.3)   # echo
    for i in range(8):
        music.add(2.0 + i * 0.25, inst_bass(n2m('A2') + (12 if i % 4 == 2 else 0), 0.22, 0.34))
        if i % 2 == 0:
            music.add(2.0 + i * 0.25, hat(0.06))
    music.add(2.0, kick(0.5)); music.add(3.0, kick(0.5))
    music.add(3.5, snare(0.25)); music.add(3.75, snare(0.18))

    # ---------------- levels 1-3 (4-16 s), 120 BPM
    chords = {  # bar start -> (root midi for bass, chord tones for arps)
        'Am': (n2m('A2'), ['A4', 'C5', 'E5', 'A5']),
        'F': (n2m('F2'), ['F4', 'A4', 'C5', 'F5']),
        'C': (n2m('C3'), ['C5', 'E5', 'G5', 'C6']),
        'G': (n2m('G2'), ['G4', 'B4', 'D5', 'G5']),
    }
    bars = [(4.0, 'Am'), (6.0, 'F'), (8.0, 'C'), (10.0, 'G'), (12.0, 'Am'), (14.0, 'F')]
    leads = {
        4.0: ['E5', '-', 'A5', '-', 'G5', 'E5', 'D5', '-'],
        6.0: [None, None, 'D5', 'E5', '-', 'C5', 'A4', '-'],
        8.0: ['E5', '-', 'G5', '-', 'A5', 'G5', 'E5', '-'],
        10.0: [None, None, 'B4', 'D5', '-', 'G5', 'F5', 'D5'],
        12.0: ['A5', '-', 'C6', '-', 'B5', 'A5', 'G5', 'E5'],
        14.0: [None, None, 'F5', 'G5', '-', 'A5', 'C6', '-'],
    }
    for (t0, ch) in bars:
        root, tones = chords[ch]
        lvl = 0 if t0 < 8 else 1 if t0 < 12 else 2
        slow = (t0 == 12.0)   # bar containing L3 slow-mo (13.5-14.0)
        for i in range(8):     # bass 8ths: R R 8va R | R R 8va 5th
            t = t0 + i * 0.25
            if slow and t >= 13.5:
                break
            m = root + [0, 0, 12, 0, 0, 0, 12, 7][i]
            music.add(t, inst_bass(m, 0.22, 0.4))
        # drums
        for beat in range(4):
            t = t0 + beat * b120
            if slow and t >= 13.5:
                break
            if beat in (0, 2):
                music.add(t, kick(0.7 if beat == 0 else 0.6))
            if beat in (1, 3):
                music.add(t, snare(0.38))
            sub = 2 if lvl == 0 else 4
            for k in range(sub):
                tk = t + k * b120 / sub
                if slow and tk >= 13.5:
                    break
                music.add(tk, hat(0.09 if k % 2 == 0 else 0.06), 1.0, 0.25)
        # arps (16ths) from level 2 on
        if lvl >= 1:
            for i in range(16):
                t = t0 + i * 0.125
                if slow and t >= 13.5:
                    break
                nm = tones[[0, 1, 2, 3, 2, 1, 0, 1][i % 8]]
                music.add(t, inst_arp(n2m(nm), 0.1, 0.07 if lvl == 1 else 0.085), 1.0, -0.35)
        # lead
        pat = leads[t0]
        for i, nm in enumerate(pat):
            if nm in (None, '-'):
                continue
            t = t0 + i * 0.25
            if slow and t >= 13.5:
                continue
            L = 1
            while i + L < len(pat) and pat[i + L] == '-':
                L += 1
            dur = L * 0.25 * 0.92
            music.add(t, inst_lead(n2m(nm), dur, 0.17 + 0.03 * lvl), 1.0, 0.1)
            if lvl == 2:   # harmony a third below on level 3
                music.add(t, inst_lead(n2m(nm) - 4, dur, 0.08, duty=0.25), 1.0, -0.2)
    # crashes on impact downbeats and level starts
    for t in (4.0, 8.0, 12.0):
        music.add(t, crash(0.18, 1.0), 1.0, 0.3)

    # ---------------- speed-up: levels 4 (16-19.2 s), 150 BPM, B minor
    b150 = 0.4
    up = 2   # whole step up
    bars2 = [(16.0, 'Am'), (17.6, 'F')]
    leads2 = {16.0: ['E5', 'E5', 'A5', 'E5', 'G5', 'E5', 'D5', 'C5'],
              17.6: [None, None, 'D5', 'E5', 'F5', 'E5', 'C5', 'A4']}
    for (t0, ch) in bars2:
        root, tones = chords[ch]
        for i in range(16):
            music.add(t0 + i * 0.1, inst_bass(root + up + [0, 0, 12, 0][i % 4], 0.09, 0.42))
        for beat in range(4):
            t = t0 + beat * b150
            music.add(t, kick(0.72))
            if beat in (1, 3):
                music.add(t, snare(0.4))
            for k in range(4):
                music.add(t + k * 0.1, hat(0.08 if k % 2 == 0 else 0.055), 1.0, 0.25)
        for i in range(16):
            nm = tones[[0, 1, 2, 3][i % 4]]
            music.add(t0 + i * 0.1, inst_arp(n2m(nm) + up + 12, 0.08, 0.06), 1.0, -0.35)
        pat = leads2[t0]
        for i, nm in enumerate(pat):
            if nm in (None, '-'):
                continue
            music.add(t0 + i * 0.2, inst_lead(n2m(nm) + up, 0.18, 0.2), 1.0, 0.1)
    music.add(16.0, crash(0.22, 1.2), 1.0, 0.3)

    # ---------------- level 5 warning (19.2-20.6): siren + drone + 4-on-the-floor
    t = 19.2
    k = 0
    while t < 20.6:
        f = 880 if k % 2 == 0 else 660
        s = pulse(f, 0.19, 0.5) * env(int(0.19 * SR), 0.005, 0.02, 0.8, 0.02) * 0.11
        music.add(t, s, 1.0, -0.2 if k % 2 else 0.2)
        t += 0.2
        k += 1
    music.add(19.2, inst_bass(n2m('B1'), 1.6, 0.45))
    for beat in range(4):
        music.add(19.2 + beat * b150, kick(0.7))
    # ---------------- final build (20.6-22.4): dominant pedal, snare roll accelerating, rising arp
    t0 = 20.6
    for i in range(14):           # 16ths of F# until the slow-mo
        t = t0 + i * 0.1
        if t >= 22.0:
            break
        music.add(t, inst_bass(n2m('F#2') + (12 if i % 4 == 2 else 0), 0.09, 0.42))
    roll = []
    t = 20.8
    while t < 22.0:
        roll.append(t)
        prog = (t - 20.8) / 1.2
        t += 0.2 if prog < 0.34 else 0.1 if prog < 0.67 else 0.05
    for i, tr in enumerate(roll):
        music.add(tr, snare(0.18 + 0.22 * i / len(roll)))
    scale = ['B4', 'C#5', 'D5', 'E5', 'F#5', 'G5', 'A5', 'B5', 'C#6', 'D6', 'E6', 'F#6']
    for i in range(12):
        t = 20.8 + i * 0.1
        music.add(t, inst_lead(n2m(scale[i]), 0.09, 0.16, duty=0.25), 1.0, 0.0)
    music.add(20.8, kick(0.7)); music.add(21.2, kick(0.7)); music.add(21.6, kick(0.7))
    # slow-mo (22.0-22.4): reverse-swell into the impact
    d = 0.4
    n = int(d * SR)
    sw = hp(noise(d, SR, seed=8), 1500) * np.linspace(0, 1, n) ** 2.5 * 0.35
    music.add(22.0, sw)
    music.add(22.0, inst_bass(n2m('F#1'), 0.4, 0.4) * np.linspace(1, 0.4, n))

    # ---------------- end jingle (after the third slam)
    jt = 26.55
    jn = ['G4', 'C5', 'E5', 'G5']
    for i, nm in enumerate(jn):
        music.add(jt + i * 0.09, inst_lead(n2m(nm), 0.085, 0.2), 1.0, 0.0)
    music.add(jt + 0.36, inst_lead(n2m('C6'), 0.75, 0.2), 1.0, 0.0)
    music.add(jt + 0.36, inst_lead(n2m('E5'), 0.75, 0.09, duty=0.25), 1.0, -0.3)
    music.add(jt + 0.36, inst_lead(n2m('G5'), 0.75, 0.09, duty=0.25), 1.0, 0.3)
    music.add(jt + 0.36, inst_bass(n2m('C3'), 0.8, 0.4))
    # quiet outro: the lab arpeggio returns, fading
    arp2 = ['C5', 'E5', 'G5', 'D5', 'C5', 'G5', 'E5', 'D5']
    for i in range(12):
        t = 27.4 + i * 0.25
        g = 0.07 * (1 - i / 12)
        s = inst_arp(n2m(arp2[i % 8]), 0.2, 1.0) * g
        music.add(t, s, 1.0, -0.3)
        music.add(t + 0.1875, s, 0.35, 0.3)


# ---------------------------------------------------------------- SFX
def blip(f, d=0.03, vol=0.1, duty=0.25):
    n = int(d * SR)
    return pulse(f, d, duty) * env(n, 0.001, d * 0.6, 0.0, 0.005) * vol


def tick_train(bus, t0, t1, rate, f=1800, vol=0.035, pan=0.0, jitter=True):
    t = t0
    k = 0
    while t < t1:
        ff = f * (1 + (0.12 * ((k * 7) % 5) / 5 if jitter else 0))
        bus.add(t, blip(ff, 0.018, vol), 1.0, pan)
        t += 1.0 / rate
        k += 1


def whoosh(d, f0, f1, vol=0.3, seed=5):
    n = int(d * SR)
    x = noise(d, SR, seed=seed)
    y = sweep_bp(x, f0, f1, q=2.5, steps=32)
    e = np.sin(np.linspace(0, np.pi, n)) ** 1.5
    return y * e * vol


def thump(f0=120, f1=40, d=0.3, vol=0.8, tau=0.09):
    n = int(d * SR)
    tt = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-tt / 0.04)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / tau) * vol


def glass(d=0.5, vol=0.4, seed=11):
    n = int(d * SR)
    out = np.zeros(n)
    r = np.random.default_rng(seed)
    for _ in range(60):
        st = int(r.uniform(0, 0.6) ** 2 * n)
        f = r.uniform(2500, 9000)
        L = int(r.uniform(0.01, 0.06) * SR)
        L = min(L, n - st)
        if L <= 0:
            continue
        tt = np.arange(L) / SR
        out[st:st + L] += np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.012) * r.uniform(0.2, 1.0)
    out += hp(noise(d, SR, seed=seed), 4000) * expdec(n, 0.08) * 0.8
    return out * vol


def bell_ring(vol=0.18):
    """Bicycle bell: two inharmonic partials, rung three times."""
    d = 0.75
    n = int(d * SR)
    out = np.zeros(n)
    for k in range(3):
        st = int(k * 0.13 * SR)
        L = n - st
        tt = np.arange(L) / SR
        s = (np.sin(2 * np.pi * 2350 * tt) + 0.6 * np.sin(2 * np.pi * 3110 * tt) + 0.3 * np.sin(2 * np.pi * 4700 * tt))
        out[st:] += s * np.exp(-tt / 0.18) * (1.0 if k < 2 else 0.8)
    return out * vol / 2


def twinkle(vol=0.22):
    d = 1.2
    n = int(d * SR)
    tt = np.arange(n) / SR
    s = np.zeros(n)
    for k, f in enumerate([1568, 2093, 2637, 3136]):
        st = int(k * 0.045 * SR)
        t2 = tt[:n - st]
        s[st:] += np.sin(2 * np.pi * f * t2) * np.exp(-t2 / 0.35) * (1 - k * 0.15)
    return s * vol


def build_sfx():
    import physics
    from director import Director, CONTACT, SPEED
    import sprites_ped
    D = Director()
    LV = TL.LV
    # ---- intro
    sfx.add(0.05, pulse(np.linspace(220, 1760, int(0.3 * SR)), 0.3, 0.25) * env(int(0.3 * SR), 0.01, 0.2, 0.5, 0.08) * 0.08)
    sfx.add(0.38, blip(1760, 0.12, 0.08, 0.5))
    # title letters drop in (5 + 6 chars)
    for i in range(5):
        sfx.add(0.10 + i * 0.05, blip(660 + i * 70, 0.03, 0.05, 0.5), 1.0, -0.4 + i * 0.2)
    for i in range(6):
        sfx.add(0.35 + i * 0.05, blip(990 + i * 80, 0.03, 0.045, 0.5), 1.0, -0.4 + i * 0.16)
    sfx.add(0.9, whoosh(1.4, 200, 1400, 0.06, seed=31))           # camera tilt air
    # lock-on
    for k in range(4):
        sfx.add(2.0 + k * 0.08, blip(1200 + k * 200, 0.04, 0.06, 0.5), 1.0, 0.2)
    sfx.add(2.36, blip(2400, 0.08, 0.08, 0.5)); sfx.add(2.45, blip(3200, 0.12, 0.07, 0.5))
    tick_train(sfx, 2.6, 3.5, 22, 1500, 0.025, -0.4)
    # ---- per level HUD
    for i, L in enumerate(LV):
        s = TL.CARD_SPEED[i]
        if i < 4:
            st = L['start']
            # banner slam
            sfx.add(st, thump(160, 60, 0.25, 0.5, 0.07), 1.0, 0.0)
            sfx.add(st, hp(noise(0.12, SR, seed=40 + i), 2000) * expdec(int(0.12 * SR), 0.03) * 0.25)
            base = [72, 74, 76, 78][i]
            for k, iv in enumerate([0, 4, 7, 12]):
                sfx.add(st + 0.05 + k * 0.055, blip(mtof(base + iv), 0.06, 0.07, 0.5), 1.0, 0.0)
        card = TL.card_start(i)
        # typing ticks (name, rows, calc)
        tick_train(sfx, card + 0.2 / s, card + 0.45 / s, 16, 1300, 0.03, 0.3)
        tick_train(sfx, card + 0.55 / s, card + 0.7 / s, 18, 1700, 0.025, 0.3)
        tick_train(sfx, card + 0.75 / s, card + 0.9 / s, 18, 1700, 0.025, 0.3)
        tick_train(sfx, card + 0.95 / s, card + 1.25 / s, 22, 2100, 0.025, 0.3)
        # KE count-up: rising ticks
        c0 = TL.count_start(i)
        for k in range(10):
            sfx.add(c0 + k * 0.04, blip(600 + k * 90 + i * 60, 0.03, 0.05, 0.5), 1.0, -0.2)
        sfx.add(c0 + 0.42, blip(mtof(84 + i * 2), 0.09, 0.07, 0.5), 1.0, -0.2)
    # ---- objects: launch whoosh + impacts + bounces
    for i in range(4):
        L = LV[i]
        sz = D.obj[physics.LEVELS[i]['key']].shape[1]
        v = SPEED[i]
        # approximate real launch time
        t_launch = L['impact'] - (CONTACT[i][0] + sz + 4) / v
        if L['slow']:
            t_launch -= (L['impact'] - L['slow'][0]) * (1 - L['slow'][1])
        if i < 3:
            sfx.add(t_launch - 0.05, pulse(np.linspace(300, 900, int(0.08 * SR)), 0.08, 0.5) * env(int(0.08 * SR), 0.002, 0.06, 0, 0.01) * 0.06, 1.0, -0.8)
            sfx.add(t_launch, whoosh(L['impact'] - t_launch + 0.05, 800, 3000, 0.12 + 0.05 * i, seed=50 + i), 1.0, -0.6)
        else:
            # bowling: rolling rumble
            d = L['impact'] - t_launch
            n = int(d * SR)
            rum = lp(noise(d, 600, seed=60), 180) * np.linspace(0.2, 1, n) * 0.5
            sfx.add(t_launch, rum, 1.0, -0.5)
    # ball bounces after impact (times taken from the simulated paths)
    for i, (f0, vol, kind) in enumerate([(2600, 0.12, 'pok'), (900, 0.14, 'pomp'), (180, 0.4, 'bump')]):
        xs, ys, rs = D.paths[i]
        sz = D.obj[physics.LEVELS[i]['key']].shape[1]
        ground = TL.FEET_Y_AUDIO if hasattr(TL, 'FEET_Y_AUDIO') else None
        yb = ys + sz
        gy = yb.max()
        hits = [k for k in range(1, len(yb) - 1) if yb[k] >= gy - 0.01 and yb[k - 1] < gy - 0.01]
        t_after = TL.LV[i]['impact'] + TL.LV[i]['hitstop'] / TL.FPS
        for n_, k in enumerate(hits[:5]):
            tb = t_after + k / 240.0
            g = vol * (0.75 ** n_)
            if kind == 'bump':
                sfx.add(tb, thump(f0, 60, 0.15, g * 1.6, 0.04), 1.0, -0.4)
            else:
                nn = int(0.04 * SR)
                sfx.add(tb, sine(f0 * (1 + 0.05 * n_), 0.04) * expdec(nn, 0.01) * g, 1.0, -0.4)
    # background car in the intro
    sfx.add(1.45, whoosh(1.1, 150, 900, 0.12, seed=120) + lp(noise(1.1, 400, seed=121), 120) * np.sin(np.linspace(0, np.pi, int(1.1 * SR))) * 0.15, 1.0, 0.0)
    # L1: "pik"
    t = LV[0]['impact']
    sfx.add(t, sine(2200, 0.05) * expdec(int(0.05 * SR), 0.012) * 0.3, 1.0, -0.1)
    sfx.add(t, blip(3300, 0.02, 0.08, 0.5))
    # L2: "thwock"
    t = LV[1]['impact']
    sfx.add(t, pulse(np.linspace(700, 180, int(0.07 * SR)), 0.07, 0.5) * expdec(int(0.07 * SR), 0.03) * 0.25)
    sfx.add(t, thump(140, 70, 0.15, 0.35, 0.05))
    sfx.add(t, thump(90, 50, 0.25, 0.35, 0.07))
    sfx.add(t, hp(noise(0.06, SR, seed=70), 1500) * expdec(int(0.06 * SR), 0.015) * 0.3)
    # L3: slow-mo air + BOOM
    sfx.add(13.5, lp(noise(0.5, SR, seed=80), 600) * np.linspace(0, 1, int(0.5 * SR)) ** 2 * 0.5, 1.0, -0.3)
    t = LV[2]['impact']
    sfx.add(t, thump(150, 42, 0.6, 1.1, 0.16))
    sfx.add(t, hp(noise(0.2, SR, seed=81), 800) * expdec(int(0.2 * SR), 0.05) * 0.5)
    sfx.add(t, pulse(np.linspace(500, 90, int(0.12 * SR)), 0.12, 0.5) * expdec(int(0.12 * SR), 0.04) * 0.2)
    sfx.add(t, crash(0.25, 1.2), 1.0, 0.2)
    # spin "whirr" + dizzy tweets
    sfx.add(t + 0.17, pulse(400 + 300 * np.sin(np.linspace(0, 12 * np.pi, int(0.7 * SR))), 0.7, 0.25) * env(int(0.7 * SR), 0.01, 0.1, 0.5, 0.2) * 0.05, 1.0, 0.2)
    for k in range(5):
        tt = t + 0.75 + k * 0.22
        n = int(0.09 * SR)
        f = np.linspace(2600, 3400, n) if k % 2 == 0 else np.linspace(3200, 2500, n)
        sfx.add(tt, sine(f, 0.09) * env(n, 0.005, 0.05, 0.3, 0.02) * 0.07, 1.0, 0.3 if k % 2 else -0.1)
    # L4: pins + STRIKE + fall + phone crack + boing + catch
    t = LV[3]['impact']
    sfx.add(t, thump(130, 40, 0.8, 1.0, 0.2))
    r = np.random.default_rng(90)
    for k in range(14):
        tk = t + r.uniform(0, 0.22)
        f = r.uniform(900, 2400)
        n = int(0.05 * SR)
        clk = (pulse(f, 0.05, 0.5) * 0.4 + hp(noise(0.05, SR, seed=91 + k), 1500)) * expdec(n, 0.012) * 0.16
        sfx.add(tk, clk, 1.0, r.uniform(-0.6, 0.6))
    sfx.add(t + 0.12, crash(0.28, 1.3), 1.0, 0.0)
    for k, nm in enumerate(['C5', 'E5', 'G5', 'C6']):
        sfx.add(t + 0.15 + k * 0.06, inst_lead(n2m(nm) + 2, 0.08 if k < 3 else 0.35, 0.16, duty=0.25))
    hs = LV[3]['hitstop'] / TL.FPS
    sfx.add(t + hs + 0.3, thump(110, 40, 0.3, 0.6, 0.08))                 # body hits the lane
    sfx.add(t + hs + 0.3, lp(noise(0.25, SR, seed=92), 900) * expdec(int(0.25 * SR), 0.06) * 0.3)
    sfx.add(18.22, blip(900, 0.04, 0.06, 0.5))                              # phone lands
    sfx.add(18.27, glass(0.35, 0.35, seed=93))                              # screen cracks
    sfx.add(18.27, thump(400, 150, 0.08, 0.2, 0.02))
    up_t = t + hs + 1.12
    n = int(0.28 * SR)
    boing = pulse(220 * (1 + 0.5 * np.sin(np.linspace(0, 10 * np.pi, n)) * np.linspace(1, 0, n)) * np.linspace(1, 2.2, n), 0.28, 0.5)
    sfx.add(up_t, boing * env(n, 0.005, 0.1, 0.6, 0.08) * 0.09)
    sfx.add(t + hs + 1.4, blip(1400, 0.05, 0.08, 0.5)); sfx.add(t + hs + 1.44, blip(2100, 0.06, 0.06, 0.5))
    # result stamps
    for ts in (6.9, 10.9, 15.15, 18.66):
        sfx.add(ts, thump(200, 70, 0.18, 0.45, 0.05))
        sfx.add(ts, hp(noise(0.08, SR, seed=int(ts * 10)), 1200) * expdec(int(0.08 * SR), 0.02) * 0.2)
    # "wah-wah" for NO EFFECT
    for k, f in enumerate([392, 370]):
        n = int(0.22 * SR)
        sfx.add(7.05 + k * 0.24, pulse(f * (1 - 0.03 * np.linspace(0, 1, n)), 0.22, 0.5) * env(n, 0.01, 0.1, 0.6, 0.06) * 0.06)
    # ---- L5
    # e-bike motor whine (approach), pitch rises; slows in slow-mo
    t0, t1 = TL.BIKE_ENTER - 0.1, LV[4]['impact']
    d = t1 - t0
    n = int(d * SR)
    tt = np.arange(n) / SR + t0
    f = 180 + 60 * np.clip((tt - t0) / 0.5, 0, 1)
    f = np.where(tt >= 22.0, f * 0.55, f)
    mot = (pulse(f, d, 0.3) * 0.5 + pulse(f * 2.01, d, 0.5) * 0.25)
    mot = lp(mot, 2500) * (0.5 + 0.5 * np.sin(2 * np.pi * 30 * tt) ** 2) * np.clip((tt - t0) / 0.3, 0, 1) * 0.13
    sfx.add(t0, mot, 1.0, -0.4)
    sfx.add(TL.BELL_T, bell_ring(0.22), 1.0, -0.3)
    sfx.add(TL.ANC_T, blip(880, 0.06, 0.05, 0.5)); sfx.add(TL.ANC_T + 0.07, blip(660, 0.08, 0.05, 0.5))
    # slow-mo: deep air + heartbeat
    sfx.add(22.0, lp(noise(0.4, SR, seed=100), 400) * np.linspace(0.2, 1, int(0.4 * SR)) * 0.5)
    sfx.add(22.05, thump(70, 40, 0.2, 0.5, 0.06)); sfx.add(22.25, thump(70, 40, 0.2, 0.45, 0.06))
    # THE impact
    t = LV[4]['impact']
    sub = thump(95, 30, 2.0, 1.0, 0.55)
    sub = np.tanh(sub * 2.6) / np.tanh(2.6)          # saturate: denser without a taller peak
    sfx.add(t, sub * 1.05)
    n = int(1.4 * SR)
    tt = np.arange(n) / SR
    drop = np.sin(2 * np.pi * np.cumsum(55 * np.exp(-tt / 0.5) + 22) / SR) * np.exp(-tt / 0.6) * 0.5
    sfx.add(t + 0.02, drop)
    n = int(0.9 * SR)
    dist = np.tanh(lp(noise(0.9, 8000, seed=101), 3000) * 4) * expdec(n, 0.18) * 0.35
    sfx.add(t, bitcrush(dist, 5, 4), 1.0, 0.0)
    sfx.add(t, glass(0.7, 0.5, seed=102), 1.0, 0.15)
    sfx.add(t, crash(0.35, 2.0), 1.0, -0.15)
    hs = LV[4]['hitstop'] / TL.FPS
    # skid screech
    n = int(0.55 * SR)
    sk = bp(noise(0.55, SR, seed=103), 2200, 4200) * env(n, 0.01, 0.1, 0.7, 0.2) * 0.12
    sk *= 1 + 0.3 * np.sin(np.linspace(0, 60, n))
    sfx.add(t + hs, sk, 1.0, -0.2)
    # launch whoosh up + twinkle
    sfx.add(t + hs, whoosh(0.9, 300, 4000, 0.3, seed=104), 1.0, 0.1)
    sfx.add(t + hs + 0.95, twinkle(0.25), 1.0, 0.3)
    # TEST TERMINATED stamp + error buzz
    sfx.add(23.55, thump(180, 60, 0.25, 0.6, 0.07))
    n = int(0.3 * SR)
    sfx.add(23.6, pulse(110, 0.3, 0.5) * env(n, 0.005, 0.05, 0.8, 0.05) * 0.07)
    # (24.0 - 24.5 silence)
    sfx.add(TL.FALL_CAM, whoosh(0.5, 1500, 300, 0.07, seed=105))
    # falling whistle
    d = TL.LAND_T - TL.FALL_START + 0.12
    n = int(d * SR)
    fw = np.linspace(2200, 500, n)
    sfx.add(TL.FALL_START - 0.12, sine(fw, d) * env(n, 0.02, 0.05, 0.8, 0.03) * 0.12, 1.0, 0.1)
    # landing
    sfx.add(TL.LAND_T, thump(130, 40, 0.35, 0.85, 0.1))
    sfx.add(TL.LAND_T, lp(noise(0.3, SR, seed=106), 1200) * expdec(int(0.3 * SR), 0.07) * 0.35)
    sfx.add(TL.PHONE_CATCH, blip(1500, 0.04, 0.08, 0.5)); sfx.add(TL.PHONE_CATCH + 0.04, blip(2250, 0.06, 0.06, 0.5))
    # his headphones leaking music in the silence (tinny, quiet)
    leak = ['E5', 'A5', 'G5', 'E5', 'D5', 'C5']
    for k, nm in enumerate(leak):
        tk = TL.LAND_T + 0.08 + k * 0.08
        if tk >= TL.FREEZE_T:
            break
        s = hp(inst_lead(n2m(nm) + 12, 0.07, 0.5), 2500)
        sfx.add(tk, s * 0.05, 1.0, 0.35)
    # freeze: pause chime
    sfx.add(TL.FREEZE_T, blip(1047, 0.06, 0.08, 0.5)); sfx.add(TL.FREEZE_T + 0.07, blip(784, 0.1, 0.08, 0.5))
    sfx.add(TL.FREEZE_T, hp(noise(0.06, SR, seed=107), 3000) * expdec(int(0.06 * SR), 0.02) * 0.15)
    # end card slams
    for k, ts in enumerate([TL.END_CARD, TL.END_CARD + 0.18, TL.END_CARD + 0.36]):
        sfx.add(ts, thump(150 - k * 15, 45, 0.35, 0.75, 0.1))
        sfx.add(ts, hp(noise(0.1, SR, seed=110 + k), 1500) * expdec(int(0.1 * SR), 0.03) * 0.25)
    sfx.add(TL.END_CARD + 0.9, blip(1568, 0.05, 0.06, 0.5)); sfx.add(TL.END_CARD + 0.95, blip(2093, 0.08, 0.06, 0.5))


def build_ambience():
    """Faint city air: filtered noise + distant hum, ducked during the freeze."""
    d = DUR
    n = int(d * SR)
    base = lp(noise(d, SR, seed=200), 500) * 0.05 + lp(noise(d, SR, seed=201), 2000) * 0.012
    tt = np.arange(n) / SR
    g = np.ones(n)
    g[tt >= TL.FREEZE_T] = 0.25
    g *= np.clip(tt / 1.0, 0, 1)
    amb.add(0.0, base * g)


def master():
    L = music.L * 0.85 + sfx.L + amb.L
    R = music.R * 0.85 + sfx.R + amb.R
    # music cut at the final impact (hard), as scored
    cut = int(TL.MUSIC_CUT * SR)
    # (music bus already has nothing scheduled across the cut except tails) -> hard gate tails
    mgate = np.ones(len(L))
    mgate[cut:int(26.5 * SR)] = 0.0
    ramp = int(0.004 * SR)
    mgate[cut - ramp:cut] = np.linspace(1, 0, ramp)
    tt = np.arange(len(L)) / SR
    for (ti, depth_db, rel) in ((TL.LV[0]['impact'], -3, 0.25), (TL.LV[1]['impact'], -6, 0.35),
                                (TL.LV[2]['impact'], -10, 0.55), (TL.LV[3]['impact'], -11, 0.6)):
        g = 10 ** (depth_db / 20)
        m = tt >= ti
        k = np.clip((tt[m] - ti - 0.08) / rel, 0, 1)
        mgate[m] *= g + (1 - g) * (k * k * (3 - 2 * k))
    L = music.L * 0.85 * mgate + sfx.L + amb.L
    R = music.R * 0.85 * mgate + sfx.R + amb.R
    # gentle bus glue + limiter
    x = np.stack([L, R], 1)
    peak = np.max(np.abs(x))
    x = np.tanh(x * 1.1) / np.tanh(1.1)
    # true-peak normalise (4x oversampled estimate) to -1.5 dBTP
    over = signal.resample_poly(x, 4, 1, axis=0)
    tp = np.max(np.abs(over))
    x = x / tp * 10 ** (-2.0 / 20)
    # fade tail
    n = len(x)
    fo = int(0.45 * SR)
    end = int(TL.DURATION * SR)
    x[end - fo:end] *= np.linspace(1, 0, fo)[:, None]
    x[end:] = 0
    return x[:end]


def write_wav(path, x):
    import wave
    y = np.clip(x, -1, 1)
    y = (y * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'audio.wav'
    build_music()
    build_sfx()
    build_ambience()
    x = master()
    write_wav(out, x)
    print('wrote', out, x.shape, f'{len(x) / SR:.2f}s')
