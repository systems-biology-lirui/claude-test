"""Single source of truth for timing. Both the renderer and the audio engine import this.

Tempo: 120 BPM (beat = 15 frames, bar = 2 s) until 16.0 s, then 150 BPM (beat = 12 frames,
bar = 1.6 s). Every impact lands on a bar downbeat.
"""
FPS = 30
DURATION = 31.0
N_FRAMES = int(round(DURATION * FPS))

TEMPO = [(0.0, 120.0), (16.0, 150.0)]


def bpm_at(t):
    b = TEMPO[0][1]
    for t0, bpm in TEMPO:
        if t >= t0:
            b = bpm
    return b


def beat_pos(t):
    """Continuous beat counter (beats since 0, respecting the tempo map)."""
    total = 0.0
    for i, (t0, bpm) in enumerate(TEMPO):
        t1 = TEMPO[i + 1][0] if i + 1 < len(TEMPO) else 1e9
        if t <= t0:
            break
        seg = min(t, t1) - t0
        total += seg * bpm / 60.0
    return total


def beat_time(b):
    """Inverse of beat_pos."""
    acc = 0.0
    for i, (t0, bpm) in enumerate(TEMPO):
        t1 = TEMPO[i + 1][0] if i + 1 < len(TEMPO) else 1e9
        nb = (t1 - t0) * bpm / 60.0
        if b <= acc + nb:
            return t0 + (b - acc) * 60.0 / bpm
        acc += nb
    return None


# ---- level schedule (seconds) ----
LV = [
    dict(start=4.0, impact=6.0, hitstop=2, slow=None),
    dict(start=8.0, impact=10.0, hitstop=3, slow=None),
    dict(start=12.0, impact=14.0, hitstop=5, slow=(13.5, 0.25)),
    dict(start=16.0, impact=17.6, hitstop=4, slow=None),
    dict(start=19.2, impact=22.4, hitstop=8, slow=(22.0, 0.25)),
]

# HUD card schedule (shared by renderer and audio): typing speed multiplier per level
CARD_SPEED = [1.0, 1.0, 1.3, 1.6, 1.4]


def card_start(i):
    return LV[i]['start'] + 0.15 if i < 4 else 19.75


def count_start(i):
    """KE counter roll-up start (0.4 s long)."""
    return card_start(i) + 1.0 / CARD_SPEED[i] if i < 4 else 20.3


def traj_start(i):
    return card_start(i) + 1.3 / CARD_SPEED[i]


PHONE_INSERT = (18.22, 18.62)

INTRO_TITLE = 0.0
INTRO_LOCK = 2.0
WARNING_START = 19.2
WARNING_END = 20.6
BIKE_ENTER = 21.5
BELL_T = 21.7
ANC_T = 21.82
LAUNCH_END = 24.0           # twinkle done
FALL_CAM = 24.5             # camera tilts back down
FALL_START = 24.86
LAND_T = 25.2
PHONE_CATCH = 25.38
FREEZE_T = 25.6
END_CARD = 26.0
FADE_OUT = 30.55
MUSIC_CUT = LV[4]['impact']

# ---- time warp (hit-stop + slow motion) ----
WARP = []
for L in LV:
    if L['slow']:
        WARP.append((L['slow'][0], L['impact'], L['slow'][1]))
    WARP.append((L['impact'], L['impact'] + L['hitstop'] / FPS, 0.0))
WARP.sort()


def world_time(t):
    """Monotonic world clock: slows in slow-mo windows, stops during hit-stop."""
    w = 0.0
    last = 0.0
    for (a, b, r) in WARP:
        if t <= a:
            break
        w += a - last
        w += (min(t, b) - a) * r
        last = min(t, b)
        if t <= b:
            return w
    return w + (t - last)


W_IMPACT = [world_time(L['impact']) for L in LV]


def since_impact(t, i):
    return world_time(t) - W_IMPACT[i]


def frame_of(t):
    return int(round(t * FPS))


if __name__ == '__main__':
    for i, L in enumerate(LV):
        print(f"L{i+1}: start {L['start']:.2f}s impact {L['impact']:.2f}s (frame {frame_of(L['impact'])}) "
              f"beat {beat_pos(L['impact']):.3f}")
    print('frames', N_FRAMES)
