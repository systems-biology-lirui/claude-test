"""Single source of truth for timing. The renderer, the HUD and the audio engine all import this.

Two schedules:
  song  (default) - locked to the uploaded track 《路口淡然警告》: 150 BPM, beat n lands exactly on
                    frame 14 + 12n once the song is delayed by SONG_DELAY. Every event below is
                    derived from that beat grid (see tools/song_analyze.py for the analysis).
  synth           - the original cut with the self-synthesised chiptune (120 -> 150 BPM).
Select with the environment variable PKT_MODE=song|synth.
"""
import os

FPS = 30
MODE = os.environ.get('PKT_MODE', 'song')
_HERE = os.path.dirname(os.path.abspath(__file__))
SONG_PATH = os.environ.get('PKT_SONG', os.path.join(_HERE, '..', 'output', '《路口淡然警告》罗嘟嘟-电子.mp3'))


def q(t):
    """Snap a time to the frame grid."""
    return round(t * FPS) / FPS


if MODE == 'song':
    # measured grid: first downbeat 0.4492 s, period 0.40005 s (149.98 BPM); kick onsets within 0..9 ms
    SONG_T0 = 0.4492
    SONG_PERIOD = 0.40005
    SONG_DELAY = 14 / FPS - SONG_T0          # 17.5 ms of silence in front of the song -> beats on frames

    def beat(n):
        """Frame-exact time of song beat n (bar k downbeat = beat 4(k-1))."""
        return (14 + 12 * n) / FPS

    def beat_pos(t):
        # on the frame grid (beat n = frame 14 + 12n), so beat frames give exact integers
        return (round(t * FPS, 6) - 14) / 12.0

    def bpm_at(t):
        return 60.0 / SONG_PERIOD

    DURATION = 30.0
    LV = [
        dict(start=beat(8), impact=beat(14), hitstop=2, slow=None),                 # bar 3 groove in / bar 4 beat 3
        dict(start=beat(20), impact=beat(24), hitstop=3, slow=None),                # bar 6 / bar 7 section change
        dict(start=beat(28), impact=beat(36), hitstop=5, slow=(beat(32), 0.2)),     # breakdown = bullet time
        dict(start=beat(40), impact=beat(44), hitstop=4, slow=None),                # banner on the DROP / bar 12
        dict(start=beat(48), impact=beat(56), hitstop=8, slow=(beat(55), 0.25)),    # loud section / bar 15
    ]
    CARD_SPEED = [1.0, 1.35, 1.15, 1.6, 1.4]
    TILT_END = 1.75
    LOCK_T = beat(3)
    CAR_T = (1.1, 2.1)
    L3_CU_PRE = 0.4                  # close-up only for the last beat of the long slow motion
    RESULT_OFFSET = [0.9, 0.9, 0.95, 1.06]
    TERMINATED_T = beat(60)          # bar 16 downbeat
    LAND_T = beat(64)                # bar 17: outro starts as he lands
    NOTE_PERIOD = 2 * SONG_PERIOD
else:
    TEMPO = [(0.0, 120.0), (16.0, 150.0)]

    def bpm_at(t):
        b = TEMPO[0][1]
        for t0, bpm in TEMPO:
            if t >= t0:
                b = bpm
        return b

    def beat_pos(t):
        total = 0.0
        for i, (t0, bpm) in enumerate(TEMPO):
            t1 = TEMPO[i + 1][0] if i + 1 < len(TEMPO) else 1e9
            if t <= t0:
                break
            total += (min(t, t1) - t0) * bpm / 60.0
        return total

    SONG_DELAY = 0.0
    DURATION = 31.0
    LV = [
        dict(start=4.0, impact=6.0, hitstop=2, slow=None),
        dict(start=8.0, impact=10.0, hitstop=3, slow=None),
        dict(start=12.0, impact=14.0, hitstop=5, slow=(13.5, 0.25)),
        dict(start=16.0, impact=17.6, hitstop=4, slow=None),
        dict(start=19.2, impact=22.4, hitstop=8, slow=(22.0, 0.25)),
    ]
    CARD_SPEED = [1.0, 1.0, 1.3, 1.6, 1.4]
    TILT_END = 1.95
    LOCK_T = 2.0
    CAR_T = (1.45, 2.55)
    L3_CU_PRE = 0.5
    RESULT_OFFSET = [0.9, 0.9, 1.15, 1.06]
    TERMINATED_T = 23.55
    LAND_T = 25.2
    NOTE_PERIOD = 0.55

def r9(x):
    """Strip float noise from derived times (25.2 + 0.4 -> 25.6 exactly)."""
    return round(x, 9)


N_FRAMES = int(round(DURATION * FPS))
IMPACTS = [L['impact'] for L in LV]

# ---- intro
SUBJECT_T = r9(LOCK_T + 0.35)
INTRO_HUD_END = r9(LV[0]['start'] - 0.05)

# ---- level 5 (relative to the warning, identical in both cuts)
WARNING_START = LV[4]['start']
WARNING_END = r9(WARNING_START + 1.4)
L5_CARD = r9(WARNING_START + 0.55)
L5_COUNT = r9(WARNING_START + 1.1)
BIKE_ENTER = r9(WARNING_START + 2.3)
BELL_T = r9(WARNING_START + 2.5)
ANC_T = r9(WARNING_START + 2.62)
CRACK_FADE = (r9(LV[4]['impact'] + 0.7), r9(LV[4]['impact'] + 1.1))

# ---- fall back + ending (relative to the landing)
FALL_CAM = r9(LAND_T - 0.7)           # camera tilts back down (0.5 s)
FALL_START = r9(LAND_T - 0.34)
PHONE_CATCH = r9(LAND_T + 0.18)
FREEZE_T = r9(LAND_T + 0.4)
END_CARD = r9(LAND_T + 0.8)
FADE_OUT = r9(DURATION - 0.45)
MUSIC_CUT = LV[4]['impact']        # synth score stops here; in song mode the track plays on

# ---- level 4 phone close-up insert
PHONE_INSERT = (r9(LV[3]['impact'] + 0.62), r9(LV[3]['impact'] + 1.02))


def card_start(i):
    return LV[i]['start'] + 0.15 if i < 4 else L5_CARD


def count_start(i):
    """KE counter roll-up start (0.4 s long)."""
    return card_start(i) + 1.0 / CARD_SPEED[i] if i < 4 else L5_COUNT


def traj_start(i):
    return card_start(i) + 1.3 / CARD_SPEED[i]


def result_time(i):
    return r9(LV[i]['impact'] + RESULT_OFFSET[i])


# ---- time warp (hit-stop + slow motion)
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


def real_time_of_world(w):
    """Inverse of world_time (earliest real time reaching world time w)."""
    lo, hi = 0.0, DURATION + 5
    for _ in range(60):
        mid = (lo + hi) / 2
        if world_time(mid) < w:
            lo = mid
        else:
            hi = mid
    return hi


W_IMPACT = [world_time(L['impact']) for L in LV]


def since_impact(t, i):
    return world_time(t) - W_IMPACT[i]


def frame_of(t):
    return int(round(t * FPS))


if __name__ == '__main__':
    print('mode', MODE, 'duration', DURATION, 'frames', N_FRAMES, 'song delay', round(SONG_DELAY, 4))
    for i, L in enumerate(LV):
        s = f" slow {L['slow'][0]:.3f}" if L['slow'] else ''
        print(f"L{i+1}: start {L['start']:.3f}s (f{frame_of(L['start'])}) impact {L['impact']:.3f}s "
              f"(f{frame_of(L['impact'])}) beat {beat_pos(L['impact']):.3f}{s}")
    for k in ('LOCK_T', 'WARNING_START', 'BIKE_ENTER', 'BELL_T', 'TERMINATED_T', 'FALL_CAM', 'LAND_T',
              'FREEZE_T', 'END_CARD', 'FADE_OUT'):
        v = globals()[k]
        print(f'{k:14s} {v:7.3f}s  beat {beat_pos(v):6.2f}')
