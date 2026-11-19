"""Physics table + helpers. Kinetic energy is computed here, never typed by hand."""
import math


def kinetic_energy(m, v):
    return 0.5 * m * v * v


_RAW = [
    dict(key='pingpong', name_cn='乒乓球', name_en='PING-PONG BALL', m=0.0027, v=20.0,
         hit_word='啵', result_cn='毫无反应', result_en='NO EFFECT'),
    dict(key='tennis', name_cn='网球', name_en='TENNIS BALL', m=0.058, v=30.0,
         hit_word='嘭!', result_cn='晃了一下', result_en='WOBBLE'),
    dict(key='soccer', name_cn='足球', name_en='SOCCER BALL', m=0.43, v=25.0,
         hit_word='砰!!', result_cn='原地转圈+眩晕', result_en='SPIN + DIZZY'),
    dict(key='bowling', name_cn='保龄球', name_en='BOWLING BALL', m=7.26, v=9.0,
         hit_word='STRIKE!', result_cn='全中！手机碎屏', result_en='PHONE CRACKED'),
    dict(key='ebike', name_cn='电动自行车', name_en='E-BIKE', m=55.0 + 70.0, v=6.94,
         mass_note='车55+人70', speed_note='约25 km/h',
         hit_word='轰!!!', result_cn='测试终止', result_en='TEST TERMINATED'),
]

LEVELS = []
for r in _RAW:
    r = dict(r)
    r['ke'] = kinetic_energy(r['m'], r['v'])
    LEVELS.append(r)
for r in LEVELS:
    r['ratio'] = r['ke'] / LEVELS[0]['ke']


def fmt_num(x):
    """Format an input quantity compactly: 0.0027, 0.058, 20, 6.94, 7.26, 125."""
    if abs(x - round(x)) < 1e-9:
        return f'{int(round(x))}'
    if x < 0.01:
        return f'{x:.4f}'.rstrip('0')
    if x < 1:
        return f'{x:.3f}'.rstrip('0')
    return f'{x:.2f}'.rstrip('0').rstrip('.')


def round_half_up(x, nd=0):
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def fmt_ratio(r):
    return f'×{int(round_half_up(r)):,}'


def fmt_ke(e):
    """Format kinetic energy: <1 J -> 2 decimals, <1000 J -> 1 decimal, else thousands sep."""
    if e < 1:
        return f'{round_half_up(e, 2):.2f}'
    if e < 1000:
        return f'{round_half_up(e, 1):.1f}'
    return f'{int(round_half_up(e)):,}'


def ke_meter_frac(e, lo=-1.0, hi=4.0):
    """Log-scale position of energy e on a 0.1 J .. 10,000 J meter."""
    return min(1.0, max(0.0, (math.log10(max(e, 1e-9)) - lo) / (hi - lo)))


# ---- easing / motion helpers ----
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out_cubic(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_cubic(t):
    t = clamp(t)
    return t ** 3


def ease_in_out(t):
    t = clamp(t)
    return 3 * t * t - 2 * t * t * t


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_elastic(t):
    t = clamp(t)
    if t in (0, 1):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3) + 1


def bounce_height(t, h0, e=0.55, g=1.0, n=6):
    """Height of a ball dropped/launched with bounces: first arc peak h0, restitution e.
    t in 'arc units' where first arc lasts 1. Returns (height, landed_flag)."""
    dur = 1.0
    peak = h0
    tt = t
    for _ in range(n):
        if tt < dur:
            u = tt / dur
            return 4 * peak * u * (1 - u), False
        tt -= dur
        dur *= e
        peak *= e * e
    return 0.0, True


if __name__ == '__main__':
    for i, L in enumerate(LEVELS):
        print(i + 1, L['name_cn'], L['m'], 'kg', round(L['v'], 4), 'm/s', '->', round(L['ke'], 4), 'J',
              fmt_ke(L['ke']), 'J', f"x{L['ratio']:.1f}", f"meter={ke_meter_frac(L['ke']):.2f}")
