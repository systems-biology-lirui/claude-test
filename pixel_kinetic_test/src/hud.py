"""Arcade / lab HUD components drawn on the 270x480 canvas (screen space)."""
import math
import numpy as np
import gfx
from gfx import P, BAYER4
import font as F
from physics import LEVELS, ke_meter_frac, fmt_ke, clamp, ease_out_cubic, ease_out_back, lerp
import copy_text as T

W, H = 270, 480


class TextCache:
    def __init__(self):
        self.f12 = F.Font(12)
        self.f10 = F.Font(10)
        self.cache = {}

    def get(self, s, size=12, **kw):
        key = (s, size, tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in kw.items())))
        if key not in self.cache:
            f = self.f12 if size == 12 else self.f10
            self.cache[key] = F.text_sprite(f, s, **kw)
        return self.cache[key]

    def width(self, s, size=12, scale=1, bold=False, latin='proportional'):
        f = self.f12 if size == 12 else self.f10
        return f.width(s, latin=latin, bold=bold, scale=scale)


TX = None


def tx():
    global TX
    if TX is None:
        TX = TextCache()
    return TX


def text(canvas, s, x, y, size=12, align='left', **kw):
    """Draw text; (x, y) is the top-left of the line box (or centre/right per align).
    Returns the sprite width."""
    spr = tx().get(s, size, **kw)
    pad = 1 + (max(kw.get('shadow_off', (1, 1))) if kw.get('shadow') is not None else 0)
    w = spr.shape[1] - 2 * pad
    if align == 'center':
        x = x - w / 2
    elif align == 'right':
        x = x - w
    gfx.blit_hard(canvas, spr, int(round(x)) - pad, int(round(y)) - pad)
    return w


def typed(s, t, cps=40.0):
    """Typewriter: portion of string visible after t seconds."""
    if t <= 0:
        return ''
    n = int(t * cps) + 1
    return s[:n]


# ---------------------------------------------------------------- panels
def panel(canvas, x, y, w, h, alpha=0.86, bg='hud_bg', line='hud_line', corner='yel'):
    gfx.fill_rect(canvas, x, y, w, h, bg, alpha)
    gfx.fill_rect(canvas, x, y, w, 1, line); gfx.fill_rect(canvas, x, y + h - 1, w, 1, line)
    gfx.fill_rect(canvas, x, y, 1, h, line); gfx.fill_rect(canvas, x + w - 1, y, 1, h, line)
    L = 5
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x + w - 1, y, -1, 1), (x, y + h - 1, 1, -1), (x + w - 1, y + h - 1, -1, -1)):
        gfx.fill_rect(canvas, min(cx, cx + dx * (L - 1)), cy, L, 1, corner)
        gfx.fill_rect(canvas, cx, min(cy, cy + dy * (L - 1)), 1, L, corner)


def lamps(canvas, x, y, n_lit, t, all_red=False):
    cols = ['grn', 'grn', 'yel', 'org', 'red']
    for i in range(5):
        cx = x + i * 10
        m = gfx.circle_mask(3)
        lit = i < n_lit
        c = ('red' if all_red else cols[i]) if lit else 'gry2'
        if lit and (all_red or i == n_lit - 1) and (int(t * 6) % 2 == 1):
            c = 'red3' if all_red else c
        gfx.blit_mask(canvas, gfx.circle_mask(4), cx - 4, y - 4, 'ink')
        gfx.blit_mask(canvas, m, cx - 3, y - 3, c)
        if lit:
            gfx.put(canvas, cx - 1, y - 2, 'white', 0.8)


# ---------------------------------------------------------------- level card (top)
CARD = dict(x=8, y=26, w=254, h=108)


def level_card(canvas, lv, tl, sprites, t, slide=1.0, typing_speed=1.0, lamps_lit=None, compact=False,
               hurry=False):
    """tl: seconds since the card started appearing."""
    if tl < 0:
        return
    L = T.level_lines(lv)
    x, y, w, h = CARD['x'], CARD['y'], CARD['w'], CARD['h']
    oy = int(-(h + 14) * (1 - ease_out_cubic(slide)))
    if compact:
        h2 = 22
        panel(canvas, x, y + oy, w, h2)
        text(canvas, L['level'], x + 6, y + oy + 5, 12, color='yel', bold=True)
        text(canvas, L['name_cn'], x + 70, y + oy + 5, 12, color='wht')
        text(canvas, L['ke'], x + w - 8, y + oy + 5, 12, align='right', color='cyn', latin='monospaced')
        return
    panel(canvas, x, y + oy, w, h)
    yy = y + oy
    # header strip
    gfx.fill_rect(canvas, x + 1, yy + 1, w - 2, 15, 'hud_bg2')
    lvtxt = L['level'] + ('  FINAL' if lv == 4 else '')
    text(canvas, lvtxt, x + 6, yy + 3, 12, color='red' if lv == 4 else 'yel', bold=True)
    if hurry and int(t * 5) % 2 == 0:
        text(canvas, T.HURRY, x + 92, yy + 4, 10, color='red')
    lamps(canvas, x + w - 50, yy + 8, lv + 1 if lamps_lit is None else lamps_lit, t, all_red=(lv == 4))
    s = typing_speed
    # icon box
    bx, by = x + 6, yy + 20
    gfx.fill_rect(canvas, bx, by, 36, 36, 'ink', 0.9)
    for k in range(bx + 2, bx + 36, 6):
        gfx.fill_rect(canvas, k, by + 1, 1, 34, 'hud_line', 0.6)
    for k in range(by + 2, by + 36, 6):
        gfx.fill_rect(canvas, bx + 1, k, 34, 1, 'hud_line', 0.6)
    gfx.fill_rect(canvas, bx, by, 36, 1, 'yel2'); gfx.fill_rect(canvas, bx, by + 35, 36, 1, 'yel2')
    icon_t = (tl - 0.15 / s)
    if icon_t > 0:
        key = LEVELS[lv]['key']
        if key == 'ebike':
            spr = sprites['ebike_icon']
        else:
            spr = gfx.scale_nearest(sprites[key + '_big'], 2)
        pop = ease_out_back(clamp(icon_t / 0.25))
        if pop < 0.99:
            spr = gfx.scale_nearest(spr, max(0.1, pop))
        bob = int(round(math.sin(t * 5) * 1.0))
        gfx.blit_hard(canvas, spr, bx + 18 - spr.shape[1] // 2, by + 18 - spr.shape[0] // 2 + bob)
    # name
    nt = tl - 0.2 / s
    if nt > 0:
        name = typed(L['name_cn'], nt, 14 * s)
        text(canvas, name, x + 48, yy + 20, 12, scale=2, color='wht', outline='ink')
        if nt > 0.25 / s:
            text(canvas, typed(L['name_en'], nt - 0.25 / s, 40 * s), x + 49, yy + 46, 10, color='gry')
    rows = [
        (0.55, T.LABEL_MASS, L['mass'], L['mass_note']),
        (0.75, T.LABEL_SPEED, L['speed'], L['speed_note']),
    ]
    for i, (t0, (cn, en), val, note) in enumerate(rows):
        rt = tl - t0 / s
        if rt <= 0:
            continue
        ry = yy + 59 + i * 14
        text(canvas, cn, x + 6, ry, 12, color='yel')
        text(canvas, en, x + 32, ry + 2, 10, color='gry')
        if note:
            text(canvas, '(' + note + ')', x + 74, ry, 12, color='gry')
        v = typed(val, rt, 30 * s)
        text(canvas, v, x + w - 8, ry, 12, align='right', color='wht', latin='monospaced')
    ct = tl - 0.95 / s
    if ct > 0:
        ry = yy + 89
        gfx.fill_rect(canvas, x + 4, ry - 2, w - 8, 15, 'hud_bg2')
        text(canvas, T.LABEL_KE[0], x + 6, ry, 12, color='yel')
        text(canvas, T.LABEL_KE[1], x + 32, ry + 2, 10, color='gry')
        calc = typed(L['calc'], ct, 45 * s)
        text(canvas, calc, x + w - 8 - tx().width(L['calc'], latin='monospaced'), ry, 12, color='cyn', latin='monospaced')


# ---------------------------------------------------------------- KE meter (bottom)
METER = dict(x=8, y=318, w=254, h=70)


def ke_panel(canvas, ke_value, frac, lv, t, alpha_t=1.0, ratio_txt=None, flash=False):
    """Bottom panel. The big number sits on the LEFT so platform buttons on the right never cover it."""
    x, y, w, h = METER['x'], METER['y'], METER['w'], METER['h']
    oy = int((1 - ease_out_cubic(alpha_t)) * 90)
    y = y + oy
    panel(canvas, x, y, w, h)
    text(canvas, T.LABEL_KE[0], x + 6, y + 4, 12, color='yel')
    text(canvas, 'KINETIC ENERGY', x + 32, y + 6, 10, color='gry')
    val = fmt_ke(ke_value) + ' J'
    col = 'wht'
    if lv == 4 and int(t * 8) % 2 == 0:
        col = 'red3'
    if flash:
        col = 'yel3'
    vw = text(canvas, val, x + 6, y + 17, 12, scale=2, color=col, latin='monospaced', outline='ink', bold=True)
    if ratio_txt:
        rw = tx().width(ratio_txt, 10) + 6
        rx = x + 6 + vw + 8
        gfx.fill_rect(canvas, rx, y + 27, rw, 11, 'org')
        text(canvas, ratio_txt, rx + 3, y + 27, 10, color='ink')
    # segmented log meter
    mx, my, mw = x + 6, y + 43, w - 12
    nseg = 25
    segw = mw / nseg
    for i in range(nseg):
        f = (i + 0.5) / nseg
        segc = 'grn' if f < 0.35 else 'yel' if f < 0.6 else 'org' if f < 0.8 else 'red'
        sx = int(mx + i * segw)
        ex = int(mx + (i + 1) * segw) - 1
        lit = f <= frac
        c = segc if lit else 'hud_line'
        if lit and (i + 1) / nseg > frac - 1 / nseg and int(t * 10) % 2 == 0:
            c = 'white'
        gfx.fill_rect(canvas, sx, my, ex - sx, 8, c)
        if lit:
            gfx.fill_rect(canvas, sx, my, ex - sx, 1, 'white', 0.5)
    labels = ['0.1', '1', '10', '100', '1k', '10k']
    for i, lab in enumerate(labels):
        tx_ = mx + i / 5 * (mw - 1)
        gfx.fill_rect(canvas, int(tx_), my + 9, 1, 3, 'gry')
        lw = tx().width(lab, 10)
        ax = min(max(tx_ - lw / 2, mx), mx + mw - lw - 6)
        text(canvas, lab, ax, my + 12, 10, color='gry')
    text(canvas, 'J', mx + mw - 1, my + 12, 10, align='right', color='gry2')


# ---------------------------------------------------------------- banners & stamps
def hazard_bar(canvas, y, h, phase, colors=('yel', 'ink')):
    for x in range(-h, W + h):
        for k in range(h):
            if ((x + k + int(phase)) // 4) % 2 == 0:
                gfx.put(canvas, x, y + k, colors[0])
            else:
                gfx.put(canvas, x, y + k, colors[1])


def level_banner(canvas, label, tl, dur=0.55, sub=None, sub_col='red', t=0.0):
    """Slam-in banner across the middle of the screen."""
    if tl < 0 or tl > dur:
        return
    cy = 192
    k_in = ease_out_cubic(clamp(tl / 0.12))
    k_out = clamp((tl - (dur - 0.12)) / 0.12)
    bh = int(52 * k_in * (1 - k_out))
    if bh <= 2:
        return
    gfx.fill_rect(canvas, 0, cy - bh // 2, W, bh, 'ink', 0.92)
    hazard_bar(canvas, cy - bh // 2, 3, tl * 60)
    hazard_bar(canvas, cy + bh // 2 - 3, 3, -tl * 60)
    if bh > 30:
        dx = int((1 - ease_out_back(clamp(tl / 0.18))) * 160) - int(k_out * 220)
        text(canvas, label, W // 2 + dx, cy - 18, 12, align='center', scale=3, bold=True,
             gradient=['yel3', 'yel', 'yel', 'org'], outline='ink', shadow='red2', shadow_off=(2, 2))
        if sub and int(t * 6) % 2 == 0:
            text(canvas, sub, W // 2 - dx, cy + 14, 10, align='center', color=sub_col)


def stamp(canvas, cn, en, tl, color='red', cx=W // 2, cy=196, angle=0):
    """Axis-aligned rubber stamp (no rotation: keeps pixel text crisp). Slams in from 2.2x."""
    if tl < 0:
        return
    t1 = tx()
    s_cn = t1.get(cn, 12, scale=2, color='wht', outline='ink')
    s_en = t1.get(en, 10, color='red3' if color == 'red' else 'wht')
    wbox = max(s_cn.shape[1], s_en.shape[1]) + 14
    hbox = s_cn.shape[0] + s_en.shape[0] + 4
    can = np.zeros((hbox + 6, wbox + 6, 3), np.float32)
    can[:] = P[color]
    can[2:-2, 2:-2] = P['ink']
    can[3:-3, 3:-3] = P['red2'] if color == 'red' else P[color]
    # worn-ink texture on the outer border
    rough = gfx.dither_mask(can.shape[0], can.shape[1], 0.18, 1, 2)
    edge = np.zeros(can.shape[:2], bool); edge[:2] = True; edge[-2:] = True; edge[:, :2] = True; edge[:, -2:] = True
    can[edge & rough] = P['red2'] if color == 'red' else P['ink']
    gfx.blit_hard(can, s_cn, (can.shape[1] - s_cn.shape[1]) // 2, 3)
    gfx.blit_hard(can, s_en, (can.shape[1] - s_en.shape[1]) // 2, 3 + s_cn.shape[0] - 3)
    box = np.zeros((can.shape[0], can.shape[1], 4), np.uint8)
    box[..., :3] = gfx.to_uint8(can); box[..., 3] = 255
    k = clamp(tl / 0.1)
    sc = 1.0 + 1.2 * (1 - ease_out_cubic(k))
    if sc > 1.01:
        box = gfx.scale_nearest(box, sc)
    jx = int(round(math.sin(tl * 80) * 2 * math.exp(-tl * 12))) if tl > 0.1 else 0
    gfx.blit_hard(canvas, box, cx - box.shape[1] // 2 + jx, cy - box.shape[0] // 2)


def impact_word(canvas, word, sx, sy, tl, scale=2, palette=('yel3', 'yel', 'org'), shake=0, dur=0.7):
    if tl < 0 or tl > dur:
        return
    pop = ease_out_back(clamp(tl / 0.1), 2.5)
    sc = max(1, int(round(scale * pop))) if pop < 1 else scale
    dx = int(round(math.sin(tl * 90) * shake)) if shake else 0
    spr = tx().get(word, 12, scale=sc, bold=True, gradient=list(palette), outline='ink', shadow='ink',
                   shadow_off=(1, 1), latin='proportional')
    if tl > dur - 0.15:
        a = (dur - tl) / 0.15
        m = gfx.dither_mask(spr.shape[0], spr.shape[1], a)
        spr = spr.copy(); spr[~m, 3] = 0
    gfx.blit_hard(canvas, spr, int(sx - spr.shape[1] / 2 + dx), int(sy - spr.shape[0] / 2 - tl * 14))


# ---------------------------------------------------------------- intro
def title(canvas, tl, t):
    if tl < 0:
        return
    out = clamp((tl - 1.4) / 0.3)
    dy = -int(ease_out_cubic(out) * 300)
    band = clamp(tl / 0.3) * (1 - out)
    if band > 0:
        gfx.fill_rect(canvas, 0, 108 + dy // 3, W, 126, 'ink', 0.55 * band)
    lines = [(T.TITLE_CN_1, 3, ['wht', 'wht', 'neon_cyan2'], 'ink', 'blue0', 120),
             (T.TITLE_CN_2, 2, ['yel3', 'yel', 'org'], 'ink', 'red2', 168)]
    for i, (s, sc, grad, ol, sh, y) in enumerate(lines):
        st = tl - 0.15 - i * 0.25
        if st <= 0:
            continue
        # per-character drop-in
        n = len(s)
        cw = 12 * sc
        total = n * cw
        x0 = W // 2 - total // 2
        for j, ch in enumerate(s):
            ct = st - j * 0.05
            if ct <= 0:
                continue
            dd = int((1 - ease_out_back(clamp(ct / 0.22), 2.2)) * -40)
            spr = tx().get(ch, 12, scale=sc, bold=True, gradient=grad, outline=ol, shadow=sh, shadow_off=(2, 2))
            gfx.blit_hard(canvas, spr, x0 + j * cw - 2, y + dd + dy)
    et = tl - 0.75
    if et > 0:
        s = typed(T.TITLE_EN, et, 60)
        text(canvas, s, W // 2, 200 + dy, 10, align='center', color='wht')
        gfx.fill_rect(canvas, 40, 214 + dy, int(190 * ease_out_cubic(clamp(et / 0.4))), 1, 'yel')
        text(canvas, 'PIXEL TRAFFIC LAB', W // 2, 218 + dy, 10, align='center', color='gry')


def reticle(canvas, tl, box, t):
    """Corner brackets that shrink onto box=(x0,y0,x1,y1)."""
    if tl < 0:
        return
    k = ease_out_cubic(clamp(tl / 0.35))
    x0 = lerp(6, box[0], k); y0 = lerp(130, box[1], k)
    x1 = lerp(W - 7, box[2], k); y1 = lerp(420, box[3], k)
    col = 'red' if (tl > 0.35 and int(t * 8) % 2 == 0) else 'yel'
    L = 7
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        for th in range(2):
            gfx.fill_rect(canvas, int(min(cx, cx + dx * (L - 1))), int(cy + dy * th), L, 1, col)
            gfx.fill_rect(canvas, int(cx + dx * th), int(min(cy, cy + dy * (L - 1))), 1, L, col)
    if tl > 0.35:
        if int(t * 6) % 2 == 0 or tl > 0.9:
            text(canvas, T.TARGET_LOCKED, (x0 + x1) / 2, y0 - 13, 10, align='center', color='red', outline='ink')
        # crosshair ticks
        cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
        gfx.fill_rect(canvas, int(cxm), int(y0) - 3, 1, 3, col)
        gfx.fill_rect(canvas, int(cxm), int(y1) + 1, 1, 3, col)


def subject_card(canvas, tl, anchor, t, lines=None, highlight=None):
    if tl < 0:
        return
    x, y, w, h = 8, 150, 120, 66
    k = ease_out_cubic(clamp(tl / 0.2))
    ww = max(2, int(w * k))
    panel(canvas, x, y, ww, h, corner='red')
    if k < 0.95:
        return
    # leader line to the subject
    gfx.line(canvas, x + w, y + h - 6, anchor[0], anchor[1], 'yel', dash=(2, 1))
    gfx.fill_rect(canvas, anchor[0] - 1, anchor[1] - 1, 3, 3, 'yel')
    text(canvas, T.SUBJECT_HEAD, x + 5, y + 4, 12, color='yel')
    text(canvas, T.SUBJECT_HEAD_EN, x + 55, y + 6, 10, color='gry')
    lines = lines or T.SUBJECT_LINES
    for i, s in enumerate(lines):
        lt = tl - 0.25 - i * 0.22
        if lt <= 0:
            continue
        col = 'wht'
        if highlight is not None and i == highlight and int(t * 6) % 2 == 0:
            col = 'red3'
        part = typed(s, lt, 30)
        if '0%' in s and part.endswith('0%'):
            pre = part[:-2]
            wpre = text(canvas, pre, x + 5, y + 19 + i * 15, 12, color=col)
            text(canvas, '0%', x + 5 + wpre, y + 19 + i * 15, 12, color='red', bold=True)
        else:
            wv = text(canvas, part, x + 5, y + 19 + i * 15, 12, color=col)
            if i == 0 and part == s:
                text(canvas, '', x + 5 + wv + 2, y + 19, 12, color='grn')


def warning_overlay(canvas, tl, t):
    if tl < 0:
        return
    k = ease_out_cubic(clamp(tl / 0.15))
    cy = 186
    bh = int(64 * k)
    gfx.fill_rect(canvas, 0, cy - bh // 2, W, bh, 'ink', 0.85)
    hazard_bar(canvas, cy - bh // 2, 5, t * 50, ('red', 'ink'))
    hazard_bar(canvas, cy + bh // 2 - 5, 5, -t * 50, ('red', 'ink'))
    if bh > 40:
        on = int(t * 4) % 2 == 0
        grad = ['red3', 'red', 'red2'] if on else ['wht', 'red3', 'red']
        text(canvas, T.WARNING, W // 2, cy - 20, 12, align='center', scale=3, bold=True, gradient=grad,
             outline='ink', shadow='red2', shadow_off=(2, 2))
        for sx in (18, W - 30):
            text(canvas, '', sx, cy - 16, 12, scale=1, color='red' if on else 'yel')
        text(canvas, T.WARNING_CN, W // 2, cy + 13, 12, align='center', color='wht')


def bubble(canvas, s, x, y, tl, size=12, color='ink', bg='wht', tail='down'):
    """Speech bubble whose tail points at (x, y)."""
    if tl < 0:
        return
    sc = ease_out_back(clamp(tl / 0.15))
    tw = tx().width(s, size)
    bw, bh = tw + 10, 16
    bx = int(clamp(x - bw // 2 + 6, 3, W - bw - 3))
    by = int(y - bh - 6)
    if sc < 0.6:
        return
    gfx.fill_rect(canvas, bx - 1, by - 1, bw + 2, bh + 2, 'ink')
    gfx.fill_rect(canvas, bx, by, bw, bh, bg)
    x = int(clamp(x, bx + 4, bx + bw - 6))
    for k in range(5):
        gfx.fill_rect(canvas, x - 2 + k // 2, by + bh + k, max(1, 4 - k), 1, bg)
        gfx.put(canvas, x - 3 + k // 2, by + bh + k, 'ink')
    text(canvas, s, bx + 5, by + 2, size, color=color)


def tag(canvas, s, x, y, bg='cyn', fg='ink', size=12):
    w = tx().width(s, size) + 6
    hh = 14 if size == 12 else 11
    gfx.fill_rect(canvas, x - 1, y - 1, w + 2, hh + 2, 'ink')
    gfx.fill_rect(canvas, x, y, w, hh, bg)
    text(canvas, s, x + 3, y + 1, size, color=fg)


def slow_tag(canvas, t, rate=0.25):
    label = f'SLOW ×{rate:.2f}'
    w = max(70, tx().width(label, 10) + 15)
    gfx.fill_rect(canvas, 8, 132, w, 13, 'ink', 0.8)
    if int(t * 4) % 2 == 0:
        gfx.blit_mask(canvas, gfx.circle_mask(2), 11, 136, 'red')
    text(canvas, label, 18, 133, 10, color='wht')


def down_arrow(canvas, x, y, col='yel'):
    rows = ["..###..", "..###..", "..###..", "#######", ".#####.", "..###..", "...#..."]
    for j, r in enumerate(rows):
        for i, ch in enumerate(r):
            if ch == '#':
                gfx.put(canvas, x + i, y + j, col)
    for j, r in enumerate(rows):
        for i, ch in enumerate(r):
            if ch == '#':
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    xi, yj = i + dx, j + dy
                    if not (0 <= yj < len(rows) and 0 <= xi < len(r) and rows[yj][xi] == '#'):
                        gfx.put(canvas, x + xi, y + yj, 'ink')


def end_card(canvas, tl, t):
    """Final conclusion over the frozen frame. tl: seconds since END_CARD."""
    if tl < 0:
        return
    k = ease_out_cubic(clamp(tl / 0.25))
    # dark gradient plate behind the text
    for y in range(40, 206):
        a = 0.78 * k * (1 - max(0, (y - 180) / 26))
        gfx.fill_rect(canvas, 0, y, W, 1, 'ink', a)
    def slam(s, size, sc, y, t0, grad, sh, bold=False):
        u = tl - t0
        if u < 0:
            return
        spr = tx().get(s, size, scale=sc, gradient=grad, outline='ink', shadow=sh, shadow_off=(2, 2), bold=bold)
        z = 1.0 + 1.0 * (1 - ease_out_cubic(clamp(u / 0.12)))
        if z > 1.01:
            spr = gfx.scale_nearest(spr, z)
        gfx.blit_hard(canvas, spr, W // 2 - spr.shape[1] // 2, y - (spr.shape[0] - 40) // 2 if sc == 3 else y)
    slam(T.END_HEAD, 12, 2, 54, 0.0, ['yel3', 'yel', 'yel'], 'red2')
    slam(T.END_MAIN[:5], 12, 3, 90, 0.18, ['wht', 'wht', 'neon_cyan2'], 'red2')
    slam(T.END_MAIN[5:], 12, 3, 132, 0.36, ['wht', 'wht', 'neon_cyan2'], 'red2')
    et = tl - 0.6
    if et > 0:
        text(canvas, typed(T.END_EN, et, 50), W // 2, 180, 10, align='center', color='yel')
    # sidewalk highlight + pointer
    st = tl - 0.9
    if st > 0:
        on = int(t * 4) % 2 == 0
        y0, y1 = 216, 257
        for x in range(2, W - 2, 4):
            gfx.fill_rect(canvas, x, y0, 2, 1, 'yel' if on else 'yel2')
            gfx.fill_rect(canvas, x, y1, 2, 1, 'yel' if on else 'yel2')
        lab = T.END_ARROW
        tw = tx().width(lab, 12) + 10
        bx, by = 14, 198
        gfx.fill_rect(canvas, bx - 1, by - 1, tw + 2, 16, 'ink')
        gfx.fill_rect(canvas, bx, by, tw, 14, 'yel')
        text(canvas, lab, bx + 5, by + 1, 12, color='ink')
        bounce = int(abs(math.sin(t * 6)) * 3)
        down_arrow(canvas, bx + tw + 6, by + 2 + bounce, 'yel')
    nt = tl - 1.4
    if nt > 0:
        text(canvas, T.END_NOTE, W // 2, 352, 12, align='center', color='gry')
