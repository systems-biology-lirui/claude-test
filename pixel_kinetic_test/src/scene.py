"""Procedural pixel-art street at dusk.

Street-frame coordinates: camera (0,0) shows x 0..269, y 0..479.
Layers (parallax applies to vertical camera moves; horizontal shake is 1:1):
  sky      factor 0.30
  far      factor 0.60   (distant skyline silhouettes)
  near     factor 1.00   (facades, shops, sidewalk, lanes)
"""
import numpy as np
import gfx
from gfx import P, BAYER4
import font as F

W, H = 270, 480
MARGIN = 24                      # extra pixels left/right for shake
LW = W + 2 * MARGIN              # layer width

# street geometry (screen/world y, camera y = 0)
Y_SHOP_BASE = 215
Y_SIDEWALK = (215, 258)
Y_CURB = (258, 263)
Y_LANE = (263, 312)
Y_LINE = (312, 314)
Y_ROAD = (314, 520)
FEET_Y = 296                     # where the pedestrian stands (lane centre)
GROUND_Y = FEET_Y

SKY_FACTOR, FAR_FACTOR = 0.30, 0.60
SKY_TOP = -700                   # sky layer covers SKY_TOP .. 480 (in layer coords)
NEAR_TOP = -120                  # near layer covers NEAR_TOP .. 480
FAR_TOP = 0

rng = np.random.default_rng(7)


def _c(name):
    return P[name].astype(np.float32)


def _gradient_dither(h, stops, y0=0):
    """Vertical gradient quantised to palette colours with 4x4 ordered dithering.
    stops: list of (y, colour_name) in layer coordinates (increasing y)."""
    ys = np.array([s[0] for s in stops], np.float32)
    out = np.zeros((h, LW, 3), np.float32)
    yy = np.arange(h, dtype=np.float32) + y0
    idx = np.clip(np.searchsorted(ys, yy, side='right') - 1, 0, len(stops) - 2)
    t = np.clip((yy - ys[idx]) / (ys[idx + 1] - ys[idx]), 0, 1)
    thr = BAYER4[np.arange(h)[:, None] % 4, np.arange(LW)[None, :] % 4]
    pick_next = t[:, None] > thr
    ca = np.stack([_c(stops[i][1]) for i in idx])
    cb = np.stack([_c(stops[i + 1][1]) for i in idx])
    out[:] = np.where(pick_next[..., None], cb[:, None, :], ca[:, None, :])
    return out


# ---------------------------------------------------------------- sky
def build_sky():
    h = 520 - SKY_TOP
    stops = [(SKY_TOP, 'sky0'), (-420, 'sky0'), (-260, 'sky1'), (-130, 'sky2'), (-30, 'sky3'),
             (40, 'sky4'), (95, 'sky5'), (135, 'sky6'), (165, 'sky7'), (190, 'sky8'), (215, 'sky9'),
             (240, 'sky10'), (520, 'sky11')]
    img = _gradient_dither(h, stops, SKY_TOP)
    stars = []
    r = np.random.default_rng(11)
    for _ in range(260):
        x = int(r.integers(0, LW)); y = int(r.integers(SKY_TOP, 60))
        dens = np.clip((60 - y) / 400.0, 0.05, 1.0)
        if r.random() > dens:
            continue
        big = r.random() < 0.08
        stars.append((x, y, big, float(r.random() * 6.28), float(0.6 + r.random() * 2.0)))
        yy = y - SKY_TOP
        col = 'star' if r.random() < 0.6 else 'star2'
        img[yy, x] = _c(col)
        if big:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + dx < LW:
                    img[yy + dy, x + dx] = img[yy + dy, x + dx] * 0.4 + _c('star2') * 0.6
    # crescent moon + soft dithered halo
    mx, my, mr = 205, -170, 9
    R = mr + 6
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.sqrt(xx ** 2 + yy ** 2)
    d2 = np.sqrt((xx - 4) ** 2 + (yy + 3) ** 2)
    moon = (d <= mr + 0.3) & (d2 > mr - 0.5)
    halo = (d > mr + 0.5) & (d <= R) & (gfx.dither_mask(2 * R + 1, 2 * R + 1, 0.0) | (BAYER4[(yy + R) % 4, (xx + R) % 4] < 0.45 * (1 - (d - mr) / (R - mr))))
    y0 = my - SKY_TOP - R
    reg = img[y0:y0 + 2 * R + 1, mx - R:mx + R + 1]
    reg[halo] = reg[halo] * 0.55 + _c('sky5') * 0.45
    reg[moon] = _c('star')
    rim = moon & ~((np.sqrt((xx + 1) ** 2 + (yy - 1) ** 2) <= mr + 0.3) & (np.sqrt((xx - 3) ** 2 + (yy + 2) ** 2) > mr - 0.5))
    reg[rim] = _c('star2')
    # thin dusk clouds near the horizon
    for (cx, cy, cw) in [(30, 70, 70), (150, 100, 90), (240, 55, 50), (90, -60, 60), (200, 20, 40)]:
        for k in range(3):
            yy = cy + k - SKY_TOP
            x0 = cx - cw // 2 + k * 3
            x1 = cx + cw // 2 - k * 5
            colr = 'sky8' if cy > 80 else 'sky6' if cy > 0 else 'sky4'
            img[yy, max(0, x0):min(LW, x1)] = _c(colr) * 0.55 + img[yy, max(0, x0):min(LW, x1)] * 0.45
    return {'img': img, 'top': SKY_TOP, 'stars': stars}


# ---------------------------------------------------------------- far skyline
def build_far():
    h = 260
    img = np.zeros((h, LW, 4), np.uint8)
    r = np.random.default_rng(5)
    x = 0
    antennas = []
    while x < LW:
        bw = int(r.integers(14, 34))
        top = int(r.integers(70, 160))
        col = ['far0', 'far1', 'far2'][int(r.integers(0, 3))]
        bw = min(bw, LW - x)
        img[top:, x:x + bw, :3] = P[col]
        img[top:, x:x + bw, 3] = 255
        # roof details
        if r.random() < 0.35:
            ax = x + bw // 2
            ah = int(r.integers(6, 16))
            img[top - ah:top, ax, :3] = P[col]; img[top - ah:top, ax, 3] = 255
            antennas.append((ax, top - ah - 1))
        elif r.random() < 0.4:
            sw = bw // 2
            img[top - 4:top, x + 2:x + 2 + sw, :3] = P[col]; img[top - 4:top, x + 2:x + 2 + sw, 3] = 255
        # tiny lit windows
        for wy in range(top + 4, h, 5):
            for wx in range(x + 2, min(LW, x + bw - 2), 3):
                if r.random() < 0.18:
                    img[wy, wx, :3] = P['farwin' if r.random() < 0.7 else 'farwin2']
        x += bw + int(r.integers(-3, 3))
    return {'img': img, 'top': FAR_TOP, 'antennas': antennas}


# ---------------------------------------------------------------- near layer
def _rect(img, x, y, w, h, col, top=NEAR_TOP):
    y0 = y - top
    x0 = x + MARGIN
    ys, ye = max(0, y0), min(img.shape[0], y0 + h)
    xs, xe = max(0, x0), min(img.shape[1], x0 + w)
    if ys < ye and xs < xe:
        img[ys:ye, xs:xe, :3] = P[col]
        img[ys:ye, xs:xe, 3] = 255


def _px(img, x, y, col, top=NEAR_TOP):
    yy, xx = y - top, x + MARGIN
    if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
        img[yy, xx, :3] = P[col]; img[yy, xx, 3] = 255


def _spr(img, spr, x, y, top=NEAR_TOP):
    """Paste RGBA sprite into the RGBA layer (binary alpha)."""
    h, w = spr.shape[:2]
    yy, xx = y - top, x + MARGIN
    for j in range(h):
        if not (0 <= yy + j < img.shape[0]):
            continue
        for i in range(w):
            if spr[j, i, 3] > 127 and 0 <= xx + i < img.shape[1]:
                img[yy + j, xx + i] = spr[j, i]


def _blend_rect(img, x, y, w, h, col, a, top=NEAR_TOP, dither=None):
    y0 = y - top; x0 = x + MARGIN
    ys, ye = max(0, y0), min(img.shape[0], y0 + h)
    xs, xe = max(0, x0), min(img.shape[1], x0 + w)
    if ys >= ye or xs >= xe:
        return
    reg = img[ys:ye, xs:xe, :3].astype(np.float32)
    c = _c(col)
    if dither is not None:
        m = gfx.dither_mask(ye - ys, xe - xs, dither, xs, ys)
        reg[m] = reg[m] * (1 - a) + c * a
    else:
        reg[:] = reg * (1 - a) + c * a
    img[ys:ye, xs:xe, :3] = reg.astype(np.uint8)


def _window(img, x, y, w, h, lit, r):
    _rect(img, x - 1, y - 1, w + 2, h + 2, 'win_frame')
    if lit == 'warm':
        _rect(img, x, y, w, h, 'win_warm')
        _rect(img, x, y + h - 2, w, 2, 'win_warm2')
        if r.random() < 0.5:   # curtain
            _rect(img, x, y, max(1, w // 3), h, 'win_warm2')
    elif lit == 'tv':
        _rect(img, x, y, w, h, 'win_dim2')
        _rect(img, x + 1, y + h - 4, w - 2, 3, 'win_tv')
    else:
        _rect(img, x, y, w, h, 'win_dim')
        _px(img, x, y, 'win_dim2')
    # sill
    _rect(img, x - 1, y + h + 1, w + 2, 1, 'bd4')


def _ac_unit(img, x, y):
    _rect(img, x, y, 8, 6, 'curb1'); _rect(img, x, y, 8, 1, 'curb2')
    _rect(img, x + 1, y + 2, 4, 3, 'curb0'); _px(img, x + 6, y + 3, 'curb2')
    _rect(img, x, y + 6, 8, 1, 'bd0')


def _building(img, x0, x1, top, base_col, hi_col, lo_col, r, floor_h=22, win_w=8, win_h=10, bays=None):
    _rect(img, x0, top, x1 - x0, Y_SHOP_BASE - top, base_col)
    _rect(img, x0, top, x1 - x0, 2, hi_col)               # parapet light
    _rect(img, x0, top + 2, x1 - x0, 1, lo_col)
    _rect(img, x1 - 2, top, 2, Y_SHOP_BASE - top, lo_col)  # side shadow
    wins = []
    y = top + 8
    while y + win_h < 158:
        bx = x0 + 6
        while bx + win_w < x1 - 4:
            roll = r.random()
            lit = 'warm' if roll < 0.42 else 'tv' if roll < 0.52 else 'dark'
            _window(img, bx, y, win_w, win_h, lit, r)
            wins.append((bx, y, win_w, win_h, lit))
            if r.random() < 0.12:
                _ac_unit(img, bx + win_w + 2, y + win_h - 4)
            bx += win_w + 9
        _rect(img, x0, y + floor_h - 6, x1 - x0, 1, lo_col)   # floor line
        y += floor_h
    return wins


def _bike_icon(img, x, y, col, top=NEAR_TOP):
    """13x8 white bicycle pictogram (side view)."""
    rows = [
        ".....##......",
        "......#..##..",
        "..#######....",
        ".##..#..#.##.",
        "#..#.#...#..#",
        "#..#..#..#..#",
        "#..#.....#..#",
        ".##.......##.",
    ]
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            if ch == '#':
                _px(img, x + i, y + j, col, top)


def build_near(f12):
    top = NEAR_TOP
    h = 520 - top
    img = np.zeros((h, LW, 4), np.uint8)
    r = np.random.default_rng(3)
    info = {'windows': [], 'neon': []}

    # --- buildings (x in street coords, can extend into margins) ---
    info['windows'] += _building(img, -24, 96, 42, 'brick1', 'brick2', 'brick0', r)
    info['windows'] += _building(img, 96, 192, 10, 'bd2', 'bd4', 'bd1', r, floor_h=20, win_w=7, win_h=11)
    info['windows'] += _building(img, 192, 300, 58, 'bd1', 'bd3', 'bd0', r, floor_h=23)
    # balconies on building A
    for by in (86, 130):
        _rect(img, 4, by, 34, 2, 'curb1'); _rect(img, 4, by + 2, 34, 6, 'brick0')
        for k in range(4, 38, 3):
            _px(img, k, by + 3, 'curb1')
        # laundry
        for k, c in enumerate(['neon_cyan2', 'hood0', 'wht']):
            _rect(img, 8 + k * 9, by - 7, 5, 6, c)
            _rect(img, 8 + k * 9, by - 8, 5, 1, 'curb2')

    # --- shop fronts: 158..215 ---
    # sign band + awning bases
    # 1) convenience store  x -24..96
    _rect(img, -24, 158, 120, 12, 'grn2'); _rect(img, -24, 158, 120, 1, 'grn')
    _rect(img, -24, 169, 120, 1, 'bd0')
    sign = F.text_sprite(f12, '便利店', color='wht')
    _spr(img, sign, 14, 158)
    s24 = F.text_sprite(f12, '24H', color='yel')
    _spr(img, s24, 58, 158)
    _rect(img, -24, 170, 120, 45, 'shop_lit')
    # shelves inside
    for sy in (176, 186, 196):
        _rect(img, -20, sy, 52, 2, 'shop_in')
        for k in range(-20, 32, 4):
            _rect(img, k, sy - 4, 3, 4, ['neon_pink', 'grn', 'blue2', 'org', 'yel'][(k // 4) % 5])
    # door (glass) + frame
    _rect(img, 44, 172, 22, 43, 'win_frame'); _rect(img, 46, 174, 18, 41, 'shop_lit2')
    _rect(img, 54, 174, 1, 41, 'win_frame')
    _rect(img, 70, 176, 22, 30, 'win_frame'); _rect(img, 72, 178, 18, 26, 'shop_lit')
    _rect(img, 72, 192, 18, 2, 'shop_in')
    _rect(img, -24, 213, 120, 2, 'bd0')
    # 2) milk tea  x 96..192
    _rect(img, 96, 158, 96, 12, 'bd0'); _rect(img, 96, 169, 96, 1, 'ink')
    info['neon'].append(('tea', 104, 158))
    _rect(img, 96, 170, 96, 45, 'bd1')
    _rect(img, 100, 174, 56, 30, 'win_frame'); _rect(img, 102, 176, 52, 26, 'shop_in')
    _rect(img, 102, 176, 52, 3, 'shop_lit2')
    for k in range(106, 150, 9):    # cups on counter
        _rect(img, k, 192, 5, 7, 'shop_lit'); _rect(img, k, 192, 5, 1, 'neon_pink2')
        _rect(img, k + 2, 189, 1, 3, 'ink')
    _rect(img, 102, 199, 52, 3, 'brick2')
    _rect(img, 162, 174, 22, 41, 'win_frame'); _rect(img, 164, 176, 18, 39, 'shop_in')
    _rect(img, 96, 213, 96, 2, 'bd0')
    # 3) bike repair  x 192..300  (roll-up shutter half open)
    _rect(img, 192, 158, 108, 12, 'bd0'); _rect(img, 192, 169, 108, 1, 'ink')
    info['neon'].append(('fix', 200, 158))
    _rect(img, 192, 170, 108, 45, 'bd2')
    _rect(img, 198, 172, 74, 43, 'win_frame')
    _rect(img, 200, 174, 70, 18, 'curb1')                      # shutter (half)
    for k in range(175, 192, 2):
        _rect(img, 200, k, 70, 1, 'curb0')
    _rect(img, 200, 192, 70, 23, 'shop_in')                    # lit interior
    _rect(img, 200, 192, 70, 2, 'shop_lit2')
    for bx in (206, 232):                                       # bikes inside
        _bike_icon(img, bx, 198, 'bd0')
        _bike_icon(img, bx + 1, 198, 'bd1')
    _rect(img, 192, 213, 108, 2, 'bd0')

    # --- sidewalk 215..258 ---
    y0, y1 = Y_SIDEWALK
    _rect(img, -24, y0, LW, y1 - y0, 'sw2')
    rows_y = [215, 221, 228, 236, 245, 258]
    for i in range(len(rows_y) - 1):
        a, b = rows_y[i], rows_y[i + 1]
        _rect(img, -24, a, LW, 1, 'sw1')
        _rect(img, -24, a + 1, LW, 1, 'sw3')
        step = 14 + i * 2
        off = (i % 2) * step // 2
        for x in range(-24 + off, 300, step):
            _rect(img, x, a + 1, 1, b - a - 1, 'sw1')
    # tactile paving strip (yellow) y 228..236
    _rect(img, -24, 228, LW, 8, 'tact1')
    _rect(img, -24, 228, LW, 1, 'tact2')
    _rect(img, -24, 235, LW, 1, 'tact0')
    for x in range(-24, 300, 4):
        _rect(img, x, 230, 2, 1, 'tact0'); _rect(img, x + 2, 232, 2, 1, 'tact0')
    # light spill from shop windows onto sidewalk (warm, dithered)
    for (sx, sw) in [(-24, 120), (100, 56), (200, 70)]:
        _blend_rect(img, sx, 215, sw, 13, 'shop_lit2', 0.35, dither=0.5)

    # --- curb 258..263 ---
    _rect(img, -24, 258, LW, 2, 'curb3')
    _rect(img, -24, 260, LW, 3, 'curb1')
    for x in range(-24, 300, 18):
        _rect(img, x, 260, 1, 3, 'curb0')

    # --- bike lane 263..312 ---
    y0, y1 = Y_LANE
    lane = np.zeros((y1 - y0, LW, 3), np.float32)
    base = _c('lane2')
    lane[:] = base
    noise = np.random.default_rng(21).random((y1 - y0, LW))
    lane[noise < 0.10] = _c('lane1')
    lane[noise > 0.93] = _c('lane3')
    lane[:3] = _c('lane1'); lane[0] = _c('lane0')        # curb shadow
    lane[-2:] = _c('lane1')
    img[y0 - top:y1 - top, :, :3] = lane.astype(np.uint8)
    img[y0 - top:y1 - top, :, 3] = 255
    # painted bicycle symbol (oblique, squashed) + arrow
    sym = [
        "..........WWW.................",
        "...........WW......WWWWW......",
        "....WWWWWWWWWWWWWWWW..........",
        "..WW.....WW......W.WW..WWWW...",
        "WW.....WWWWW.....W..WWW....WW.",
        "WW....WW...WW...W..WW.......WW",
        "WW...WW.....WW.W...WW.......WW",
        "..WW.WW......WWW....WW....WW..",
        "....WWW.......W.......WWWW....",
    ]
    for (sx, sy) in [(22, 280), (226, 280)]:
        for j, row in enumerate(sym):
            for i, ch in enumerate(row):
                if ch == 'W':
                    _px(img, sx + i, sy + j, 'paint' if (i + j) % 7 else 'paint2')
    arrow = [
        "..............W.....",
        "..............WWW...",
        "WWWWWWWWWWWWWWWWWWW.",
        "WWWWWWWWWWWWWWWWWWWW",
        "WWWWWWWWWWWWWWWWWWW.",
        "..............WWW...",
        "..............W.....",
    ]
    for (sx, sy) in [(70, 283), (180, 296)]:
        for j, row in enumerate(arrow):
            for i, ch in enumerate(row):
                if ch == 'W':
                    _px(img, sx + i, sy + j, 'paint' if (i * 3 + j) % 9 else 'paint2')
    # --- solid white line ---
    _rect(img, -24, 312, LW, 2, 'paint')
    for x in range(-24, 300, 11):
        _px(img, x, 313, 'paint2')

    # --- motor lane 314..480 ---
    y0, y1 = Y_ROAD
    road = np.zeros((y1 - y0, LW, 3), np.float32)
    road[:] = _c('asp2')
    nz = np.random.default_rng(31).random((y1 - y0, LW))
    road[nz < 0.12] = _c('asp1')
    road[nz > 0.94] = _c('asp3')
    # subtle darkening toward bottom (vignette in palette steps)
    for yy in range(y1 - y0):
        if yy > 120:
            m = gfx.dither_mask(1, LW, min(1, (yy - 120) / 60), 0, yy)[0]
            road[yy, m] = _c('asp1')
    img[y0 - top:y1 - top, :, :3] = road.astype(np.uint8)
    img[y0 - top:y1 - top, :, 3] = 255
    _rect(img, -24, 314, LW, 2, 'asp1')                 # line shadow
    for x in range(-24, 300, 40):                         # dashed lane divider
        _rect(img, x, 404, 22, 2, 'paint2')
        _rect(img, x, 404, 22, 1, 'paint')
    # manhole cover
    mh = gfx.ellipse_mask(9, 4)
    hh, ww = mh.shape
    for j in range(hh):
        for i in range(ww):
            if mh[j, i]:
                _px(img, 38 + i, 352 + j, 'asp4' if (i + j) % 3 else 'asp1')
    # oil stain / patch
    _blend_rect(img, 196, 362, 30, 6, 'asp0', 0.5, dither=0.5)

    # --- street furniture: trees, lamp, sign (drawn on near layer, behind actors) ---
    for tx in (14, 252):
        _tree(img, tx, 256, r)
    _lamp(img, 92, 256)
    _bike_sign(img, f12, 196, 256)
    info['lamp_head'] = (114, 116)
    return {'img': img, 'top': NEAR_TOP, 'info': info}


def _tree(img, x, base, r):
    # tree pit
    _rect(img, x - 7, base - 2, 15, 3, 'sw0'); _rect(img, x - 6, base - 2, 13, 1, 'trunk0')
    # trunk
    for y in range(base - 64, base - 1):
        _rect(img, x - 1, y, 3, 1, 'trunk1'); _px(img, x + 1, y, 'trunk0')
    _px(img, x - 2, base - 30, 'trunk1'); _px(img, x + 2, base - 44, 'trunk1')
    # canopy: clustered blobs with highlight on upper-left, shadow lower-right
    rr = np.random.default_rng(x)
    blobs = [(0, -82, 15), (-11, -74, 11), (11, -73, 11), (-6, -92, 10), (7, -91, 10), (0, -66, 10)]
    for (bx, by, br) in blobs:
        m = gfx.circle_mask(br)
        d = m.shape[0]
        for j in range(d):
            for i in range(d):
                if not m[j, i]:
                    continue
                dx, dy = i - br, j - br
                shade = (dx + dy) / (2 * br)
                n = rr.random() * 0.25
                if shade + n < -0.45:
                    c = 'tr4'
                elif shade + n < -0.1:
                    c = 'tr3'
                elif shade + n < 0.35:
                    c = 'tr2'
                else:
                    c = 'tr1'
                _px(img, x + bx + dx, base + by + dy, c)
    # dark outline under canopy
    for i in range(-14, 15):
        _px(img, x + i, base - 56 + abs(i) // 5, 'tr0')


def _lamp(img, x, base):
    _rect(img, x - 2, base - 3, 5, 3, 'pole0')
    for y in range(base - 140, base - 3):
        _rect(img, x - 1, y, 2, 1, 'pole1'); _px(img, x, y, 'pole0')
    # arm over the lane
    for i in range(0, 24):
        yy = base - 140 - int(4 * np.sin(i / 23 * np.pi / 2))
        _rect(img, x + i, yy, 1, 2, 'pole1')
    # head
    _rect(img, x + 18, base - 146, 12, 3, 'pole0')
    _rect(img, x + 19, base - 143, 10, 2, 'lamp')
    _rect(img, x + 20, base - 141, 8, 1, 'lamp2')


def _bike_sign(img, f12, x, base):
    # pole
    for y in range(base - 70, base):
        _rect(img, x - 1, y, 2, 1, 'pole2'); _px(img, x, y, 'pole1')
    # blue circular sign with white bicycle
    cy = base - 78
    m = gfx.circle_mask(10)
    ring = m & ~np.pad(gfx.circle_mask(8), 2)
    d = m.shape[0]
    for j in range(d):
        for i in range(d):
            if m[j, i]:
                _px(img, x - 10 + i, cy - 10 + j, 'wht' if ring[j, i] else 'blue1')
    _bike_icon(img, x - 6, cy - 4, 'wht')
    # text plate
    t = F.text_sprite(f12, '非机动车道', color='wht')
    tw = t.shape[1]
    px0 = x - tw // 2 - 2
    _rect(img, px0, base - 66, tw + 4, 15, 'wht')
    _rect(img, px0 + 1, base - 65, tw + 2, 13, 'blue1')
    _spr(img, t, px0 + 2, base - 65)


class Scene:
    def __init__(self):
        self.f12 = F.Font(12)
        self.sky = build_sky()
        self.far = build_far()
        self.near = build_near(self.f12)
        bake_lamp_light(self.near)
        self.neon_tea = self._neon_sprite('奶茶', 'neon_pink', 'neon_pink2', cup=True)
        self.neon_fix = self._neon_sprite('修车', 'neon_cyan', 'neon_cyan2', cup=False)

    def _neon_sprite(self, text, c1, c2, cup):
        t = F.text_sprite(self.f12, text, color=c2, outline=c1, outline_diag=False)
        return t

    def draw(self, canvas, cam_x, cam_y, t, flicker=True):
        """Composite background layers onto canvas (float32 H x W x 3)."""
        ix = int(round(cam_x)) + MARGIN
        # sky
        sy = int(round(cam_y * SKY_FACTOR)) - self.sky['top']
        canvas[:] = self.sky['img'][sy:sy + H, ix:ix + W]
        # twinkling stars (only where visible)
        for (x, y, big, ph, sp) in self.sky['stars']:
            yy = y - self.sky['top'] - sy
            xx = x - ix
            if 0 <= yy < H and 0 <= xx < W:
                s = np.sin(t * sp * 2 + ph)
                if s > 0.85:
                    canvas[yy, xx] = P['white']
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if 0 <= xx + dx < W and 0 <= yy + dy < H:
                            canvas[yy + dy, xx + dx] = canvas[yy + dy, xx + dx] * 0.5 + P['star'] * 0.5
                elif s < -0.6:
                    canvas[yy, xx] = canvas[yy, xx] * 0.5 + P['sky2'] * 0.5
        # far skyline
        fy = int(round(cam_y * FAR_FACTOR))
        self._layer(canvas, self.far['img'], self.far['top'] - fy, ix)
        for (ax, ay) in self.far['antennas']:
            if (int(t * 1.3 + ax * 0.37) % 2) == 0:
                yy = ay + self.far['top'] - fy
                xx = ax - ix
                if 0 <= yy < H and 0 <= xx < W:
                    canvas[yy, xx] = P['red']
        # near
        ny = int(round(cam_y))
        self._layer(canvas, self.near['img'], self.near['top'] - ny, ix)
        # neon signs (flicker)
        on_tea = not (flicker and (int(t * 30) % 97) in (3, 4, 40))
        on_fix = not (flicker and (int(t * 30) % 71) in (10, 11, 12))
        for (kind, x, y) in self.near['info']['neon']:
            spr = self.neon_tea if kind == 'tea' else self.neon_fix
            if (kind == 'tea' and on_tea) or (kind == 'fix' and on_fix):
                gfx.blit_hard(canvas, spr, x - cam_x - 1, y - ny - 1)
            else:
                gfx.blit_hard(canvas, gfx.silhouette(spr, 'bd2'), x - cam_x - 1, y - ny - 1)
        return ny

    def _layer(self, canvas, img, top_screen, ix):
        """Blit RGBA layer whose row 0 sits at screen y=top_screen."""
        h = img.shape[0]
        y0 = max(0, top_screen); y1 = min(H, top_screen + h)
        if y0 >= y1:
            return
        src = img[y0 - top_screen:y1 - top_screen, ix:ix + W]
        a = src[..., 3:4] > 127
        dst = canvas[y0:y1]
        np.copyto(dst, src[..., :3].astype(np.float32), where=a)

    pass


def bake_lamp_light(near):
    """Warm light pool from the street lamp baked into the near layer (dithered)."""
    img = near['img']
    lx, ly = near['info']['lamp_head']
    ys = np.arange(Y_SIDEWALK[0], Y_LINE[1])
    xs = np.arange(-MARGIN, W + MARGIN)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    dx = (xx - lx) / 64.0
    dy = (yy - 284) / 36.0
    d = dx * dx + dy * dy
    a = np.clip(1 - d, 0, 1)
    thr = BAYER4[yy % 4, xx % 4]
    m = (thr < a * 0.75)
    reg = img[ys[0] - NEAR_TOP:ys[-1] + 1 - NEAR_TOP, :, :3].astype(np.float32)
    reg[m] = reg[m] * 0.78 + P['glow'].astype(np.float32) * 0.22
    img[ys[0] - NEAR_TOP:ys[-1] + 1 - NEAR_TOP, :, :3] = reg.astype(np.uint8)
