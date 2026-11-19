"""Test objects, e-bike + rider, phone, FX sprites."""
import numpy as np
from gfx import sprite, flip_h

BALL_KEY = {
    'k': 'ink', 'W': 'pp0', 'w': 'pp1', 'v': 'pp2',
    'T': 'tn0', 't': 'tn1', 'u': 'tn2', 'x': 'wht',
    'S': 'sc0', 's': 'sc1', 'B': 'sc2',
    'O': 'bw0', 'o': 'bw1', 'b': 'bw2', 'n': 'bw3', 'h': 'ink',
}

PINGPONG = [
    ".kk.",
    "kWWk",
    "kWvk",
    ".kk.",
]
PINGPONG_BIG = [  # for HUD card (12x12)
    "....kkkk....",
    "..kkWWWWkk..",
    ".kWWWWWWwwk.",
    ".kWxxWWWWwk.",
    "kWWxWWWWWwvk",
    "kWWWWWWWWwvk",
    "kWWWWWWWwwvk",
    "kWWWWWWwwvvk",
    ".kWWWWwwvvk.",
    ".kwwwwwvvvk.",
    "..kkvvvvkk..",
    "....kkkk....",
]
TENNIS = [
    "..kkk..",
    ".kTTxk.",
    "kTxTTtk",
    "kTTxxtk",
    "kTTTTxk",
    ".kttuk.",
    "..kkk..",
]
TENNIS_BIG = [
    "....kkkk....",
    "..kkTTTTkk..",
    ".kTTTxTTTtk.",
    ".kTxxTTTTtk.",
    "kTTxTTTTTTtk",
    "kTxTTTTTTxtk",
    "kTxTTTTTxTuk",
    "kTTxTTTxTTuk",
    ".kTTxxxTTtk.",
    ".kttTTTttuk.",
    "..kkttuukk..",
    "....kkkk....",
]
SOCCER = [
    "...kkkk...",
    ".kkSBBSkk.",
    ".kSSBBSSk.",
    "kBSSSSSSBk",
    "kBBSSSSBBk",
    "kSSSBBSSsk",
    "kSSBBBBSsk",
    ".kSSBBSssk",
    ".kksSSsskk",
    "...kkkk...",
]
SOCCER_BIG = [
    ".....kkkkk.....",
    "...kkSBBBSkk...",
    "..kSSSBBBSSSk..",
    ".kBSSSSBSSSSBk.",
    ".kBBSSSSSSSBBk.",
    "kSBSSSSSSSSSBsk",
    "kSSSSSBBBSSSSsk",
    "kSSSSBBBBBSSSsk",
    "kSBSSSBBBSSSBsk",
    "kBBSSSSSSSSSBBk",
    ".kBSSSSSSSSsBk.",
    ".kSSSSBBBSSssk.",
    "..kssSBBBSsskk.",
    "...kkssssskk...",
    ".....kkkkk.....",
]
BOWLING = [
    "....kkkk....",
    "..kkOOookk..",
    ".kOOOoooobk.",
    ".kOOoooohbk.",
    "kOOooohhoobk",
    "kOoooooohobk",
    "kooooooooobk",
    "koooooooobbk",
    ".kooooobbbk.",
    ".kbbbbbbbnk.",
    "..kknnnnkk..",
    "....kkkk....",
]
BOWLING_BIG = [
    ".....kkkkk.....",
    "...kkOOooookk..",
    "..kOOOOoooooBk.",
    ".kOOOooooohhbbk",
    ".kOOooooohhhobk",
    "kOOoooooooooobk",
    "kOooooohhooobbk",
    "koooooohhooobbk",
    "kooooooooooobbk",
    "koooooooooobbbk",
    ".koooooooobbbnk",
    ".kbooooobbbbnk.",
    "..kbbbbbbbnnk..",
    "...kknnnnnkk...",
    ".....kkkkk.....",
]

# ---------------------------------------------------------------- e-bike + rider (side view, facing right) 48 x 44
BIKE_KEY = {
    'k': 'ink', 'R': 'helm1', 'q': 'helm0', 'r': 'helm2', 'V': 'visor', 'x': 'wht',
    'S': 'skin0', 's': 'skin1',
    'A': 'jk0', 'B': 'jk1', 'b': 'jk2', 'P': 'pant0', 'p': 'pant1',
    'W': 'bk0', 'w': 'bk1', 'v': 'bk2', 'g': 'bk3', 't': 'tyre', 'm': 'pole2',
    'Y': 'hl', 'y': 'hl2', 'G': 'gry', 'E': 'shoe2', 'e': 'shoe1',
}
EBIKE = [
    "...............kkkkkk...........................",  # 0
    ".............kkqqRRRRkk.........................",  # 1
    "............kqqRRRRRRRk.........................",  # 2
    "...........kqRRRRRRRRRRk........................",  # 3
    "...........kRRRRRRRRRkkkk.......................",  # 4
    "...........krRRRRRRRkVVVVk......................",  # 5
    "...........krrRRRRRkVVxVVk......................",  # 6
    "...........kkrrrrrrkkSSSSk......................",  # 7
    "............kkkkkkkSSSSSsk......................",  # 8
    "..............kbbkkSSSSsk.......................",  # 9
    ".............kbBBBBkkkkk........................",  # 10
    "............kbBBBBBBBBkk........................",  # 11
    "...........kbBBBAABBBBBBkk......................",  # 12
    "...........kbBBAAABBBBBBBBkk....................",  # 13
    "...........kbBBAABBkkBBBBBBBkk..................",  # 14
    "...........kbBBABBBk.kkkkBBBSSkk................",  # 15
    "...........kbBBBBBBk.....kkkSSSSk...............",  # 16
    "...........kbbBBBBbk........kkkkgggk............",  # 17
    "...........kbbbBBbbk...........kgk..............",  # 18
    "..........kkpPPPPPPPkkkkk......kgk..............",  # 19
    ".........kpPPPPPPPPPPPPPPk.....kgkk.............",  # 20
    "........kppPPPPPPPPPPPPPPPk...kWWWWk............",  # 21
    "...kkkkkkkkkkkkkkpppPPPPPPk..kWWWWWWkkk.........",  # 22
    "...kvvvvvvvvvvvvkk.kppPPPPk.kWWWWWWkYYk.........",  # 23
    "...kgggggggggwWWWk..kkpPPPk.kWWWWWkYYYYk........",  # 24
    "....kkkkkkkkkWWWWWk..kpPPk..kwWWWWkYYyk.........",  # 25
    "............kWWWWWWkkkpPPk.kwwWWWWkkkk..........",  # 26
    "...........kwWWWWWWWWkEEEEkkwwwWWWk.............",  # 27
    "...........kwwwWWWWWWkeEEEEEkwwWWk..............",  # 28
    "..kkkkk....kvvwwwwwwwwkkkkkkkvwwwk....kkkkk.....",  # 29
    ".kttttttk...kkvvvvvvvvvvvvvvvvvvk...kttttttk....",  # 30
    "kttkkkkttk....kkkkkkkkkkkkkkkkkk...kttkkkkttk...",  # 31
    "ttk.mm.kttk......................kttk.mm.kttk...",  # 32
    "tk.m..m.ktk......................ktk.m..m.ktk...",  # 33
    "tkm.GG.mktk......................ktkm.GG.mktk...",  # 34
    "tkm.GG.mktk......................ktkm.GG.mktk...",  # 35
    "tk.m..m.ktk......................ktk.m..m.ktk...",  # 36
    "ttk.mm.kttk......................kttk.mm.kttk...",  # 37
    "kttkkkkttk........................kttkkkkttk....",  # 38
    ".kttttttk..........................kttttttk.....",  # 39
    "..kkkkkk............................kkkkkk......",  # 40
]

# ---------------------------------------------------------------- background car (side view, facing right) 52 x 20
CAR_KEY = {'k': 'ink', 'B': 'blue1', 'b': 'blue0', 'L': 'blue2', 'W': 'win_tv', 'w': 'blue3',
           't': 'tyre', 'm': 'pole2', 'Y': 'hl', 'R': 'red', 'g': 'gry2', 'G': 'gry'}
CAR = [
    "..............kkkkkkkkkkkkkkkkk.....................",
    "............kkLLLLLLLLLLLLLLLLLkk...................",
    "..........kkLLkwwwwwwwkkwwwwwwwkLkk.................",
    ".........kLLLkwWWWWWWWkkwWWWWWWWkLLkk...............",
    "........kLLLkWWWWWWWWWkkWWWWWWWWWkLLLkkk............",
    "....kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkLLkkkkkkkk....",
    "..kkBBBBBBBBBBBBBBBBBBBBkBBBBBBBBBBBBBBBBBBBBBBBkk..",
    ".kRBBLLLLLLLLLLLLLLLLLLLkLLLLLLLLLLLLLLLLLLLLLBBYYk.",
    ".kRBBBBBBBBBBBBBBBBBBBBBkBBBBBBBBBBBBBBBBBBBBBBBYYk.",
    ".kbBBBBBBBBBBBBBBBBBBBBBkBBBBBBBBBBBBBBBBBBBBBBBBbk.",
    ".kbbBBBBBkkkkkkBBBBBBBBBkBBBBBBBBBBBBBkkkkkkBBBBbbk.",
    ".kbbbbbbkttttttkbbbbbbbbbbbbbbbbbbbbbkttttttkbbbbbk.",
    "..kkkkkkttkkkkttkkkkkkkkkkkkkkkkkkkkkttkkkkttkkkkk..",
    "......kttkmmmkttk...................kttkmmmkttk.....",
    "......kttkmGmkttk...................kttkmGmkttk.....",
    "......kttkmmmkttk...................kttkmmmkttk.....",
    ".......kttkkkttk.....................kttkkkttk......",
    "........kttttkk.......................kttttkk.......",
    ".........kkkk..........................kkkk.........",
]

# ---------------------------------------------------------------- phone (loose, flying) 6 x 9
PHONE_KEY = {'k': 'ink', 'P': 'ph0', 'p': 'ph1', 'G': 'phg', 'g': 'phg2', 'x': 'wht', 'r': 'red'}
PHONE = [
    ".kkkk.",
    "kGGGGk",
    "kGgGGk",
    "kGGgGk",
    "kGGGgk",
    "kgGGGk",
    "kGGGGk",
    "kpPPpk",
    ".kkkk.",
]
PHONE_CRACKED = [
    ".kkkk.",
    "kGxGGk",
    "kGgxGk",
    "kxGxxk",
    "kGxGgk",
    "kxGxGk",
    "kGGGxk",
    "kpPPpk",
    ".kkkk.",
]
# big phone for the close-up insert (crack appears on it) 24 x 40
PHONE_XL_KEY = {'k': 'ink', 'P': 'ph0', 'p': 'ph1', 'G': 'phg', 'g': 'phg2', 'x': 'wht', 'c': 'cyn2', 'd': 'hud_bg2'}


def phone_xl(cracked=False):
    w, h = 26, 44
    a = [['.' for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            edge = x in (0, w - 1) or y in (0, h - 1)
            corner = (x in (0, w - 1)) and (y in (0, h - 1))
            if corner:
                continue
            if edge:
                a[y][x] = 'k'
            elif x in (1, w - 2) or y in (1, 2, h - 2, h - 3):
                a[y][x] = 'p' if y in (1, h - 2) else 'P'
            else:
                a[y][x] = 'G' if (x + y) % 9 else 'g'
    # fake UI on screen: video feed rows
    for y in range(6, 38, 8):
        for x in range(4, w - 4):
            a[y][x] = 'c'
            a[y + 1][x] = 'g'
    for y in range(8, 12):
        for x in range(4, 9):
            a[y][x] = 'd'
    rows = [''.join(r) for r in a]
    spr = sprite(rows, PHONE_XL_KEY)
    if cracked:
        # deterministic crack web from an impact point
        rng = np.random.default_rng(4)
        cx, cy = 15, 17
        pts = [(cx, cy)]
        for ang in np.linspace(0, 2 * np.pi, 9, endpoint=False) + 0.3:
            x, y = float(cx), float(cy)
            L = rng.integers(10, 24)
            a0 = ang
            for step in range(int(L)):
                a0 += rng.normal(0, 0.25)
                x += np.cos(a0); y += np.sin(a0)
                xi, yi = int(round(x)), int(round(y))
                if 2 <= xi < w - 2 and 3 <= yi < h - 3:
                    spr[yi, xi, :3] = (246, 245, 251)
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]:
            spr[cy + dy, cx + dx, :3] = (255, 255, 255)
    return spr


# ---------------------------------------------------------------- FX
STAR_SMALL = [
    "..y..",
    ".yYy.",
    "yYYYy",
    ".yYy.",
    "..y..",
]
STAR_KEY = {'y': 'spark2', 'Y': 'spark'}
NOTE = [
    "..kk",
    "..kW",
    "..kW",
    "..kW",
    "kkkW",
    "kWWk",
    "kWWk",
    ".kk.",
]
NOTE2 = [
    ".kkkkk",
    ".kWWWW",
    ".kW..W",
    ".kW..W",
    "kkW.kk",
    "WWkkWW",
    "WWkkWW",
    ".kk.kk",
]
NOTE_KEY = {'k': 'ink', 'W': 'neon_cyan2'}


def build():
    d = {}
    d['pingpong'] = sprite(PINGPONG, BALL_KEY)
    d['pingpong_big'] = sprite(PINGPONG_BIG, BALL_KEY)
    d['tennis'] = sprite(TENNIS, BALL_KEY)
    d['tennis_big'] = sprite(TENNIS_BIG, BALL_KEY)
    d['soccer'] = sprite(SOCCER, BALL_KEY)
    d['soccer_big'] = sprite(SOCCER_BIG, BALL_KEY)
    d['bowling'] = sprite(BOWLING, BALL_KEY)
    d['bowling_big'] = sprite(BOWLING_BIG, BALL_KEY)
    rows = list(EBIKE)
    torso = [
        "..............kbbkkSSSSsk",
        ".............kbBBBBkkkkk",
        "............kbBBBBBBBBkk",
        "...........kbBBBAABBBBBBkk",
        "...........kbBBAAABBBBBBBBk",
        "...........kbBBAABBBkkBBBBBk",
        "...........kbBBABBBBk.kBBBBBk",
        "...........kbBBABBBBk..kBBBBBk",
        "...........kbBBABBBBk...kBBBBBk",
        "...........kbBBBBBBBk....kBBBBBk",
        "...........kbBBBBBBBk.....kBBBBBk",
        "...........kbBBBBBBBk......kkBBSSkk",
        "...........kbbBBBBbbk.......kkSSSSk",
        "...........kbbBBBBbbk........kkkgggk",
        "...........kbbbBBbbk...........kgk",
    ]
    torso = [r.ljust(48, '.') for r in torso]
    rows = rows[:9] + torso + rows[19:]
    sh = len(rows) - len(EBIKE)
    d['ebike'] = sprite(rows, BIKE_KEY)
    d['ebike_meta'] = {'wheels': [(5, 34 + sh), (38, 34 + sh)], 'front_x': 45, 'ground_row': 40 + sh,
                       'headlight': (38, 24 + sh), 'bell': (31, 17 + sh), 'rider_head': (19, 5)}
    d['car'] = sprite(CAR, CAR_KEY)
    d['phone'] = sprite(PHONE, PHONE_KEY)
    d['phone_cracked'] = sprite(PHONE_CRACKED, PHONE_KEY)
    d['phone_xl'] = phone_xl(False)
    d['phone_xl_cracked'] = phone_xl(True)
    d['star'] = sprite(STAR_SMALL, STAR_KEY)
    d['note'] = sprite(NOTE, NOTE_KEY)
    d['note2'] = sprite(NOTE2, NOTE_KEY)
    return d


def draw_wheel_spokes(spr, cx, cy, angle):
    """Overdraw rotating spokes inside a wheel of the e-bike sprite (in place)."""
    import math
    from palette import P
    h, w = spr.shape[:2]
    for y in range(cy - 3, cy + 4):
        for x in range(cx - 3, cx + 4):
            if 0 <= x < w and 0 <= y < h and (x - cx) ** 2 + (y - cy) ** 2 <= 10:
                spr[y, x, :3] = P['tyre']; spr[y, x, 3] = 255
    for k in range(3):
        a = angle + k * 2 * math.pi / 3
        for r in (1, 2, 3):
            x = int(round(cx + math.cos(a) * r)); y = int(round(cy + math.sin(a) * r))
            if 0 <= x < w and 0 <= y < h:
                spr[y, x, :3] = P['pole2']
    spr[cy, cx, :3] = P['gry']
