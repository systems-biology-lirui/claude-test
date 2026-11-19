"""Pedestrian sprites (3/4 view, facing right). 22 x 56 px.

Head down, phone in both hands at chest, big white over-ear headphones,
orange hoodie, blue jeans, white sneakers. Face lit cyan by the phone.
"""
from gfx import sprite

KEY = {
    'k': 'ink',
    'H': 'hair0', 'h': 'hair1', 'j': 'hair2',
    'W': 'hp0', 'w': 'hp1', 'v': 'hp2', 'r': 'hpr',
    'S': 'skin0', 's': 'skin1', 'z': 'skin2',
    'c': 'lit0', 'C': 'lit1',
    'L': 'hood0', 'O': 'hood1', 'o': 'hood2', 'd': 'hood3',
    'J': 'jean0', 'B': 'jean1', 'b': 'jean2', 'n': 'jean3',
    'T': 'shoe0', 't': 'shoe1', 'E': 'shoe2',
    'P': 'ph0', 'p': 'ph1', 'G': 'phg', 'g': 'phg2',
    'x': 'wht',
}

# idle frame A
IDLE_A = [
    "......................",  # 0
    "........kkkkkk........",  # 1
    "......kkWWWWWWkk......",  # 2
    ".....kWWkkkkkkWWk.....",  # 3
    "....kWkHHhhhhHHkWk....",  # 4
    "....kWkHhjjjhhHHkk....",  # 5
    "...kWkHHhjjhhHHHHk....",  # 6
    "...kWkHHHhhHHHHHHHk...",  # 7
    "..kvWWkHHHHHHHHHHHk...",  # 8
    "..kWWWWkHHHHHHHHHSk...",  # 9
    "..kWrWWkHHHHHHHSSSSk..",  # 10
    "..kWWWWkHHHHHSSSSkSk..",  # 11
    "..kvWWWkHHHHsSSSSSSk..",  # 12
    "...kvvkHHHHssSSSSSsk..",  # 13
    "....kkkHHHHkssSSSSsk..",  # 14
    "......kkHHkdkssSCk....",  # 15
    ".....kdookkookkkkk....",  # 16
    "....kdoOOOOOOLLk......",  # 17
    "....kdoOOOOLLLOOk.....",  # 18
    "...kdoOOOOLLxLOOOk....",  # 19
    "...kdoOOOLLOxOOOOk....",  # 20
    "...kdoOOOLOOxOOOOOk...",  # 21
    "...kdoOOOOOOxOOkkkkk..",  # 22
    "...kdooOOOOOOOkgGGGPk.",  # 23
    "...kdooOOOOOOkSgGGGPk.",  # 24
    "...kddoOOOOOkSSPPPPpk.",  # 25
    "...kddooOOOkSSSPPPPpk.",  # 26
    "...kddooOOOOkSSPPPPpk.",  # 27
    "...kddoOOOOOOkkkPPpk..",  # 28
    "...kddooOOOOOOOOkkk...",  # 29
    "...kddooOOOOOOOOOk....",  # 30
    "...kdddoooooooooOk....",  # 31
    "...kdddooooooooook....",  # 32
    "...kkdddddddddddok....",  # 33
    "....knnbbBBBBBBbbk....",  # 34
    "....knbbBBBBBBBBbk....",  # 35
    "....knbBBBJBBBBnbk....",  # 36
    "....knbBBJJBBBknbk....",  # 37
    "....knbBBJBBBknnbk....",  # 38
    "....knbBBJBBk.knbk....",  # 39
    "....knbBBJBBk.knbk....",  # 40
    "....knbBBJBBk.knbk....",  # 41
    "....knbBBJBBk.knbk....",  # 42
    "....knbBBJBBk.knbk....",  # 43
    "....knbBJJBBk.knbk....",  # 44
    "....knbBJBBBk.knbk....",  # 45
    "....knbBJBBBk.knbk....",  # 46
    "....knbBJBBBk.knbk....",  # 47
    "....knbBJBBBk.knbk....",  # 48
    "....knbBJBBBk.knbk....",  # 49
    "....knnbbbbbk.knnk....",  # 50
    "....kkkkkkkkk.kkkkk...",  # 51
    "....kTTTTTTTk.kttTTk..",  # 52
    "....kTTTTTTTTkkttTTTk.",  # 53
    "....kEEEEEEEEEkEEEEEk.",  # 54
    ".....kkkkkkkkk.kkkkk..",  # 55
]


# hands without phone (phone has flown away) - replaces rows 22..28
NOPHONE_ROWS = {
    22: "...kdoOOOOOOxOOk......",
    23: "...kdooOOOOOOOk.......",
    24: "...kdooOOOOOOkSk......",
    25: "...kddoOOOOOkSSSk.....",
    26: "...kddooOOOkSSSSk.....",
    27: "...kddooOOOOkSSk......",
    28: "...kddoOOOOOOkk.......",
}

# phone pixel box in sprite coords (for glow / crack overlays)
PHONE_BOX = (15, 22, 21, 29)   # x0, y0, x1, y1 (exclusive)
HAND_POINT = (17, 25)          # where a loose phone attaches when caught
HEAD_POINT = (11, 6)           # head centre (for stars / hit points)


def _bob(spr):
    """Upper body (rows 0..33) one pixel lower: nod to the beat."""
    out = spr.copy()
    out[1:35] = spr[0:34]
    out[0] = 0
    return out


def build():
    import numpy as np
    from palette import P
    idle = sprite(IDLE_A, KEY)
    rows = list(IDLE_A)
    for k, v in NOPHONE_ROWS.items():
        rows[k] = v
    nophone = sprite(rows, KEY)
    # glow variants (scrolling feed): swap bright/dim glow pixels
    def glow_variant(spr, phase):
        out = spr.copy()
        x0, y0, x1, y1 = PHONE_BOX
        reg = out[y0:y1, x0:x1]
        g1 = np.all(reg[..., :3] == P['phg'], -1) & (reg[..., 3] > 0)
        g2 = np.all(reg[..., :3] == P['phg2'], -1) & (reg[..., 3] > 0)
        if phase == 1:
            reg[g1, :3] = P['phg2']; reg[g2, :3] = P['phg']
        return out
    def cracked(spr):
        out = spr.copy()
        for (x, y) in [(16, 25), (17, 26), (18, 25), (19, 26), (18, 27)]:
            out[y, x, :3] = P['wht']
        return out
    d = {}
    for ph in (0, 1):
        a = glow_variant(idle, ph)
        d[f'idle_{ph}'] = a
        d[f'bob_{ph}'] = _bob(a)
        c = cracked(a)
        d[f'idle_c_{ph}'] = c
        d[f'bob_c_{ph}'] = _bob(c)
    d['nophone'] = nophone
    d['idle_a'] = idle
    return d
