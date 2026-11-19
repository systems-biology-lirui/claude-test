"""Pixel font renderer built directly on Ark Pixel Font glyph bitmaps (SIL OFL 1.1).

Glyph sources: TakWolf/ark-pixel-font, assets/glyphs/{12,10}/cmap/... one PNG per code point.
All glyphs are normalised to a common line box so CJK, Latin and symbols share a baseline:
  12px: line height 12, baseline at row 10
  10px: line height 10, baseline at row 8 (Latin only)
"""
import os
import re
import numpy as np
from PIL import Image
from palette import P

_BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'ark_pixel_glyphs')
ROOT = os.environ.get('ARK_GLYPHS') or (_BUNDLED if os.path.isdir(_BUNDLED) else '/home/user/_fontsrc/ark/assets/glyphs')
_NAME_RE = re.compile(r'^([0-9A-F]{4,6})(?: ([a-z_,]+))?\.png$')

# custom glyphs (private use area) drawn by hand, 12px line box
CUSTOM12 = {
    '': [  # check mark
        "..........",
        "..........",
        ".........#",
        "........##",
        ".......##.",
        "#.....##..",
        "##...##...",
        ".##.##....",
        "..###.....",
        "...#......",
        "..........",
        "..........",
    ],
    '': [  # warning triangle
        ".....#.....",
        "....###....",
        "....#.#....",
        "...##.##...",
        "...#.#.#...",
        "..##.#.##..",
        "..#..#..#..",
        ".##.....##.",
        ".#...#...#.",
        "###########",
        "...........",
        "...........",
    ],
    '': [  # filled star
        ".....#.....",
        ".....#.....",
        "....###....",
        "###########",
        ".#########.",
        "..#######..",
        "...#####...",
        "..###.###..",
        "..##...##..",
        ".#.......#.",
        "...........",
        "...........",
    ],
}


class Font:
    def __init__(self, size=12):
        self.size = size
        self.line_h = 12 if size == 12 else 10
        self.baseline = 10 if size == 12 else 8
        self.index = {}   # (category, cp) -> {variant: path}
        base = os.path.join(ROOT, str(size), 'cmap')
        for cat in ('common', 'proportional', 'monospaced'):
            d = os.path.join(base, cat)
            if not os.path.isdir(d):
                continue
            for dirpath, _, files in os.walk(d):
                for f in files:
                    m = _NAME_RE.match(f)
                    if not m:
                        continue
                    cp = int(m.group(1), 16)
                    variants = (m.group(2) or 'default').split(',')
                    slot = self.index.setdefault((cat, cp), {})
                    for v in variants:
                        slot[v] = os.path.join(dirpath, f)
        self.cache = {}
        self.custom = {}
        if size == 12:
            for ch, rows in CUSTOM12.items():
                m = np.array([[c == '#' for c in r] for r in rows], bool)
                self.custom[ch] = m

    def _path(self, cp, latin):
        order = ['common', latin, 'monospaced' if latin == 'proportional' else 'proportional']
        for cat in order:
            slot = self.index.get((cat, cp))
            if slot:
                for v in ('zh_hans', 'zh_cn', 'default'):
                    if v in slot:
                        return slot[v]
                return next(iter(slot.values()))
        return None

    def glyph(self, ch, latin='proportional'):
        key = (ch, latin)
        if key in self.cache:
            return self.cache[key]
        if ch in self.custom:
            m = self.custom[ch]
            self.cache[key] = m
            return m
        p = self._path(ord(ch), latin)
        if p is None:
            raise KeyError(f'glyph missing for {ch!r} (U+{ord(ch):04X}) in {self.size}px')
        a = np.array(Image.open(p).convert('RGBA'))[..., 3] > 127
        h = a.shape[0]
        if self.size == 12:
            if h == 18:
                a = a[4:16]
            elif h != 12:
                raise ValueError(f'unexpected glyph height {h} for {ch!r}')
        else:
            if h == 16:
                a = a[4:14]
            elif h == 10:
                pass
            elif h == 12:   # fall back: 12px CJK in 10px line -> keep, caller beware
                a = a[1:11]
            else:
                raise ValueError(f'unexpected glyph height {h} for {ch!r}')
        self.cache[key] = a
        return a

    def check(self, s, latin='proportional'):
        missing = []
        for ch in s:
            if ch == '\n':
                continue
            try:
                self.glyph(ch, latin)
            except KeyError:
                missing.append(ch)
        return missing

    def mask(self, s, latin='proportional', spacing=0, bold=False):
        gl = [self.glyph(ch, latin) for ch in s]
        w = sum(g.shape[1] for g in gl) + spacing * max(0, len(gl) - 1) + (1 if bold else 0)
        m = np.zeros((self.line_h, max(w, 1)), bool)
        x = 0
        for g in gl:
            m[:, x:x + g.shape[1]] |= g
            x += g.shape[1] + spacing
        if bold:
            m[:, 1:] |= m[:, :-1].copy()
        return m

    def width(self, s, latin='proportional', spacing=0, bold=False, scale=1):
        return self.mask(s, latin, spacing, bold).shape[1] * scale


def _col(c):
    return np.asarray(P[c] if isinstance(c, str) else c, np.uint8)


def render(font, s, color='wht', latin='proportional', spacing=0, bold=False, scale=1,
           outline=None, shadow=None, shadow_off=(1, 1), gradient=None, trim=True,
           outline_diag=True):
    """Render a single line of text to an RGBA sprite.

    gradient: list of colour names applied top->bottom across the glyph rows (pre-scale).
    outline/shadow are 1 low-res pixel regardless of scale.
    """
    m = font.mask(s, latin, spacing, bold)
    if trim:
        # trim fully empty top/bottom rows only for big display text? keep line box for layout
        pass
    h, w = m.shape
    fill = np.zeros((h, w, 3), np.uint8)
    if gradient:
        rows = np.where(m.any(1))[0]
        top, bot = (rows.min(), rows.max()) if len(rows) else (0, h - 1)
        for y in range(h):
            t = 0 if bot == top else (y - top) / (bot - top)
            i = int(np.clip(round(t * (len(gradient) - 1)), 0, len(gradient) - 1))
            fill[y] = _col(gradient[i])
    else:
        fill[:] = _col(color)
    if scale > 1:
        m = np.repeat(np.repeat(m, scale, 0), scale, 1)
        fill = np.repeat(np.repeat(fill, scale, 0), scale, 1)
        h, w = m.shape
    pad = 1 + (max(shadow_off) if shadow is not None else 0)
    H, W = h + 2 * pad, w + 2 * pad
    out = np.zeros((H, W, 4), np.uint8)
    core = np.zeros((H, W), bool)
    core[pad:pad + h, pad:pad + w] = m
    if outline is not None:
        g = core.copy()
        g[1:, :] |= core[:-1, :]; g[:-1, :] |= core[1:, :]; g[:, 1:] |= core[:, :-1]; g[:, :-1] |= core[:, 1:]
        if outline_diag:
            g[1:, 1:] |= core[:-1, :-1]; g[1:, :-1] |= core[:-1, 1:]; g[:-1, 1:] |= core[1:, :-1]; g[:-1, :-1] |= core[1:, 1:]
        body = g
    else:
        body = core
    if shadow is not None:
        dx, dy = shadow_off
        sh = np.zeros_like(body)
        sh[dy:, dx:] = body[:H - dy, :W - dx]
        sh &= ~body
        out[sh, :3] = _col(shadow); out[sh, 3] = 255
    if outline is not None:
        ring = body & ~core
        out[ring, :3] = _col(outline); out[ring, 3] = 255
    fc = np.zeros((H, W, 3), np.uint8)
    fc[pad:pad + h, pad:pad + w] = fill
    out[core, :3] = fc[core]; out[core, 3] = 255
    return out, pad


def text_sprite(font, s, **kw):
    spr, _ = render(font, s, **kw)
    return spr
