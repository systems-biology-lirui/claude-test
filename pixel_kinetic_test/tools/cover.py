"""Designed cover / thumbnail (1080x1920) built from the same engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from PIL import Image
import gfx, hud, fx
import copy_text as T
from physics import LEVELS, fmt_ke
from director import Director

out = sys.argv[1]
D = Director()
c = D.render(21.9, draw_hud=False)
# darken the sky band for the title
for y in range(0, 200):
    a = 0.82 * (1 - max(0, (y - 150) / 50))
    gfx.fill_rect(c, 0, y, 270, 1, 'ink', a)
hud.hazard_bar(c, 22, 4, 0, ('yel', 'ink'))
hud.text(c, 'PIXEL TRAFFIC LAB', 135, 30, 10, align='center', color='gry')
for j, ch in enumerate(T.TITLE_CN_1):
    spr = hud.tx().get(ch, 12, scale=3, bold=True, gradient=['wht', 'wht', 'neon_cyan2'], outline='ink',
                       shadow='blue0', shadow_off=(2, 2))
    gfx.blit_hard(c, spr, 135 - 90 + j * 36 - 2, 50)
for j, ch in enumerate(T.TITLE_CN_2):
    spr = hud.tx().get(ch, 12, scale=2, bold=True, gradient=['yel3', 'yel', 'org'], outline='ink',
                       shadow='red2', shadow_off=(2, 2))
    gfx.blit_hard(c, spr, 135 - 72 + j * 24 - 2, 96)
hud.text(c, T.TITLE_EN, 135, 128, 10, align='center', color='wht')
# energy span tag
a = fmt_ke(LEVELS[0]['ke']) + ' J'
b = fmt_ke(LEVELS[4]['ke']) + ' J'
line = f'{a}  →  {b}'
wv = hud.tx().width(line, 12, scale=2, bold=True, latin='monospaced')
gfx.fill_rect(c, 135 - wv // 2 - 8, 146, wv + 16, 34, 'ink', 0.9)
hud.text(c, line, 135, 150, 12, align='center', scale=2, bold=True, color='yel', latin='monospaced', outline='ink')
# subject + incoming e-bike callouts
hud.bubble(c, T.BELL, int(D.ebike_x(21.9)[0]) + 19, 255, 1.0)
hud.tag(c, T.ANC, 154, 242)
hud.stamp(c, '5 LEVELS', 'KINETIC ENERGY TEST', 1.0, color='red', cx=135, cy=360)
Image.fromarray(gfx.upscale(gfx.to_uint8(c), 4)).save(out, optimize=True)
print('cover ->', out)
