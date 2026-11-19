import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from PIL import Image
import gfx, sprites_obj
out = sys.argv[1]
sp = sprites_obj.build()
items = list(sp.items())
pad = 4
W = sum(s.shape[1] + pad for _, s in items) + pad
H = max(s.shape[0] for _, s in items) + 2 * pad
canvas = np.zeros((H, W, 3), np.float32); canvas[:] = gfx.P['lane2']; canvas[:H//2] = gfx.P['sw2']
x = pad
for n, s in items:
    gfx.blit(canvas, s, x, pad); x += s.shape[1] + pad
Image.fromarray(gfx.upscale(gfx.to_uint8(canvas), 6)).save(out); print(out)
