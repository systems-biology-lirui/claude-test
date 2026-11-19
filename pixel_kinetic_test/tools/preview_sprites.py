import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from PIL import Image
import gfx
import sprites_ped

out = sys.argv[1] if len(sys.argv) > 1 else 'sprites_preview.png'
sp = sprites_ped.build()
items = list(sp.items())
pad = 6
W = sum(s.shape[1] + pad for _, s in items) + pad
H = max(s.shape[0] for _, s in items) + 2 * pad
canvas = np.zeros((H, W, 3), np.float32)
canvas[:] = gfx.P['lane2']
canvas[: H // 2] = gfx.P['sw2']
x = pad
for name, s in items:
    gfx.blit(canvas, s, x, pad)
    x += s.shape[1] + pad
img = gfx.upscale(gfx.to_uint8(canvas), 8)
Image.fromarray(img).save(out)
print('saved', out, img.shape)
