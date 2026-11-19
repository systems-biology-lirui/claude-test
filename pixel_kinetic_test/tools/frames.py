"""Render selected times to a contact sheet for review.
usage: python3 tools/frames.py out.png t1 t2 t3 ... [--scale 2] [--cols 4]"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from PIL import Image, ImageDraw
import gfx
from director import Director

args = sys.argv[1:]
out = args.pop(0)
scale = 1
cols = 4
if '--scale' in args:
    i = args.index('--scale'); scale = int(args[i + 1]); del args[i:i + 2]
if '--cols' in args:
    i = args.index('--cols'); cols = int(args[i + 1]); del args[i:i + 2]
times = [float(a) for a in args]
D = Director()
imgs = []
t0 = time.time()
for t in times:
    c = D.render(t)
    imgs.append(gfx.upscale(gfx.to_uint8(c), scale))
print(f'{len(times)} frames in {time.time()-t0:.2f}s')
h, w = imgs[0].shape[:2]
rows = (len(imgs) + cols - 1) // cols
sheet = Image.new('RGB', (cols * (w + 6) + 6, rows * (h + 22) + 6), (30, 30, 30))
d = ImageDraw.Draw(sheet)
for k, (t, im) in enumerate(zip(times, imgs)):
    r, c = divmod(k, cols)
    x, y = 6 + c * (w + 6), 6 + r * (h + 22)
    sheet.paste(Image.fromarray(im), (x, y + 16))
    d.text((x, y + 2), f't={t:.2f}s  f={int(round(t*30))}', fill=(255, 220, 80))
sheet.save(out)
print('saved', out)
