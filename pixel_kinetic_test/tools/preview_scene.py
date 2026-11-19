import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from PIL import Image
import gfx, scene, sprites_ped
out = sys.argv[1]
cam_y = float(sys.argv[2]) if len(sys.argv) > 2 else 0
t0 = time.time()
S = scene.Scene()
t1 = time.time()
canvas = np.zeros((480, 270, 3), np.float32)
S.draw(canvas, 0, cam_y, 0.0)
t2 = time.time()
ped = sprites_ped.build()['idle_a']
gfx.blit(canvas, ped, 140 - 11, scene.FEET_Y - ped.shape[0] + 1 - cam_y)
img = gfx.upscale(gfx.to_uint8(canvas), 2)
Image.fromarray(img).save(out)
print(f'build {t1-t0:.2f}s draw {t2-t1:.3f}s ->', out)
