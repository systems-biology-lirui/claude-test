"""Export lossless key frames (1080x1920 PNG) + a labelled contact sheet."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gfx
from director import Director

KEYS = [
    ('01_title', 0.9), ('02_target_locked', 3.5), ('03_L1_card', 5.72), ('04_L1_impact', 6.07),
    ('05_L1_no_effect', 7.2), ('06_L2_wobble', 10.2), ('07_L3_slowmo', 13.8), ('08_L3_impact_flash', 14.07),
    ('09_L3_spin', 14.45), ('10_L3_dizzy', 15.4), ('11_L4_strike', 17.85), ('12_L4_phone_crack', 18.45),
    ('13_L5_warning', 20.55), ('14_L5_bell', 21.9), ('15_L5_slowmo', 22.25), ('16_L5_impact', 22.5),
    ('17_L5_launch', 22.9), ('18_test_terminated', 23.8), ('19_landing', 25.25), ('20_conclusion', 27.5),
]

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
D = Director()
small = []
for name, t in KEYS:
    img = gfx.to_uint8(D.render(t))
    Image.fromarray(gfx.upscale(img, 4)).save(os.path.join(out, f'{name}.png'), optimize=True)
    small.append((name, t, img))
cols = 5
rows = (len(small) + cols - 1) // cols
w, h = 270, 480
sheet = Image.new('RGB', (cols * (w + 8) + 8, rows * (h + 26) + 8), (18, 16, 30))
d = ImageDraw.Draw(sheet)
try:
    lab_font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 14)
except Exception:
    lab_font = None
for k, (name, t, img) in enumerate(small):
    r, c = divmod(k, cols)
    x, y = 8 + c * (w + 8), 8 + r * (h + 26)
    d.text((x, y + 2), f'{name}  t={t:.2f}s', fill=(255, 210, 63), font=lab_font)
    sheet.paste(Image.fromarray(img), (x, y + 22))
sheet.save(os.path.join(out, 'contact_sheet.png'), optimize=True)
print('keyframes:', len(KEYS), '->', out)
