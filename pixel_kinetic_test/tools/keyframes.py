"""Export lossless key frames (1080x1920 PNG) + a labelled contact sheet."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gfx
from director import Director

import timeline as TL
I = TL.IMPACTS
KEYS = [  # defined relative to the timeline, so they work for both cuts (synth values in comments)
    ('01_title', 0.9), ('02_target_locked', TL.LOCK_T + 1.5), ('03_L1_card', TL.card_start(0) + 1.57),
    ('04_L1_impact', I[0] + 0.07), ('05_L1_no_effect', TL.result_time(0) + 0.3), ('06_L2_wobble', I[1] + 0.2),
    ('07_L3_slowmo', I[2] - 0.2), ('08_L3_impact_flash', I[2] + 0.07), ('09_L3_spin', I[2] + 0.45),
    ('10_L3_dizzy', TL.result_time(2) + 0.25), ('11_L4_strike', I[3] + 0.25),
    ('12_L4_phone_crack', TL.PHONE_INSERT[0] + 0.23), ('13_L5_warning', TL.WARNING_START + 1.35),
    ('14_L5_bell', TL.BELL_T + 0.2), ('15_L5_slowmo', I[4] - 0.15), ('16_L5_impact', I[4] + 0.1),
    ('17_L5_launch', I[4] + 0.5), ('18_test_terminated', TL.TERMINATED_T + 0.25), ('19_landing', TL.LAND_T + 0.05),
    ('20_conclusion', TL.END_CARD + 1.5),
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
