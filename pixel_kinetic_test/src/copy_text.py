"""All on-screen copy in one place (so it can be glyph-checked and proof-read)."""
from physics import LEVELS, fmt_ke, fmt_num, fmt_ratio

TITLE_CN_1 = '非机动车道'
TITLE_CN_2 = '行人动能测试'
TITLE_EN = 'PEDESTRIAN KINETIC ENERGY TEST'

SUBJECT_HEAD = '测试对象'
SUBJECT_HEAD_EN = 'TEST SUBJECT'
SUBJECT_LINES = ['位置：非机动车道', '耳机：ON  手机：ON', '环境感知：0%']
TARGET_LOCKED = 'TARGET LOCKED'

LABEL_MASS = ('质量', 'MASS')
LABEL_SPEED = ('速度', 'SPEED')
LABEL_KE = ('动能', 'KE')
LABEL_RESULT = ('结果', 'RESULT')
FORMULA = 'KE = ½ × m × v²'

WARNING = 'WARNING'
WARNING_CN = '高动能物体接近'
HURRY = 'HURRY UP!'
SLOW = 'SLOW ×0.25'
BELL = '叮铃铃！'
ANC = '降噪 ON'
TERMINATED_CN = '测试终止'
TERMINATED_EN = 'TEST TERMINATED'
STILL_HERE = '位置：仍在非机动车道'

END_HEAD = '实验结论'
END_MAIN = '非机动车道不是人行道'
END_EN = 'THE BIKE LANE IS NOT A SIDEWALK'
END_ARROW = '人行道在这里'
END_NOTE = '虚构像素动画 · 请勿模仿'

SIGN_PLATE = '非机动车道'
SHOP_SIGNS = ['便利店', '奶茶', '修车', '24H']


def level_lines(lv):
    """HUD strings for a level, generated from the physics table (no hand-typed KE)."""
    L = LEVELS[lv]
    return {
        'level': f'LEVEL {lv + 1:02d}',
        'name_cn': L['name_cn'],
        'name_en': L['name_en'],
        'mass': f"{fmt_num(L['m'])} kg",
        'mass_note': L.get('mass_note', ''),
        'speed': f"{fmt_num(L['v'])} m/s",
        'speed_note': L.get('speed_note', ''),
        'ke': f"{fmt_ke(L['ke'])} J",
        'calc': f"= ½ × {fmt_num(L['m'])} × {fmt_num(L['v'])}² = {fmt_ke(L['ke'])} J",
        'ratio': fmt_ratio(L['ratio']),
        'hit_word': L['hit_word'],
        'result_cn': L['result_cn'],
        'result_en': L['result_en'],
    }


def all_strings():
    out = [TITLE_CN_1, TITLE_CN_2, TITLE_EN, SUBJECT_HEAD, SUBJECT_HEAD_EN, *SUBJECT_LINES, TARGET_LOCKED,
           *LABEL_MASS, *LABEL_SPEED, *LABEL_KE, *LABEL_RESULT, FORMULA, WARNING, WARNING_CN, HURRY, SLOW,
           BELL, ANC, TERMINATED_CN, TERMINATED_EN, STILL_HERE, END_HEAD, END_MAIN, END_EN, END_ARROW,
           END_NOTE, SIGN_PLATE, *SHOP_SIGNS, 'FINAL', '', '♪', '↑', '←', '→']
    for lv in range(len(LEVELS)):
        out += [v for v in level_lines(lv).values() if v]
    return out
