"""Colour palette for the pixel video.

A dusk city palette: deep indigo night sky, magenta/orange sunset, cool lavender
street greys, a green bike lane, and a HUD built on safety yellow + warning red.
"""
import numpy as np


def hx(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


C = {
    # inks
    'ink': '#120e1f', 'ink2': '#1c1630',
    # sky (top -> horizon)
    'sky0': '#0f0c22', 'sky1': '#17123a', 'sky2': '#221a4f', 'sky3': '#33225f',
    'sky4': '#4b2b6e', 'sky5': '#6d3477', 'sky6': '#963f7c', 'sky7': '#c24f78',
    'sky8': '#e2686c', 'sky9': '#f48a5f', 'sky10': '#fbad63', 'sky11': '#ffd07a',
    'star': '#fff3d6', 'star2': '#bfb5e8',
    # far skyline
    'far0': '#2a1f4d', 'far1': '#36285c', 'far2': '#45346d', 'farwin': '#e7a35c', 'farwin2': '#ffd27a',
    # buildings
    'bd0': '#231b3d', 'bd1': '#2f2550', 'bd2': '#3c3163', 'bd3': '#4c4178', 'bd4': '#625796',
    'brick0': '#4a2c4e', 'brick1': '#5c3659', 'brick2': '#71426a',
    'win_warm': '#ffcf73', 'win_warm2': '#f0a052', 'win_dim': '#3e3466', 'win_dim2': '#4d4280',
    'win_tv': '#8fd3ff', 'win_frame': '#1b1532',
    'shop_lit': '#fff2c6', 'shop_lit2': '#ffd98f', 'shop_in': '#e9c27a',
    # street
    'sw0': '#5f587f', 'sw1': '#736c94', 'sw2': '#867fa6', 'sw3': '#9a94b8', 'sw4': '#b2acce',
    'tact0': '#b8892a', 'tact1': '#e3b43b', 'tact2': '#f7d36a',
    'curb0': '#4a4469', 'curb1': '#7d77a0', 'curb2': '#b9b4d2', 'curb3': '#d8d4ea',
    'lane0': '#1b4a3f', 'lane1': '#235c4c', 'lane2': '#2c6f59', 'lane3': '#388466', 'lane4': '#4a9c78',
    'paint': '#e9edf5', 'paint2': '#b9c0d3', 'paintw': '#8f97ae',
    'asp0': '#221e33', 'asp1': '#2b263f', 'asp2': '#35304b', 'asp3': '#403a58', 'asp4': '#4d4767',
    # signs / metal
    'blue0': '#173a85', 'blue1': '#2558bf', 'blue2': '#4d86e6', 'blue3': '#9cc4ff',
    'pole0': '#2e2a47', 'pole1': '#4a4568', 'pole2': '#6c6790',
    'lamp': '#fff3c4', 'lamp2': '#ffd98f', 'glow': '#ffcf80',
    # trees
    'tr0': '#14302f', 'tr1': '#1d4439', 'tr2': '#285a45', 'tr3': '#367050', 'tr4': '#4c8a5c',
    'trunk0': '#2a1d2c', 'trunk1': '#45303f',
    # neon
    'neon_pink': '#ff5fa8', 'neon_pink2': '#ffa6d0', 'neon_cyan': '#4ff0ff', 'neon_cyan2': '#b5fbff',
    'neon_green': '#6dff9a',
    # HUD
    'hud_bg': '#0d1230', 'hud_bg2': '#161d45', 'hud_line': '#2b3570',
    'yel': '#ffd23f', 'yel2': '#c99a1c', 'yel3': '#fff1a8',
    'red': '#ff3b45', 'red2': '#a81b2e', 'red3': '#ff9a9a',
    'org': '#ff962e', 'grn': '#4fe07a', 'grn2': '#1f8f4a', 'cyn': '#5ce8f0', 'cyn2': '#1f8fa8',
    'wht': '#f6f5fb', 'gry': '#9a9cbc', 'gry2': '#5d6088',
    # pedestrian
    'hair0': '#24172a', 'hair1': '#3c2846', 'hair2': '#5a3f63',
    'skin0': '#f6c9a1', 'skin1': '#d9966f', 'skin2': '#a8644f',
    'lit0': '#c9f1ea', 'lit1': '#8fd8d8',
    'hood0': '#ffb064', 'hood1': '#ff7d2e', 'hood2': '#d4551d', 'hood3': '#9c3a17',
    'jean0': '#6f93d6', 'jean1': '#4466ad', 'jean2': '#2e4880', 'jean3': '#1f2f58',
    'shoe0': '#f7f5fb', 'shoe1': '#bdb8d4', 'shoe2': '#6f6890',
    'hp0': '#f7f7ff', 'hp1': '#c3c5de', 'hp2': '#8b8db0', 'hpr': '#ff4d5a',
    'ph0': '#2a2a40', 'ph1': '#4a4a6a', 'phg': '#9af4ff', 'phg2': '#52c6ea',
    # e-bike / rider
    'bk0': '#e8ebf5', 'bk1': '#b5b9cf', 'bk2': '#7b7f9e', 'bk3': '#3e4060', 'tyre': '#1a1726',
    'helm0': '#ff6a6a', 'helm1': '#e8313a', 'helm2': '#9c1c2b', 'visor': '#3a3a5c',
    'jk0': '#5a7bd0', 'jk1': '#34509c', 'jk2': '#22356c', 'pant0': '#3b3656', 'pant1': '#28243d',
    'hl': '#fff8d0', 'hl2': '#ffe58a',
    # balls
    'pp0': '#ffffff', 'pp1': '#dcdcec', 'pp2': '#a9a9c6',
    'tn0': '#e9ff6a', 'tn1': '#c8e43a', 'tn2': '#93ab22',
    'sc0': '#ffffff', 'sc1': '#cfd0e2', 'sc2': '#22202f',
    'bw0': '#9a86ff', 'bw1': '#5c45d0', 'bw2': '#3a2a96', 'bw3': '#22175e',
    # fx
    'spark': '#fff27a', 'spark2': '#ffbf3a', 'dust0': '#c9c2de', 'dust1': '#9b94b8',
    'white': '#ffffff',
}
P = {k: hx(v) for k, v in C.items()}


def col(name):
    return P[name]
