"""Director: given a time t (seconds), compose one 270x480 frame.

World coordinates match scene.py (camera 0,0 shows x 0..269, y 0..479).
Animation of actors uses the warped world clock (timeline.world_time) so hit-stop and
slow motion apply consistently; HUD, shake and flashes use real time.
"""
import math
import numpy as np
import gfx
import fx
import hud
import scene
import sprites_ped
import sprites_obj
import timeline as TL
import copy_text as T
from gfx import P
from physics import LEVELS, ke_meter_frac, clamp, lerp, ease_out_cubic, ease_in_cubic, ease_in_out, ease_out_back

W, H = 270, 480
PX = 129                       # pedestrian sprite left (world)
FEET = scene.FEET_Y            # 296: last row of the sprite
PTOP = FEET - 55               # 241
LVS = TL.LV
IMP = [L['impact'] for L in LVS]

# contact geometry (world): (sprite_left, sprite_top) of each object at the moment of contact
CONTACT = {
    0: (PX + 2 - 4, PTOP + 10 - 2),     # ping-pong hits the headphone cup
    1: (PX + 2 - 7, PTOP + 9 - 3),      # tennis ball, head
    2: (PX + 3 - 10, PTOP + 22 - 5),    # soccer ball, upper back
    3: (PX + 4 - 12, FEET - 11),        # bowling ball rolls into the heel
    4: (PX + 4 - 45, FEET - 45),        # e-bike front wheel into the calves
}
# approach speeds in world px per world-second (stylised, ordered like the real speeds)
SPEED = {0: 450.0, 1: 560.0, 2: 500.0, 3: 300.0, 4: 230.0}


def sim_ball(x0, y0, vx, vy, size, g=900.0, e=0.55, fr=0.8, dur=3.0, ground=FEET + 1, spin=1.0):
    """Precompute a bouncing-ball path from contact. Returns arrays sampled at 240 Hz."""
    dt = 1 / 240.0
    n = int(dur / dt) + 2
    xs = np.zeros(n); ys = np.zeros(n); rs = np.zeros(n)
    x, y, rot = float(x0), float(y0), 0.0
    for i in range(n):
        xs[i], ys[i], rs[i] = x, y, rot
        vy += g * dt
        x += vx * dt; y += vy * dt
        rot += vx * dt / max(1.0, size / 2) * spin
        if y + size > ground:
            y = ground - size
            vy = -vy * e
            vx *= fr
            if abs(vy) < 40:
                vy = 0.0
                vx *= 0.995
    return xs, ys, rs


def sample(path, dw):
    xs, ys, rs = path
    i = int(clamp(dw * 240, 0, len(xs) - 1))
    return xs[i], ys[i], rs[i]


class Director:
    def __init__(self):
        self.scene = scene.Scene()
        self.ped = sprites_ped.build()
        self.obj = sprites_obj.build()
        eb = self.obj['ebike']
        self.obj['ebike_icon'] = gfx.scale_nearest(eb, 0.7)
        self.hud_sprites = dict(self.obj)
        # post-impact ball paths
        c = CONTACT
        self.paths = {
            0: sim_ball(c[0][0], c[0][1], -150, -260, 4, e=0.62),
            1: sim_ball(c[1][0], c[1][1], -190, -300, 7, e=0.6),
            2: sim_ball(c[2][0], c[2][1], -130, -420, 10, e=0.5),
            3: sim_ball(c[3][0], c[3][1], -115, 0, 12, e=0.0, fr=1.0, g=0.0),
        }
        # phone (L4): flies out of the hands
        hx, hy = PX + sprites_ped.HAND_POINT[0], PTOP + sprites_ped.HAND_POINT[1]
        self.phone_path = sim_ball(hx - 3, hy - 4, 120, -250, 9, e=0.0, fr=0.0, dur=2.0)
        self.crack = None
        self.bursts = []
        self._make_bursts()

    # ------------------------------------------------------------ bursts (world-time based)
    def _make_bursts(self):
        W_ = TL.W_IMPACT
        B = []
        cx, cy = CONTACT[0][0] + 4, CONTACT[0][1] + 2
        B.append(fx.Burst(W_[0], cx, cy, 6, speed=(30, 80), colors=('white', 'pp1'), life=(0.15, 0.3), seed=1))
        cx, cy = CONTACT[1][0] + 7, CONTACT[1][1] + 3
        B.append(fx.Burst(W_[1], cx, cy, 10, speed=(40, 120), colors=('tn0', 'spark', 'white'), life=(0.2, 0.4), seed=2))
        cx, cy = CONTACT[2][0] + 10, CONTACT[2][1] + 5
        B.append(fx.Burst(W_[2], cx, cy, 18, speed=(60, 170), colors=('spark', 'white', 'spark2'), life=(0.25, 0.5), seed=3))
        cx, cy = CONTACT[3][0] + 12, CONTACT[3][1] + 6
        B.append(fx.Burst(W_[3], cx, cy, 16, speed=(50, 160), angle=(10, 170), colors=('spark', 'white', 'bw0'),
                          life=(0.25, 0.55), seed=4, gravity=500, ground=FEET))
        cx, cy = CONTACT[4][0] + 45, CONTACT[4][1] + 32
        B.append(fx.Burst(W_[4], cx, cy, 36, speed=(80, 260), colors=('spark', 'white', 'spark2', 'red'),
                          life=(0.3, 0.8), seed=5, gravity=420))
        B.append(fx.Burst(W_[4], cx, FEET - 1, 14, speed=(30, 90), angle=(0, 180), colors=('dust0', 'dust1'),
                          life=(0.4, 0.8), seed=6, gravity=-20, drag=4, size=(2, 3)))
        self.bursts = B

    # ------------------------------------------------------------ camera / shots
    def camera(self, t):
        """Returns (cam_y, zoom, focus, shake)."""
        cam_y, zoom, focus = 0.0, 1, None
        if t < TL.TILT_END + 0.05:
            k = ease_in_out(clamp((t - 0.1) / (TL.TILT_END - 0.1)))
            cam_y = -440 * (1 - k)
        # close-ups (cut on action)
        if IMP[0] <= t < IMP[0] + 0.75:
            zoom, focus = 2, (136, 252)
        elif IMP[1] <= t < IMP[1] + 0.8:
            zoom, focus = 2, (136, 254)
        elif IMP[2] - TL.L3_CU_PRE <= t < IMP[2] + 0.95:
            zoom, focus = 2, (138 + 12 * clamp((t - IMP[2]) / 0.6), 262)
        elif IMP[3] <= t < IMP[3] + 0.47:
            zoom, focus = 2, (146, 268)
        elif LVS[4]['slow'][0] <= t < IMP[4] + 8 / 30:
            zoom, focus = 2, (124, 262)
        # launch tilt (follow the subject)
        if IMP[4] + 8 / 30 <= t < TL.FALL_CAM:
            st = self.flight_state(t)
            cam_y = st['cam_y']
        elif TL.FALL_CAM <= t < TL.FALL_CAM + 0.5:
            k = ease_in_out((t - TL.FALL_CAM) / 0.5)
            cam_y = -460 * (1 - k)
        # shake
        A = 0.0
        for i, amp, dec in ((1, 1.2, 9), (2, 3.5, 6), (3, 4.0, 6), (4, 7.0, 3.5)):
            if t >= IMP[i]:
                A += amp * math.exp(-(t - IMP[i]) * dec)
        if t >= TL.LAND_T:
            A += 3.0 * math.exp(-(t - TL.LAND_T) * 9)
        l4_land = IMP[3] + 4 / 30 + 0.3
        if t >= l4_land:
            A += 2.5 * math.exp(-(t - l4_land) * 9)
        if TL.WARNING_START <= t < TL.WARNING_END:
            A += 0.6
        if t >= TL.FREEZE_T:
            A = 0
        sx = int(round(A * math.sin(t * 91.0)))
        sy = int(round(A * math.cos(t * 77.0)))
        return cam_y, zoom, focus, (sx, sy)

    # ------------------------------------------------------------ pedestrian
    def beat_bob(self, t):
        if IMP[4] <= t < TL.LAND_T:
            return False
        ph = TL.beat_pos(t) % 1.0
        return ph < 0.4

    def ped_x(self, t):
        x = PX
        dw3 = TL.since_impact(t, 2)
        if dw3 >= 0:
            x = PX + 22 * ease_out_cubic(clamp(dw3 / 0.55))
        if t >= LVS[3]['start']:
            x = PX + 22 * (1 - ease_in_out(clamp((t - LVS[3]['start']) / 0.5)))
        return x

    def flight_state(self, t):
        dw = TL.since_impact(t, 4)
        y = PTOP - 620 * ease_out_cubic(clamp(dw / 1.0))
        x = PX + 30 * dw + 10
        cam_y = max(-460.0, min(0.0, y - 150))
        rot = 600 * dw
        scale = 1.0 - 0.88 * ease_in_cubic(clamp((dw - 0.5) / 0.45))
        return dict(dw=dw, x=x, y=y, cam_y=cam_y, rot=rot, scale=scale)

    def ped_draw(self, canvas, t, cam):
        """Draw the pedestrian (and his loose phone). cam(x,y)->screen."""
        P_ = self.ped
        glow = (int(t * 7.5) % 2)
        cracked = t >= IMP[3] + 1.4
        bob = self.beat_bob(t)
        base = ('bob' if bob else 'idle') + ('_c' if cracked else '') + f'_{glow}'
        spr = P_[base]
        x, top = self.ped_x(t), PTOP
        ox, oy = 0, 0
        draw_phone = None
        flashw = False

        # ---- L2 wobble
        dw2 = TL.since_impact(t, 1)
        if 0 <= dw2 < 0.9:
            A = 3.2 * math.exp(-dw2 * 4.5) * math.sin(dw2 * 26)
            offs = [int(round(A * max(0.0, (40 - r)) / 40)) for r in range(spr.shape[0])]
            spr, shift = gfx.shear_rows(spr, offs)
            ox -= shift
            if dw2 < 0.05:
                flashw = True
        # ---- L3 spin + hop + dizzy
        dw3 = TL.since_impact(t, 2)
        if 0 <= dw3 < 2.2 and t < LVS[3]['start']:
            if dw3 < 0.85:
                p = ease_out_cubic(dw3 / 0.85)
                ang = p * 4 * math.pi
                c = math.cos(ang)
                sx = c if abs(c) > 0.12 else (0.12 if c >= 0 else -0.12)
                w0 = spr.shape[1]
                spr = gfx.scale_nearest(spr, sx, 1.0)
                ox += (w0 - spr.shape[1]) // 2
                hop = 8 * 4 * (dw3 / 0.5) * (1 - dw3 / 0.5) if dw3 < 0.5 else 0
                oy -= int(round(hop))
                if dw3 < 0.04:
                    flashw = True
            else:
                sway = 2.0 * math.sin(t * 9) * math.exp(-(dw3 - 0.85) * 0.8)
                offs = [int(round(sway * max(0.0, (34 - r)) / 34)) for r in range(spr.shape[0])]
                spr, shift = gfx.shear_rows(spr, offs)
                ox -= shift
        # ---- L4 fall forward like a bowling pin, lie, spring up
        dw4 = TL.since_impact(t, 3)
        if 0 <= dw4 < 1.32:
            spr = P_['nophone']
            if dw4 < 0.3:
                ang = 90 * ease_in_cubic(dw4 / 0.3)
            elif dw4 < 0.42:
                u = (dw4 - 0.3) / 0.12
                ang = 90 - 7 * math.sin(u * math.pi)
            elif dw4 < 1.12:
                ang = 90
            else:
                ang = 90 * (1 - ease_out_back(clamp((dw4 - 1.12) / 0.2)))
            pivot = (20, 55)
            rs, (px_, py_) = gfx.rotsprite(spr, ang, pivot)
            wx = x + pivot[0] + (6 * ease_out_cubic(clamp(dw4 / 0.3)) if dw4 < 1.12 else 6 * (1 - clamp((dw4 - 1.12) / 0.2)))
            wy = top + pivot[1]
            if dw4 >= 1.12:
                wy -= int(10 * math.sin(clamp((dw4 - 1.12) / 0.2) * math.pi))
            sx_, sy_ = cam(wx - px_, wy - py_)
            self._shadow(canvas, cam, x + 11, 30 if ang > 45 else 20)
            gfx.blit_hard(canvas, rs, sx_, sy_, mask_color=P['white'] if dw4 < 0.03 else None)
            self._phone_l4(canvas, t, cam)
            return
        if IMP[3] + 1.32 / 1.0 <= t < IMP[3] + 1.6:
            self._phone_l4(canvas, t, cam)
        # ---- L5 launch
        dw5 = TL.since_impact(t, 4)
        if dw5 >= 0 and t < TL.FALL_START:
            if dw5 <= 0.0:
                pass
            st = self.flight_state(t)
            if t >= IMP[4] + 8 / 30 or dw5 > 0:
                if st['scale'] > 0.14 and dw5 < 0.97:
                    s = P_['nophone']
                    if st['scale'] < 0.99:
                        s = gfx.scale_nearest(s, st['scale'])
                    rs, (px_, py_) = gfx.rotsprite(s, st['rot'], (s.shape[1] / 2, s.shape[0] / 2))
                    cx, cy = st['x'] + 11, st['y'] + 28
                    # afterimages
                    for k in (3, 2, 1):
                        st2 = self.flight_state(t - k / 60.0)
                        sx2, sy2 = cam(st2['x'] + 11 - px_, st2['y'] + 28 - py_)
                        m = gfx.dither_mask(rs.shape[0], rs.shape[1], 0.5 - 0.12 * k)
                        ghost = rs.copy(); ghost[~m, 3] = 0
                        gfx.blit_hard(canvas, ghost, sx2, sy2, mask_color=P['hood0'])
                    sx_, sy_ = cam(cx - px_, cy - py_)
                    gfx.blit_hard(canvas, rs, sx_, sy_)
                    self._phone_l5(canvas, t, cam)
                return
            # (hit-stop: fall through and draw standing, flashed)
            flashw = True
            spr = P_['nophone'] if False else spr
        # ---- fall back from the sky and land
        if TL.FALL_START <= t < TL.LAND_T:
            tau = t - TL.FALL_START
            dur = TL.LAND_T - TL.FALL_START
            yy = -80 + (PTOP + 80) * (tau / dur) ** 2
            ang = 540 * (1 - tau / dur)
            s = P_['nophone']
            rs, (px_, py_) = gfx.rotsprite(s, ang, (11, 28))
            sx_, sy_ = cam(PX + 11 - px_, yy + 28 - py_)
            for k in (2, 1):
                yk = -80 + (PTOP + 80) * (max(0, tau - k / 50) / dur) ** 2
                m = gfx.dither_mask(rs.shape[0], rs.shape[1], 0.35)
                g_ = rs.copy(); g_[~m, 3] = 0
                gfx.blit_hard(canvas, g_, sx_, cam(0, yk + 28 - py_)[1], mask_color=P['hood0'])
            gfx.blit_hard(canvas, rs, sx_, sy_)
            return
        if TL.LAND_T <= t < TL.PHONE_CATCH:
            spr = P_['nophone']
        if TL.LAND_T <= t < TL.LAND_T + 0.2:
            u = (t - TL.LAND_T) / 0.2
            if u < 0.4:
                sq = (1.22, 0.8)
            elif u < 0.75:
                sq = (0.92, 1.08)
            else:
                sq = (1.0, 1.0)
            w0, h0 = spr.shape[1], spr.shape[0]
            spr = gfx.scale_nearest(spr, sq[0], sq[1])
            ox += (w0 - spr.shape[1]) // 2
            oy += h0 - spr.shape[0]
        if TL.LAND_T - 0.4 < t < TL.PHONE_CATCH and t >= TL.LAND_T - 0.25:
            self._phone_fall(canvas, t, cam)

        # shadow + body
        self._shadow(canvas, cam, x + 11 + ox * 0, 18)
        sx_, sy_ = cam(x + ox, top + oy)
        gfx.blit_hard(canvas, spr, sx_, sy_, mask_color=P['white'] if flashw else None)

    def _shadow(self, canvas, cam, cx, w):
        m = gfx.ellipse_mask(w // 2, 2)
        sx, sy = cam(cx - w // 2, FEET - 1)
        dm = gfx.dither_mask(m.shape[0], m.shape[1], 0.55, int(sx), int(sy))
        gfx.blit_mask(canvas, m & dm, sx, sy, 'lane0')

    def _phone_l4(self, canvas, t, cam):
        dw = TL.since_impact(t, 3)
        if dw < 0:
            return
        cracked = dw > 0.5
        spr = self.obj['phone_cracked' if cracked else 'phone']
        if dw < 1.2:
            x, y, _ = sample(self.phone_path, dw)
            if y + 9 >= FEET:
                y = FEET - 6
                spr = gfx.rot90(spr, 1)           # lying flat on the lane
                rot = 0
            else:
                rot = dw * 900
                spr, (px_, py_) = gfx.rotsprite(spr, rot)
                x -= px_ - 3; y -= py_ - 4
            gfx.blit_hard(canvas, spr, *cam(x, y))
        elif dw < 1.42:
            # hops back into his hands
            x0, y0, _ = sample(self.phone_path, 1.19)
            hx, hy = self.ped_x(t) + sprites_ped.HAND_POINT[0] - 3, PTOP + sprites_ped.HAND_POINT[1] - 4
            u = (dw - 1.2) / 0.22
            x = lerp(x0, hx, u)
            y = lerp(FEET - 6, hy, u) - 30 * 4 * u * (1 - u)
            s2, (px_, py_) = gfx.rotsprite(spr, -720 * u)
            gfx.blit_hard(canvas, s2, *cam(x - px_ + 3, y - py_ + 4))

    def _phone_l5(self, canvas, t, cam):
        dw = TL.since_impact(t, 4)
        x = PX + 17 + 90 * dw
        y = PTOP + 20 - 520 * ease_out_cubic(clamp(dw / 0.9)) + 40 * dw
        spr, (px_, py_) = gfx.rotsprite(self.obj['phone_cracked'], dw * 1100)
        gfx.blit_hard(canvas, spr, *cam(x - px_, y - py_))

    def _phone_fall(self, canvas, t, cam):
        hx, hy = PX + sprites_ped.HAND_POINT[0] - 3, PTOP + sprites_ped.HAND_POINT[1] - 4
        u = clamp((t - (TL.LAND_T - 0.25)) / (TL.PHONE_CATCH - (TL.LAND_T - 0.25)))
        y = lerp(-20, hy, u * u)
        spr, (px_, py_) = gfx.rotsprite(self.obj['phone_cracked'], 1080 * (1 - u))
        gfx.blit_hard(canvas, spr, *cam(hx - px_ + 3, y - py_ + 4))

    # ------------------------------------------------------------ objects
    def objects_draw(self, canvas, t, cam):
        for i in range(4):
            L = LVS[i]
            if not (L['start'] + 1.0 <= t < (LVS[i + 1]['start'] if i < 3 else TL.WARNING_START)):
                continue
            dw = TL.since_impact(t, i)
            key = LEVELS[i]['key']
            spr = self.obj[key]
            cx, cy = CONTACT[i]
            sz = spr.shape[1]
            v = SPEED[i]
            if dw < 0:
                x = cx + v * dw
                if i == 3:
                    y = cy
                    rot = (x / 6.0) * 57.3
                else:
                    # gentle ballistic arc into the contact point
                    g = 260.0
                    vy_imp = {0: 40.0, 1: 15.0, 2: 55.0}[i]
                    y = cy + vy_imp * dw + 0.5 * g * dw * dw * 0 - 0.5 * g * dw * dw * -0.0
                    rot = dw * 1500
                if x < -sz - 2:
                    continue
                # afterimage trail
                for k in ((6, 5, 4, 3, 2, 1) if i == 0 else (4, 3, 2, 1)):
                    dk = dw - k * 0.012 * (1 if i != 3 else 2)
                    xk = cx + v * dk
                    yk = y if i == 3 else cy + {0: 40.0, 1: 15.0, 2: 55.0}[i] * dk
                    m = gfx.dither_mask(spr.shape[0], spr.shape[1], max(0.1, 0.6 - 0.09 * k))
                    gh = spr.copy(); gh[~m, 3] = 0
                    gfx.blit_hard(canvas, gh, *cam(xk, yk), mask_color=P['white'] if i < 3 else P['bw0'])
                if i >= 0:
                    sxl, syl = cam(x, y + sz // 2)
                    fx.speed_lines(canvas, sxl - 26, sxl - 3, syl, n=3, spacing=3, color='white',
                                   phase=int(t * 30))
                if i in (2, 3):
                    rs, (px_, py_) = gfx.rotsprite(spr, rot)
                    gfx.blit_hard(canvas, rs, *cam(x + sz / 2 - px_, y + sz / 2 - py_))
                else:
                    gfx.blit_hard(canvas, spr, *cam(x, y))
                if i < 3:
                    self._ball_shadow(canvas, cam, x + sz / 2, FEET - (y + sz), sz)
            else:
                x, y, r = sample(self.paths[i], dw)
                if i in (2, 3):
                    rs, (px_, py_) = gfx.rotsprite(spr, math.degrees(r))
                    gfx.blit_hard(canvas, rs, *cam(x + sz / 2 - px_, y + sz / 2 - py_))
                else:
                    gfx.blit_hard(canvas, spr, *cam(x, y))
                if i < 3:
                    self._ball_shadow(canvas, cam, x + sz / 2, FEET - (y + sz), sz)

    def _ball_shadow(self, canvas, cam, cx, height, sz):
        w = max(2, int(sz * (1.0 - min(0.6, height / 120))))
        m = gfx.ellipse_mask(max(1, w // 2), 1)
        sx, sy = cam(cx - w // 2, FEET - 1)
        dm = gfx.dither_mask(m.shape[0], m.shape[1], 0.5, int(sx), int(sy))
        gfx.blit_mask(canvas, m & dm, sx, sy, 'lane0')

    def ebike_x(self, t):
        dw = TL.since_impact(t, 4)
        cx = CONTACT[4][0]
        if dw < 0:
            return cx + SPEED[4] * dw, SPEED[4]
        v0, a = 170.0, 380.0
        tt = min(dw, v0 / a)
        x = cx + v0 * tt - 0.5 * a * tt * tt
        v = max(0.0, v0 - a * dw)
        if dw > 0.95:                      # rider recovers and rides off to the right
            u = dw - 0.95
            x += 0.5 * 320 * u * u
            v = 320 * u
        return x, v

    def ebike_draw(self, canvas, t, cam):
        if not (TL.BIKE_ENTER - 0.3 <= t < TL.FALL_CAM + 0.6):
            return
        x, v = self.ebike_x(t)
        y = CONTACT[4][1]
        spr = self.obj['ebike'].copy()
        meta = self.obj['ebike_meta']
        ang = x / 7.0
        for (wx, wy) in meta['wheels']:
            sprites_obj.draw_wheel_spokes(spr, wx, wy, ang)
        dw = TL.since_impact(t, 4)
        # headlight cone on the ground ahead
        hlx, hly = meta['headlight']
        sx, sy = cam(x + hlx + 2, y + hly)
        for k in range(1, 46):
            hh = 1 + k // 5
            a = 0.5 * (1 - k / 46)
            for j in range(-hh, hh + 1):
                yy = int(sy + j + k // 6)
                xx = int(sx + k)
                if 0 <= xx < W and 0 <= yy < H and gfx.BAYER4[yy % 4, xx % 4] < a:
                    canvas[yy, xx] = canvas[yy, xx] * 0.55 + P['hl'] * 0.45
        # skid marks + smoke after impact
        if dw > 0:
            cx0 = CONTACT[4][0]
            for (wx, wy) in meta['wheels']:
                x0s, x1s = cx0 + wx, x + wx
                if x1s > x0s:
                    a_, b_ = cam(x0s, FEET)
                    c_, d_ = cam(x1s, FEET)
                    gfx.fill_rect(canvas, a_, b_ - 1, c_ - a_, 1, 'asp0', 0.8)
            if v > 10 and dw < 0.5:
                for k in range(4):
                    ph = (t * 13 + k * 0.25) % 1.0
                    px_, py_ = cam(x + 4 - ph * 14, FEET - 4 - ph * 10)
                    r = int(1 + ph * 4)
                    m = gfx.circle_mask(r)
                    dm = gfx.dither_mask(m.shape[0], m.shape[1], 1 - ph)
                    gfx.blit_mask(canvas, m & dm, px_ - r, py_ - r, 'dust0')
        wob = 0
        if 0 < dw < 0.6:
            wob = int(round(math.sin(dw * 40) * 1.5 * (1 - dw / 0.6)))
        self._shadow(canvas, cam, x + 24, 40)
        gfx.blit_hard(canvas, spr, *cam(x + wob, y))
        return x

    # ------------------------------------------------------------ background traffic (intro only)
    CAR_T = TL.CAR_T

    def car_draw(self, canvas, t, cam):
        t0, t1 = self.CAR_T
        if not (t0 <= t < t1):
            return
        spr = self.obj['car']
        x = -60 + (t - t0) / (t1 - t0) * 400
        y = 386 - spr.shape[0]
        sx, sy = cam(x, y)
        # headlight wash on the road ahead
        for k in range(1, 30):
            for j in range(-1 - k // 8, 2 + k // 8):
                xx, yy = sx + spr.shape[1] + k, sy + 8 + j + k // 10
                if 0 <= xx < W and 0 <= yy < H and gfx.BAYER4[yy % 4, xx % 4] < 0.45 * (1 - k / 30):
                    canvas[yy, xx] = canvas[yy, xx] * 0.6 + P['hl'] * 0.4
        m = gfx.ellipse_mask(24, 2)
        dm = gfx.dither_mask(m.shape[0], m.shape[1], 0.5, sx, sy)
        gfx.blit_mask(canvas, m & dm, sx + 2, sy + spr.shape[0] - 3, 'asp0')
        gfx.blit_hard(canvas, spr, sx, sy)
        fx.speed_lines(canvas, sx - 30, sx - 4, sy + 9, n=3, spacing=4, color='paint2', phase=int(t * 30))

    # ------------------------------------------------------------ music notes
    def notes_draw(self, canvas, t, cam):
        if IMP[4] <= t < TL.LAND_T + 0.2:
            return
        if t >= TL.FREEZE_T:
            t = TL.FREEZE_T
        w = TL.world_time(t)
        period = TL.NOTE_PERIOD
        k0 = int(w / period)
        for k in range(k0 - 3, k0 + 1):
            ts = k * period
            age = w - ts
            if age < 0 or age > 1.5:
                continue
            # skip notes that would spawn during flight/away
            tspawn_real = ts
            px = self.ped_x(ts) + 2
            if IMP[3] <= ts < IMP[3] + 1.4:
                continue
            x = px - 4 - age * 9 + math.sin(age * 6 + k) * 2
            y = PTOP + 4 - age * 22
            spr = self.obj['note' if k % 2 else 'note2']
            m = gfx.dither_mask(spr.shape[0], spr.shape[1], 1 - max(0, (age - 0.9) / 0.6))
            s2 = spr.copy(); s2[~m, 3] = 0
            gfx.blit_hard(canvas, s2, *cam(x, y))

    def dizzy_draw(self, canvas, t, cam):
        dw3 = TL.since_impact(t, 2)
        if not (0.55 < dw3 < 2.1) or t >= LVS[3]['start']:
            return
        hx = self.ped_x(t) + 11
        hy = PTOP - 3
        for k in range(3):
            a = t * 7 + k * 2 * math.pi / 3
            x = hx + math.cos(a) * 11
            y = hy + math.sin(a) * 3
            if math.sin(a) < -0.2:
                continue
            gfx.blit_hard(canvas, self.obj['star'], *cam(x - 2, y - 2))

    # ------------------------------------------------------------ trajectory preview
    def trajectory_draw(self, canvas, t, cam):
        for i in range(4):
            L = LVS[i]
            t0 = TL.traj_start(i)
            dw = TL.since_impact(t, i)
            sz = self.obj[LEVELS[i]['key']].shape[1]
            v = SPEED[i]
            t_launch = TL.real_time_of_world(TL.W_IMPACT[i] - (CONTACT[i][0] + sz + 4) / v)
            if not (t0 <= t < t_launch + 0.15):
                continue
            if int(t * 10) % 2 == 1 and t < t_launch:
                continue
            cx, cy = CONTACT[i]
            y = cy + sz / 2
            sx0, sy0 = cam(0, y)
            sx1, sy1 = cam(cx + sz, y)
            gfx.line(canvas, sx0, sy0, sx1 - 1, sy1, 'yel', dash=(2, 2), phase=int(t * 30))
            gfx.fill_rect(canvas, sx1 - 1, sy1 - 2, 1, 5, 'red')
            gfx.fill_rect(canvas, sx1 - 3, sy1, 5, 1, 'red')

    # ------------------------------------------------------------ main compose
    def render(self, t, draw_hud=True):
        cam_y, zoom, focus, (shx, shy) = self.camera(t)
        frozen = t >= TL.FREEZE_T
        tt = TL.FREEZE_T if frozen else t
        if frozen:
            cam_y, zoom, focus, (shx, shy) = self.camera(TL.FREEZE_T - 1e-3)
            shx = shy = 0
        canvas = np.zeros((H, W, 3), np.float32)

        insert = TL.PHONE_INSERT[0] <= tt < TL.PHONE_INSERT[1]
        if insert:
            self._phone_insert(canvas, tt)
            self._zoom = 2
            self._cam = (0, 0)
            self._zoom_origin = (0, 0)
        else:
            camx = shx
            camy = cam_y + shy

            def cam(x, y):
                return int(round(x - camx)), int(round(y - camy))
            self._cam = (camx, camy)
            self._zoom = zoom

            self.scene.draw(canvas, camx, camy, tt)
            self.car_draw(canvas, tt, cam)
            self.trajectory_draw(canvas, tt, cam)
            bx = self.ebike_draw(canvas, tt, cam)
            self.objects_draw(canvas, tt, cam)
            self.ped_draw(canvas, tt, cam)
            self.dizzy_draw(canvas, tt, cam)
            self.notes_draw(canvas, tt, cam)
            w_ = TL.world_time(tt)
            for b in self.bursts:
                b.draw(canvas, w_, lambda x, y: cam(x, y))
            self.world_fx(canvas, tt, cam)
            if zoom == 2:
                fxx, fyy = focus
                sx, sy = cam(fxx, fyy)
                x0 = int(clamp(sx - W // 4, 0, W - W // 2))
                y0 = int(clamp(sy - H // 4, 0, H - H // 2))
                crop = canvas[y0:y0 + H // 2, x0:x0 + W // 2]
                canvas = np.repeat(np.repeat(crop, 2, 0), 2, 1).copy()
                self._zoom_origin = (x0, y0)
            if frozen:
                fx.desaturate(canvas, 0.55, 0.1)
                canvas *= 0.85
                fx.scanlines(canvas, 0.2)
            self.screen_fx(canvas, tt, cam, zoom, focus)
        if draw_hud:
            self.hud_draw(canvas, tt, 2 if insert else zoom)
            if frozen:
                self.end_draw(canvas, t)
        self.post_fx(canvas, t, frozen)
        return canvas

    # ------------------------------------------------------------ fx in world space (pre-zoom)
    def world_fx(self, canvas, t, cam):
        # impact starbursts (frame-based age, real time)
        specs = {0: (0.45, CONTACT[0][0] + 3, CONTACT[0][1] + 2),
                 1: (0.75, CONTACT[1][0] + 6, CONTACT[1][1] + 3),
                 2: (1.0, CONTACT[2][0] + 10, CONTACT[2][1] + 5),
                 3: (1.0, CONTACT[3][0] + 12, CONTACT[3][1] + 6),
                 4: (2.0, CONTACT[4][0] + 44, CONTACT[4][1] + 34)}
        for i, (s, x, y) in specs.items():
            age = int(round((t - IMP[i]) * 30))
            if 0 <= age <= 5 + LVS[i]['hitstop']:
                a = max(0, age - LVS[i]['hitstop'] + 1) if age >= LVS[i]['hitstop'] else 0
                fx.draw_impact(canvas, *cam(x, y), a, size=s, seed=i)
        # shockwave rings
        for i, rmax, dur in ((2, 40, 0.35), (4, 120, 0.5)):
            age = t - IMP[i]
            if 0 <= age < dur:
                x, y = specs[i][1], specs[i][2]
                fx.draw_ring(canvas, *cam(x, y), rmax * ease_out_cubic(age / dur), 'white', thick=2 if i == 4 else 1,
                             alpha=1 - age / dur, squash=0.55)
                if i == 4 and age > 0.08:
                    fx.draw_ring(canvas, *cam(x, y), rmax * 0.7 * ease_out_cubic((age - 0.08) / dur), 'spark',
                                 thick=1, alpha=1 - age / dur, squash=0.55)
        # dust when L3 lands from hop / L4 body hits the ground / L5 landing
        dw3 = TL.since_impact(t, 2)
        if 0.45 < dw3 < 1.2 and t < LVS[3]['start']:
            fx.draw_dust(canvas, *cam(self.ped_x(t) + 11, FEET), dw3 - 0.45, seed=31)
        dw4 = TL.since_impact(t, 3)
        if 0.3 < dw4 < 1.0:
            fx.draw_dust(canvas, *cam(PX + 40, FEET), dw4 - 0.3, n=8, spread=30, seed=41)
        if TL.LAND_T <= t < TL.LAND_T + 0.7:
            fx.draw_dust(canvas, *cam(PX + 11, FEET), t - TL.LAND_T, n=8, spread=26, seed=51)
        # L5 twinkle where he vanished
        dw5 = TL.since_impact(t, 4)
        if 0.95 <= dw5 < 1.6 and t < TL.FALL_CAM:
            st = self.flight_state(TL.LV[4]['impact'] + 8 / 30 + 0.95 + 0.0)
            age = dw5 - 0.95
            sc = 1 + 2 * math.sin(clamp(age / 0.5) * math.pi)
            spr = gfx.scale_nearest(self.obj['star'], max(1, int(round(sc))))
            sx, sy = cam(st['x'] + 11, st['y'] + 28)
            if age < 0.55:
                gfx.blit_hard(canvas, spr, sx - spr.shape[1] // 2, sy - spr.shape[0] // 2)
                for k in range(4):
                    a = k * math.pi / 2 + age * 6
                    L_ = int(4 + 10 * math.sin(clamp(age / 0.5) * math.pi))
                    gfx.line(canvas, sx, sy, sx + math.cos(a) * L_, sy + math.sin(a) * L_, 'spark', alpha=0.8)

    # ------------------------------------------------------------ screen-space fx after zoom
    def screen_fx(self, canvas, t, cam, zoom, focus):
        pass

    def to_screen(self, x, y):
        """World -> final screen coords using the camera/zoom of the current frame."""
        camx, camy = self._cam
        sx, sy = x - camx, y - camy
        if self._zoom == 2:
            ox, oy = self._zoom_origin
            sx, sy = (sx - ox) * 2, (sy - oy) * 2
        return int(round(sx)), int(round(sy))

    def _phone_insert(self, canvas, t):
        """Close-up insert: the phone lands, screen cracks."""
        u = t - TL.PHONE_INSERT[0]
        canvas[:] = P['hud_bg']
        # radial speed lines
        for k in range(24):
            a = k / 24 * 2 * math.pi + 0.13
            r0 = 70 + (k % 3) * 10
            gfx.line(canvas, 135 + math.cos(a) * r0, 240 + math.sin(a) * r0 * 1.4,
                     135 + math.cos(a) * 300, 240 + math.sin(a) * 300, 'hud_line')
        cracked = u >= 0.05
        spr = self.obj['phone_xl_cracked' if cracked else 'phone_xl']
        spr = gfx.scale_nearest(spr, 4)
        rs, _ = gfx.rotsprite(spr, -10)
        bounce = int(8 * math.exp(-u * 18) * math.cos(u * 60))
        gfx.blit_hard(canvas, rs, 135 - rs.shape[1] // 2, 240 - rs.shape[0] // 2 + bounce,
                      mask_color=P['white'] if 0.05 <= u < 0.085 else None)
        if cracked:
            hud.impact_word(canvas, '咔!', 200, 150, u - 0.05, scale=3, palette=('wht', 'cyn', 'cyn2'))
            fx.Burst(0.05, 150, 210, 14, speed=(60, 160), colors=('white', 'cyn', 'phg'), life=(0.2, 0.4),
                     seed=9, gravity=300).draw(canvas, u, lambda x, y: (x, y))

    # ------------------------------------------------------------ HUD
    def ke_display(self, t):
        """Value and meter fraction shown on the bottom panel at time t."""
        cur = None
        for i, L in enumerate(LVS):
            c0 = TL.count_start(i)
            if t >= c0:
                prev = LEVELS[i - 1]['ke'] if i > 0 else 0.1
                k = ease_out_cubic(clamp((t - c0) / 0.4))
                v = 10 ** lerp(math.log10(prev), math.log10(LEVELS[i]['ke']), k)
                cur = (i, v, ke_meter_frac(v), k >= 1)
        return cur

    def hud_draw(self, canvas, t, zoom):
        sprites = self.hud_sprites
        # intro
        if t < TL.LOCK_T + 0.05:
            hud.title(canvas, t + 0.05, t)
        if TL.LOCK_T <= t < TL.INTRO_HUD_END:
            box = (PX - 2, PTOP - 4, PX + 23, FEET + 2)
            hud.reticle(canvas, t - TL.LOCK_T, box, t)
            hud.subject_card(canvas, t - TL.SUBJECT_T, (PX + 4, PTOP + 14), t)
        # levels
        hide = (t >= IMP[4] + 8 / 30)
        cur_lv = None
        for i, L in enumerate(LVS):
            nxt = LVS[i + 1]['start'] if i < 4 else 99
            if L['start'] <= t < nxt:
                cur_lv = i
        if cur_lv is not None and not hide:
            i = cur_lv
            L = LVS[i]
            sp = TL.CARD_SPEED[i]
            card_t0 = TL.card_start(i)
            tl = t - card_t0
            slide = clamp(tl / 0.25)
            hud.level_card(canvas, i, tl, sprites, t, slide=slide, typing_speed=sp, compact=(zoom == 2),
                           hurry=(i == 3 and t < LVS[3]['start'] + 0.9))
        # KE panel
        kd = self.ke_display(t)
        if kd is not None and zoom == 1 and not hide:
            i, v, frac, done = kd
            ratio = T.level_lines(i)['ratio'] if (done and i > 0) else None
            hud.ke_panel(canvas, v, frac, i, t, alpha_t=clamp((t - TL.count_start(0)) / 0.3), ratio_txt=ratio,
                         flash=(not done))
        # banners
        for i, L in enumerate(LVS[:4]):
            dur = 0.55 if i < 3 else 0.42
            hud.level_banner(canvas, f'LEVEL {i + 1:02d}', t - L['start'], dur=dur,
                             sub=(T.HURRY if i == 3 else None), t=t)
        if TL.WARNING_START <= t < TL.WARNING_END:
            hud.warning_overlay(canvas, t - TL.WARNING_START, t)
        # impact words
        words = [(0, 'L', 2, ('wht', 'pp1', 'cyn')), (1, 'L', 2, ('yel3', 'yel', 'org')),
                 (2, 'L', 3, ('yel3', 'yel', 'org')), (3, 'C', 3, ('yel3', 'yel', 'org', 'red')),
                 (4, 'C', 4, ('wht', 'yel', 'org', 'red'))]
        for (i, pos, sc, pal) in words:
            tl = t - IMP[i]
            if 0 <= tl < 0.75:
                word = LEVELS[i]['hit_word']
                if pos == 'L':
                    cx_, cy_ = CONTACT[i]
                    sx, sy = self.to_screen(cx_ - 8, cy_ - 14)
                    sx = int(clamp(sx, 40, W - 40)); sy = int(clamp(sy, 60, H - 60))
                elif i == 3:
                    sx, sy = 135, 150
                else:
                    sx, sy = 135, 120
                hud.impact_word(canvas, word, sx, sy, tl, scale=sc, palette=pal, shake=(2 if i >= 3 else 0),
                                dur=0.75 if i < 3 else 0.6)
        # slow tag
        for i in (2, 4):
            s0 = LVS[i]['slow'][0]
            if s0 <= t < IMP[i]:
                hud.slow_tag(canvas, t, LVS[i]['slow'][1])
        # L5 bell + ANC
        if TL.BELL_T <= t < IMP[4] - 0.05:
            bx, _ = self.ebike_x(t)
            sx, sy = self.to_screen(bx + 19, CONTACT[4][1] + 2)
            hud.bubble(canvas, T.BELL, int(sx), int(sy), t - TL.BELL_T)
        if TL.ANC_T <= t < IMP[4]:
            sx, sy = self.to_screen(PX + 22, PTOP + 2)
            sx = min(sx + 2, W - 60)
            hud.tag(canvas, T.ANC, int(sx), int(sy))
        # result stamps
        res = [(i, TL.result_time(i)) for i in range(4)]
        for (i, ts) in res:
            nxt = LVS[i + 1]['start'] if i < 3 else TL.WARNING_START
            if ts <= t < nxt - 0.05:
                stc = 'red'
                hud.stamp(canvas, LEVELS[i]['result_cn'], LEVELS[i]['result_en'], t - ts, color=stc)
        if TL.TERMINATED_T <= t < TL.FALL_CAM:
            hud.stamp(canvas, T.TERMINATED_CN, T.TERMINATED_EN, t - TL.TERMINATED_T, color='red', cy=230, angle=-5)
        # L5 crack overlay
        if IMP[4] <= t < TL.CRACK_FADE[1]:
            if self.crack is None:
                self.crack = fx.make_crack(150, 250)
            prog = clamp((t - IMP[4]) / 0.07 + 0.35)
            c0, c1 = TL.CRACK_FADE
            alpha = 1.0 if t < c0 else 1 - (t - c0) / (c1 - c0)
            if alpha > 0:
                lines = self.crack
                fx.draw_crack(canvas, lines, prog, alpha)

    def end_draw(self, canvas, t):
        tl = t - TL.FREEZE_T
        # red brackets around the subject + "still in the bike lane" tag
        on = int(t * 5) % 2 == 0
        box = (PX - 3, PTOP - 4, PX + 24, FEET + 2)
        L = 6
        col = 'red' if on else 'red3'
        for (cx, cy, dx, dy) in ((box[0], box[1], 1, 1), (box[2], box[1], -1, 1), (box[0], box[3], 1, -1), (box[2], box[3], -1, -1)):
            gfx.fill_rect(canvas, min(cx, cx + dx * (L - 1)), cy, L, 1, col)
            gfx.fill_rect(canvas, cx, min(cy, cy + dy * (L - 1)), 1, L, col)
        if tl > 0.05:
            s = T.STILL_HERE
            w_ = hud.tx().width(s, 12) + 8
            x0 = int(clamp(PX + 11 - w_ // 2, 4, W - w_ - 4))
            y0 = FEET + 8
            gfx.fill_rect(canvas, x0 - 1, y0 - 1, w_ + 2, 16, 'ink')
            gfx.fill_rect(canvas, x0, y0, w_, 14, 'red')
            hud.text(canvas, s, x0 + 4, y0 + 1, 12, color='wht')
        hud.end_card(canvas, t - TL.END_CARD, t)

    # ------------------------------------------------------------ post
    def post_fx(self, canvas, t, frozen):
        # flashes
        for i, (fa, inv) in {2: (0.8, False), 3: (0.7, False), 4: (1.0, True)}.items():
            f = int(round((t - IMP[i]) * 30))
            if f == 0:
                fx.flash(canvas, 'white', fa)
            elif f == 1 and inv:
                fx.invert(canvas)
        # red alarm pulse (warning + hit-stop of L5)
        if TL.WARNING_START <= t < TL.WARNING_END:
            ph = (t - TL.WARNING_START) * 2.5
            a = 0.18 * (0.5 + 0.5 * math.sin(ph * 2 * math.pi))
            fx.tint(canvas, 'red', a)
        if IMP[4] + 2 / 30 <= t < IMP[4] + 8 / 30:
            fx.tint(canvas, 'red', 0.12)
        # fade out
        if t >= TL.FADE_OUT:
            fx.dither_fade(canvas, 'ink', clamp((t - TL.FADE_OUT) / 0.4))
