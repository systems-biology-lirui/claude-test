"""Pixel FX: analytic particles, impact bursts, shockwaves, screen cracks, post effects."""
import math
import numpy as np
import gfx
from gfx import P, BAYER4


# ---------------------------------------------------------------- particles (stateless, analytic)
class Burst:
    """A burst of particles spawned at world time t0. Positions are closed-form in time,
    so any frame can be rendered independently (no simulation state)."""

    def __init__(self, t0, x, y, n, speed=(40, 120), angle=(0, 360), colors=('spark',), life=(0.3, 0.7),
                 gravity=300.0, drag=3.0, size=(1, 2), seed=0, ground=None):
        r = np.random.default_rng(seed)
        self.t0 = t0
        self.x, self.y = x, y
        a = np.deg2rad(r.uniform(angle[0], angle[1], n))
        s = r.uniform(speed[0], speed[1], n)
        self.vx = np.cos(a) * s
        self.vy = -np.sin(a) * s
        self.life = r.uniform(life[0], life[1], n)
        self.col = [colors[i % len(colors)] for i in r.permutation(n)]
        self.size = r.integers(size[0], size[1] + 1, n)
        self.g = gravity
        self.k = drag
        self.ground = ground

    def draw(self, canvas, t, cam):
        dt = t - self.t0
        if dt < 0:
            return
        k, g = self.k, self.g
        e = math.exp(-k * dt)
        fx = (1 - e) / k
        px = self.x + self.vx * fx
        py = self.y + self.vy * fx + g / k * (dt - fx)
        for i in range(len(px)):
            if dt > self.life[i]:
                continue
            fade = 1 - dt / self.life[i]
            yy = py[i]
            if self.ground is not None and yy > self.ground:
                yy = self.ground
            sx, sy = cam(px[i], yy)
            s = int(self.size[i]) if fade > 0.4 else 1
            if fade < 0.25 and BAYER4[int(sy) % 4, int(sx) % 4] > fade * 4:
                continue
            gfx.fill_rect(canvas, sx, sy, s, s, self.col[i])


# ---------------------------------------------------------------- shapes
def star_mask(R, spikes=8, inner=0.45, jitter=0.25, seed=0, rot=0.0):
    r = np.random.default_rng(seed)
    jit = r.uniform(1 - jitter, 1 + jitter, spikes * 2)
    n = int(R * 2 + 3)
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    ang = (np.arctan2(yy, xx) + rot) % (2 * np.pi)
    seg = ang / (2 * np.pi) * spikes * 2
    i0 = np.floor(seg).astype(int) % (spikes * 2)
    i1 = (i0 + 1) % (spikes * 2)
    f = seg - np.floor(seg)
    rad = np.where(np.arange(spikes * 2) % 2 == 0, 1.0, inner) * jit
    rr = (rad[i0] * (1 - f) + rad[i1] * f) * R
    return np.sqrt(xx ** 2 + yy ** 2) <= rr


def draw_impact(canvas, sx, sy, age_frames, size=1.0, seed=0, palette=('white', 'spark', 'spark2')):
    """Comic impact starburst, 4-frame animation (age in frames)."""
    if age_frames < 0 or age_frames > 5:
        return
    R = [10, 14, 16, 15, 12, 8][age_frames] * size
    m_out = star_mask(R, spikes=9, inner=0.5, seed=seed, rot=0.2 * age_frames)
    m_in = star_mask(R * 0.6, spikes=9, inner=0.55, seed=seed + 1, rot=0.2 * age_frames + 0.3)
    n = m_out.shape[0]
    x0, y0 = int(sx - n // 2), int(sy - n // 2)
    if age_frames <= 1:
        gfx.blit_mask(canvas, m_out, x0, y0, palette[1])
        n2 = m_in.shape[0]
        gfx.blit_mask(canvas, m_in, int(sx - n2 // 2), int(sy - n2 // 2), palette[0])
    elif age_frames <= 3:
        ring = m_out.copy()
        n2 = m_in.shape[0]
        off = (n - n2) // 2
        inner = np.zeros_like(ring)
        inner[off:off + n2, off:off + n2] = m_in
        gfx.blit_mask(canvas, ring & ~inner, x0, y0, palette[2] if age_frames == 3 else palette[1])
    else:
        ring = m_out & ~gfx.dither_mask(n, n, 0.5)
        n2 = m_in.shape[0]
        off = (n - n2) // 2
        inner = np.zeros_like(ring)
        inner[off:off + n2, off:off + n2] = m_in
        gfx.blit_mask(canvas, ring & ~inner, x0, y0, palette[2])


def draw_ring(canvas, sx, sy, radius, color='white', thick=1, alpha=1.0, squash=1.0):
    if radius <= 0:
        return
    R = int(radius + thick + 1)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    d = np.sqrt(xx ** 2 + (yy / max(squash, 0.05)) ** 2)
    m = (d <= radius + thick * 0.5) & (d >= radius - thick * 0.5)
    if alpha < 1:
        m &= BAYER4[(yy.astype(int) + R) % 4, (xx.astype(int) + R) % 4] < alpha
    gfx.blit_mask(canvas, m, int(sx) - R, int(sy) - R, color)


def draw_dust(canvas, sx, sy, age, n=6, spread=18, seed=3, color='dust0'):
    """Puffs expanding outward along the ground (age in seconds, ~0.6s life)."""
    if age < 0 or age > 0.7:
        return
    r = np.random.default_rng(seed)
    for i in range(n):
        side = -1 if i % 2 == 0 else 1
        dist = spread * (0.4 + 0.6 * r.random()) * (1 - math.exp(-age * 7))
        px = sx + side * dist
        py = sy - 2 - age * 10 * r.random()
        rad = 1.5 + age * 7 * (0.6 + 0.4 * r.random())
        fade = 1 - age / 0.7
        m = gfx.circle_mask(int(rad))
        dm = gfx.dither_mask(m.shape[0], m.shape[1], fade, int(px), int(py))
        gfx.blit_mask(canvas, m & dm, int(px - rad), int(py - rad), color if i % 3 else 'dust1')


def speed_lines(canvas, x0, x1, y, n=3, spacing=3, color='white', phase=0):
    for i in range(n):
        yy = int(y + (i - n // 2) * spacing)
        length = int((x1 - x0) * (0.5 + 0.5 * ((i * 37 + phase) % 7) / 7))
        gfx.line(canvas, x1 - length, yy, x1, yy, color, alpha=0.8, dash=(4, 2), phase=phase)


# ---------------------------------------------------------------- screen crack (L5)
def make_crack(cx, cy, w=270, h=480, seed=5):
    """Return list of polylines (in screen coords) forming a radial crack web."""
    r = np.random.default_rng(seed)
    lines = []
    n_rays = 13
    for i in range(n_rays):
        a = i / n_rays * 2 * math.pi + r.normal(0, 0.12)
        x, y = float(cx), float(cy)
        pts = [(x, y)]
        L = r.uniform(150, 420)
        steps = int(L / 6)
        for s in range(steps):
            a += r.normal(0, 0.18)
            x += math.cos(a) * 6; y += math.sin(a) * 6
            pts.append((x, y))
            if r.random() < 0.08 and s > 3:   # branch
                b = a + r.choice([-1, 1]) * r.uniform(0.4, 0.9)
                bx, by = x, y
                bp = [(bx, by)]
                for _ in range(int(r.uniform(4, 12))):
                    b += r.normal(0, 0.2)
                    bx += math.cos(b) * 6; by += math.sin(b) * 6
                    bp.append((bx, by))
                lines.append(bp)
        lines.append(pts)
    # concentric shatter rings near the centre
    for rad in (14, 30, 52):
        k = int(rad * 0.9)
        a0 = r.uniform(0, 2 * math.pi)
        pts = []
        for j in range(k + 1):
            a = a0 + j / k * 2 * math.pi * r.uniform(0.6, 0.95)
            rr = rad * r.uniform(0.85, 1.15)
            pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
        lines.append(pts)
    return lines


def draw_crack(canvas, lines, progress=1.0, alpha=1.0):
    for pts in lines:
        n = max(2, int(len(pts) * min(1.0, progress)))
        for (x0, y0), (x1, y1) in zip(pts[:n - 1], pts[1:n]):
            gfx.line(canvas, x0 + 1, y0 + 1, x1 + 1, y1 + 1, 'ink', alpha=0.7 * alpha)
            gfx.line(canvas, x0, y0, x1, y1, 'white', alpha=alpha)


# ---------------------------------------------------------------- post effects
def flash(canvas, color='white', a=1.0):
    c = P[color].astype(np.float32)
    canvas[:] = canvas * (1 - a) + c * a


def invert(canvas):
    canvas[:] = 255.0 - canvas


def tint(canvas, color, a):
    c = P[color].astype(np.float32)
    canvas[:] = canvas * (1 - a) + c * a


def desaturate(canvas, amount=0.6, cool=0.08):
    g = canvas @ np.array([0.299, 0.587, 0.114], np.float32)
    canvas[:] = canvas * (1 - amount) + g[..., None] * amount
    canvas[..., 2] = np.clip(canvas[..., 2] * (1 + cool), 0, 255)
    canvas[..., 0] *= (1 - cool * 0.5)


def scanlines(canvas, strength=0.18, phase=0):
    canvas[phase % 2::2] *= (1 - strength)


def vignette(canvas, strength=0.35):
    H, W = canvas.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    m = np.clip((d - 0.55) * 1.4, 0, 1) * strength
    th = BAYER4[yy.astype(int) % 4, xx.astype(int) % 4]
    canvas[th < m] *= 0.72


def dither_fade(canvas, color, a):
    """Ordered-dither crossfade toward a colour (keeps hard pixel look)."""
    H, W = canvas.shape[:2]
    m = gfx.dither_mask(H, W, a)
    canvas[m] = P[color].astype(np.float32)
