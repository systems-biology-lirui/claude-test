"""Low-level pixel graphics: sprites as RGBA uint8 arrays, blitting, transforms."""
import numpy as np
from palette import P


def sprite(rows, keymap):
    """Build an RGBA sprite from ASCII rows. keymap: char -> palette name ('.' = clear)."""
    rows = [r for r in rows]
    h = len(rows)
    w = max(len(r) for r in rows)
    out = np.zeros((h, w, 4), np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in ('.', ' '):
                continue
            name = keymap[ch]
            out[y, x, :3] = P[name]
            out[y, x, 3] = 255
    return out


def solid(w, h, color):
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = P[color] if isinstance(color, str) else color
    out[..., 3] = 255
    return out


def flip_h(spr):
    return spr[:, ::-1].copy()


def flip_v(spr):
    return spr[::-1].copy()


def rot90(spr, k=1):
    return np.rot90(spr, k).copy()


def blit(canvas, spr, x, y, alpha=1.0, mask_color=None):
    """Alpha-composite RGBA sprite onto float32 RGB canvas at integer (x, y) top-left."""
    if spr is None:
        return
    x = int(round(x)); y = int(round(y))
    H, W = canvas.shape[:2]
    h, w = spr.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (s[..., 3:4].astype(np.float32) / 255.0) * alpha
    if mask_color is not None:
        rgb = np.broadcast_to(np.asarray(mask_color, np.float32), s[..., :3].shape)
    else:
        rgb = s[..., :3].astype(np.float32)
    dst = canvas[y0:y1, x0:x1]
    dst[:] = dst * (1 - a) + rgb * a


def blit_hard(canvas, spr, x, y, alpha_threshold=128, mask_color=None):
    """Blit with binary alpha (crisp pixel art)."""
    x = int(round(x)); y = int(round(y))
    H, W = canvas.shape[:2]
    h, w = spr.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    m = s[..., 3] >= alpha_threshold
    dst = canvas[y0:y1, x0:x1]
    if mask_color is not None:
        dst[m] = np.asarray(mask_color, np.float32)
    else:
        dst[m] = s[..., :3][m].astype(np.float32)


def scale_nearest(spr, sx, sy=None):
    if sy is None:
        sy = sx
    h, w = spr.shape[:2]
    nw = max(1, int(round(w * abs(sx))))
    nh = max(1, int(round(h * abs(sy))))
    xs = np.clip(((np.arange(nw) + 0.5) / nw * w).astype(int), 0, w - 1)
    ys = np.clip(((np.arange(nh) + 0.5) / nh * h).astype(int), 0, h - 1)
    out = spr[ys][:, xs]
    if sx < 0:
        out = out[:, ::-1]
    if sy < 0:
        out = out[::-1]
    return out.copy()


def _scale2x(a):
    """EPX / Scale2x on an RGBA array (exact colour equality)."""
    h, w = a.shape[:2]
    key = a.view(np.uint32).reshape(h, w) if a.dtype == np.uint8 and a.shape[2] == 4 else None
    if key is None:
        key = np.ascontiguousarray(a).view(np.uint32).reshape(h, w)
    p = np.pad(key, 1, mode='edge')
    B = p[0:-2, 1:-1]; D = p[1:-1, 0:-2]; F = p[1:-1, 2:]; H_ = p[2:, 1:-1]; E = p[1:-1, 1:-1]
    e0 = np.where((D == B) & (B != F) & (D != H_), D, E)
    e1 = np.where((B == F) & (B != D) & (F != H_), F, E)
    e2 = np.where((D == H_) & (D != B) & (H_ != F), D, E)
    e3 = np.where((H_ == F) & (D != H_) & (B != F), F, E)
    out = np.zeros((h * 2, w * 2), np.uint32)
    out[0::2, 0::2] = e0; out[0::2, 1::2] = e1; out[1::2, 0::2] = e2; out[1::2, 1::2] = e3
    return np.ascontiguousarray(out).view(np.uint8).reshape(h * 2, w * 2, 4)


def rotsprite(spr, angle_deg, pivot=None, pad=None):
    """RotSprite-style rotation: Scale2x x3 (8x), nearest rotate, sample block centres.
    Returns (rotated_sprite, (ox, oy)) where (ox, oy) is where the pivot lands in the
    new sprite (so blit at (px - ox, py - oy) to keep the pivot fixed)."""
    h, w = spr.shape[:2]
    if pivot is None:
        pivot = (w / 2.0, h / 2.0)
    if abs(angle_deg % 360) < 1e-6:
        return spr.copy(), pivot
    k = 8
    big = np.ascontiguousarray(spr)
    for _ in range(3):
        big = _scale2x(big)
    # output size: bounding box of rotated rect
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64) - np.array(pivot)
    rc = np.stack([corners[:, 0] * ca - corners[:, 1] * sa, corners[:, 0] * sa + corners[:, 1] * ca], 1)
    mn = np.floor(rc.min(0)) - 1
    mx = np.ceil(rc.max(0)) + 1
    ow, oh = int(mx[0] - mn[0]), int(mx[1] - mn[1])
    # sample at output pixel centres
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float64)
    X = xx + 0.5 + mn[0]
    Y = yy + 0.5 + mn[1]
    # inverse rotate into source space
    sx = X * ca + Y * sa + pivot[0]
    sy = -X * sa + Y * ca + pivot[1]
    bx = np.floor(sx * k).astype(int)
    by = np.floor(sy * k).astype(int)
    valid = (bx >= 0) & (by >= 0) & (bx < w * k) & (by < h * k)
    out = np.zeros((oh, ow, 4), np.uint8)
    out[valid] = big[by[valid], bx[valid]]
    return out, (-mn[0], -mn[1])


def outline(spr, color, diag=False):
    """Add 1px outline of `color` around opaque pixels (grows sprite by 1 each side)."""
    h, w = spr.shape[:2]
    out = np.zeros((h + 2, w + 2, 4), np.uint8)
    a = np.zeros((h + 2, w + 2), bool)
    a[1:-1, 1:-1] = spr[..., 3] > 127
    g = a.copy()
    g[1:, :] |= a[:-1, :]; g[:-1, :] |= a[1:, :]; g[:, 1:] |= a[:, :-1]; g[:, :-1] |= a[:, 1:]
    if diag:
        g[1:, 1:] |= a[:-1, :-1]; g[1:, :-1] |= a[:-1, 1:]; g[:-1, 1:] |= a[1:, :-1]; g[:-1, :-1] |= a[1:, 1:]
    ring = g & ~a
    out[ring, :3] = P[color] if isinstance(color, str) else color
    out[ring, 3] = 255
    out[1:-1, 1:-1][spr[..., 3] > 127] = spr[spr[..., 3] > 127]
    return out


def tint(spr, color, amount):
    out = spr.copy()
    c = np.asarray(P[color] if isinstance(color, str) else color, np.float32)
    out[..., :3] = np.clip(out[..., :3].astype(np.float32) * (1 - amount) + c * amount, 0, 255).astype(np.uint8)
    return out


def silhouette(spr, color):
    out = spr.copy()
    out[..., :3] = P[color] if isinstance(color, str) else color
    return out


def shear_rows(spr, offsets):
    """Shift each row horizontally by offsets[y] (int). Grows width to fit."""
    h, w = spr.shape[:2]
    offsets = np.asarray(offsets, int)
    mn, mx = min(0, offsets.min()), max(0, offsets.max())
    out = np.zeros((h, w + mx - mn, 4), np.uint8)
    for y in range(h):
        o = offsets[y] - mn
        out[y, o:o + w] = spr[y]
    return out, -mn


def fill_rect(canvas, x, y, w, h, color, alpha=1.0):
    H, W = canvas.shape[:2]
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x + w), W), min(int(y + h), H)
    if x0 >= x1 or y0 >= y1:
        return
    c = np.asarray(P[color] if isinstance(color, str) else color, np.float32)
    if alpha >= 1:
        canvas[y0:y1, x0:x1] = c
    else:
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - alpha) + c * alpha


def put(canvas, x, y, color, alpha=1.0):
    x = int(x); y = int(y)
    H, W = canvas.shape[:2]
    if 0 <= x < W and 0 <= y < H:
        c = np.asarray(P[color] if isinstance(color, str) else color, np.float32)
        canvas[y, x] = canvas[y, x] * (1 - alpha) + c * alpha


def line(canvas, x0, y0, x1, y1, color, alpha=1.0, dash=None, phase=0):
    """Bresenham line. dash=(on, off) for dotted lines."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    i = 0
    while True:
        if dash is None or ((i + phase) % (dash[0] + dash[1])) < dash[0]:
            put(canvas, x0, y0, color, alpha)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy
        i += 1


def circle_mask(r, filled=True):
    d = 2 * r + 1
    yy, xx = np.mgrid[0:d, 0:d] - r
    dist = np.sqrt(xx ** 2 + yy ** 2)
    if filled:
        return dist <= r + 0.3
    return (dist <= r + 0.5) & (dist > r - 0.5)


def ellipse_mask(rx, ry):
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    return (xx / (rx + 0.5)) ** 2 + (yy / (ry + 0.5)) ** 2 <= 1.0


def blit_mask(canvas, mask, x, y, color, alpha=1.0):
    x = int(round(x)); y = int(round(y))
    H, W = canvas.shape[:2]
    h, w = mask.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    c = np.asarray(P[color] if isinstance(color, str) else color, np.float32)
    dst = canvas[y0:y1, x0:x1]
    if alpha >= 1:
        dst[m] = c
    else:
        dst[m] = dst[m] * (1 - alpha) + c * alpha


BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16.0


def dither_mask(h, w, level, ox=0, oy=0):
    """Ordered-dither boolean mask: True with density `level` (0..1)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return BAYER4[(yy + oy) % 4, (xx + ox) % 4] < level


def to_uint8(canvas):
    return np.clip(canvas + 0.5, 0, 255).astype(np.uint8)


def upscale(img, k=4):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)
