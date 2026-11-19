"""Render the full video: frames (270x480) -> ffmpeg (nearest x4 -> 1080x1920, H.264).

usage: python3 src/render.py OUT.mp4 [--audio audio.wav] [--start S] [--end E] [--workers N] [--crf 14]
"""
import os
import sys
import time
import argparse
import subprocess
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import timeline as TL

_D = None


def _init():
    global _D
    from director import Director
    _D = Director()


def _render(i):
    import gfx
    t = i / TL.FPS
    c = _D.render(t)
    return i, gfx.to_uint8(c).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--audio', default=None)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=TL.DURATION)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--crf', type=int, default=14)
    ap.add_argument('--preset', default='slow')
    a = ap.parse_args()

    f0 = int(round(a.start * TL.FPS))
    f1 = int(round(a.end * TL.FPS))
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '270x480', '-r', str(TL.FPS), '-i', '-']
    if a.audio:
        cmd += ['-ss', f'{a.start:.4f}', '-t', f'{(f1 - f0) / TL.FPS:.4f}', '-i', a.audio]
    cmd += ['-vf', 'scale=1080:1920:flags=neighbor:out_color_matrix=bt709:out_range=tv,format=yuv420p',
            '-c:v', 'libx264', '-preset', a.preset, '-crf', str(a.crf), '-tune', 'animation',
            '-profile:v', 'high', '-g', '60', '-bf', '2',
            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    if a.audio:
        cmd += ['-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest']
    cmd += ['-movflags', '+faststart', '-r', str(TL.FPS), a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as pool:
        n = 0
        for i, buf in pool.imap(_render, range(f0, f1), chunksize=4):
            p.stdin.write(buf)
            n += 1
            if n % 60 == 0:
                el = time.time() - t0
                print(f'  frame {i}/{f1}  {n / el:.1f} fps', flush=True)
    p.stdin.close()
    p.wait()
    print(f'done {n} frames in {time.time() - t0:.1f}s -> {a.out}')


if __name__ == '__main__':
    main()
