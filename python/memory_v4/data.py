"""读取记忆赛道 NWB 文件，得到 delay 期 50 ms 分箱的 spike 计数。

所有 spike_times 都是相对 delay 起点的时间，范围 [0, delay_duration)。
"""
import glob
import os
import shutil
import tempfile
import zipfile

import h5py
import numpy as np

BIN = 0.05
NB = 120  # 6 s / 50 ms


def _strings(ds):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])


def read_nwb(fn):
    f = h5py.File(fn, 'r')
    tr = f['intervals/trials']
    u = f['units']
    d = {'name': os.path.basename(fn),
         'ch': int(os.path.basename(os.path.dirname(fn))[-1]),
         'id': tr['id'][:].astype(int),
         'delay': tr['delay_duration'][:].astype(float)}
    assert set(np.unique(d['delay'])) <= {3.0, 6.0}
    if 'sample_cue' in tr:
        d['cue'] = (tr['sample_cue'][:] == 8).astype(int)
        cor = tr['is_correct'][:]
        if cor.dtype.kind in 'SOU':
            cor = np.char.upper(_strings(tr['is_correct'])) == 'TRUE'
        d['cor'] = cor.astype(int)
        d['test'] = (tr['test_cue'][:] == 8).astype(int)
    for lv in range(3, 9):
        d['u_reg%d' % lv] = _strings(u['anatomic_level_%d' % lv])
    st = u['spike_times'][:]
    idx = u['spike_times_index'][:].astype(int)
    n, nu = len(d['id']), len(idx)
    row = np.full(d['id'].max() + 1, -1)
    row[d['id']] = np.arange(n)
    cnt = np.zeros((n, nu, NB), np.float32)
    lo = 0
    for j in range(nu):
        a = st[lo:idx[j]]
        lo = idx[j]
        r = row[a[:, 0].astype(int)]
        assert (r >= 0).all(), 'unknown trial id in spike record'
        t = a[:, 1]
        keep = (t >= 0) & (t < d['delay'][r])
        np.add.at(cnt, (r[keep], j, np.floor(t[keep] / BIN).astype(int)), 1)
    d['cnt'] = cnt
    f.close()
    return d


def read_all(pack):
    """pack: player_data.zip 或已解压的目录（包含 player_data/challenge*/）。"""
    tmp = None
    if os.path.isfile(pack):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(pack) as z:
            for m in z.namelist():
                if m.endswith('.nwb'):
                    z.extract(m, tmp)
        root = tmp
    else:
        root = pack
    try:
        files = sorted(glob.glob(os.path.join(root, 'player_data', 'challenge*', '*.nwb')))
        assert len(files) == 13, 'expected the 13 official memory NWB files'
        return {os.path.basename(f)[:-4]: read_nwb(f) for f in files}
    finally:
        if tmp:
            shutil.rmtree(tmp)


def cases(d):
    """四个评测文件，各自配对同 session 的有标签数据。"""
    out = []
    for name, q in d.items():
        if name.endswith('-eval'):
            cal, w = name[:-5] + '-train', .40
        elif name.endswith('-eval-2'):
            cal, w = name[:-7] + '-eval-1', .30 if q['ch'] == 2 else .15
        else:
            continue
        a = d[cal]
        assert len(a['u_reg5']) == len(q['u_reg5']) and (a['u_reg5'] == q['u_reg5']).all()
        out.append({'cal': a, 'qry': q, 'weight': w,
                    'stem': name.replace('-eval-2', '').replace('-eval', '')})
    out.sort(key=lambda c: -c['weight'])
    assert len(out) == 4 and abs(sum(c['weight'] for c in out) - 1) < 1e-9
    return out


def win(d, a, b, rel_end=False):
    """每个单元在 [a, b) 秒内的 sqrt 放电率；rel_end=True 时相对 delay 末尾。"""
    cnt = d['cnt']
    n, nu, _ = cnt.shape
    out = np.zeros((n, nu), np.float32)
    for i in range(n):
        D = int(round(d['delay'][i] / BIN))
        lo = int(round(a / BIN)) + (D if rel_end else 0)
        hi = int(round(b / BIN)) + (D if rel_end else 0)
        lo, hi = max(lo, 0), min(hi, D)
        out[i] = cnt[i, :, lo:hi].sum(1) / ((hi - lo) * BIN)
    return np.sqrt(out)
