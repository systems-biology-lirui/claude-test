"""记忆赛道 v4：semi-supervised cue 解码 + test-odor 门控的 behavior 预测。

用法（在仓库根目录）：
    python python/memory_v4/run_v4.py --pack player_data.zip --out output/memory_v4
"""
import argparse
import io
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import read_all, cases as make_cases  # noqa: E402
from model import (ViewBank, cue_views, test_views, predict_cue, predict_behavior,  # noqa: E402
                   folds, CUE_KS, TEST_KS)

TEAM = '谁说bio不如com'
G = {}


def setup(c):
    a, q = c['cal'], c['qry']
    na, nq = len(a['id']), len(q['id'])
    dl = np.r_[a['delay'], q['delay']]
    z = np.zeros(nq, int)
    return {'cue_bank': ViewBank(cue_views(a, q), CUE_KS, dl),
            'test_bank': ViewBank(test_views(a, q), TEST_KS, dl),
            'cue': np.r_[a['cue'], z], 'test': np.r_[a['test'], z], 'cor': np.r_[a['cor'], z],
            'na': na, 'nq': nq}


def fit(g, L, seed):
    s, pick = predict_cue(g['cue_bank'], L, g['cue'][L], seed)
    cue_hat = (s > 0).astype(int)
    cue_hat[L] = g['cue'][L]
    beh, info = predict_behavior(g['test_bank'], L, g['cue'], g['test'], g['cor'], cue_hat, seed)
    return cue_hat, beh, pick, info


def outer_job(job):
    ci, seed, k = job
    g = G[ci]
    y = g['cor'][:g['na']] * 2 + g['cue'][:g['na']]
    fo = folds(y, 5, seed)
    L, ev = np.flatnonzero(fo != k), np.flatnonzero(fo == k)
    cue_hat, beh, pick, info = fit(g, L, seed * 10 + k)
    prior = g['cor'][L].mean() >= .5
    return ci, seed, ev, cue_hat[ev], beh[ev].astype(int), np.full(len(ev), int(prior)), info


def final_job(ci):
    g = G[ci]
    L = np.arange(g['na'])
    cue_hat, beh, pick, info = fit(g, L, 0)
    q = slice(g['na'], None)
    return ci, cue_hat[q], beh[q].astype(int), pick, info


def write_zip(cs, preds, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, TEAM + '.zip')
    tmp = path + '.part'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for c, (cue, beh) in zip(cs, preds):
            ids = c['qry']['id']
            order = np.argsort(ids)
            buf = io.BytesIO()
            with h5py.File(buf, 'w') as f:
                f.create_dataset('trial_indices', data=ids[order].astype(np.int64))
                f.create_dataset('mem_pred_lbl', data=(4 + 4 * cue[order]).astype(np.int64))
                f.create_dataset('corr_pred_lbl', data=beh[order].astype(np.int64))
            z.writestr(f"{TEAM}_{c['stem']}.h5", buf.getvalue())
    # 检查 ZIP 内容后再替换正式文件
    with zipfile.ZipFile(tmp) as z:
        names = z.namelist()
        assert len(names) == 4
        for c, n in zip(cs, names):
            with h5py.File(io.BytesIO(z.read(n)), 'r') as f:
                assert set(f) == {'trial_indices', 'mem_pred_lbl', 'corr_pred_lbl'}
                assert len(f['trial_indices']) == len(c['qry']['id'])
                assert set(np.unique(f['mem_pred_lbl'][:])) <= {4, 8}
                assert set(np.unique(f['corr_pred_lbl'][:])) <= {0, 1}
    os.replace(tmp, path)
    return path


def read_previous(path, stem):
    if not os.path.isfile(path):
        return None
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith(stem + '.h5'):
                with h5py.File(io.BytesIO(z.read(n)), 'r') as f:
                    return f['mem_pred_lbl'][:] == 8, f['corr_pred_lbl'][:]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='player_data.zip')
    ap.add_argument('--out', default=os.path.join('output', 'memory_v4'))
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--previous', default=os.path.join('output', 'memory_v3', TEAM + '.zip'))
    args = ap.parse_args()
    t0 = time.time()
    cs = make_cases(read_all(args.pack))
    for ci, c in enumerate(cs):
        G[ci] = setup(c)
        print(f"{c['stem']}: {G[ci]['na']} labeled, {G[ci]['nq']} query, "
              f"{len(G[ci]['cue_bank'].names())} cue views", flush=True)
    jobs = [(ci, 200 + s, k) for ci in G for s in range(3 if G[ci]['na'] > 100 else 6) for k in range(5)]
    with Pool(args.workers) as p:
        outer = p.map(outer_job, jobs)
        final = p.map(final_job, list(G))

    lines = ['Memory track v4: semi-supervised cue decoding + test-odor-gated behavior.',
             'Outer CV: stratified 5-fold on the labeled file (train for ch1, eval-1 for ch2/ch3),',
             'repeated 3x (ch1) or 6x (ch2/ch3). Unlabeled trials (query file + held-out fold) join the EM;',
             'labels of held-out trials are never used. Score = mean of cue and behavior accuracy.', '']
    total = {'cue': 0, 'beh': 0, 'prior': 0}
    preds = []
    for ci, c in enumerate(cs):
        rs = [r for r in outer if r[0] == ci]
        g = G[ci]
        cue_acc, beh_acc, pri_acc, used = [], [], [], []
        for seed in sorted({r[1] for r in rs}):
            ev = np.concatenate([r[2] for r in rs if r[1] == seed])
            ch = np.concatenate([r[3] for r in rs if r[1] == seed])
            bh = np.concatenate([r[4] for r in rs if r[1] == seed])
            pr = np.concatenate([r[5] for r in rs if r[1] == seed])
            cue_acc.append((ch == g['cue'][ev]).mean())
            beh_acc.append((bh == g['cor'][ev]).mean())
            pri_acc.append((pr == g['cor'][ev]).mean())
        used = np.mean([r[6]['used'] for r in rs])
        _, qc, qb, pick, info = final[ci]
        preds.append((qc, qb))
        w = c['weight']
        total['cue'] += w * np.mean(cue_acc)
        total['beh'] += w * np.mean(beh_acc)
        total['prior'] += w * np.mean(pri_acc)
        lines += [f"{c['stem']}  (weight {w:.2f}, {g['na']} labeled, {g['nq']} predicted)",
                  f"  outer CV cue {np.mean(cue_acc):.4f}  per repeat {np.round(cue_acc, 4).tolist()}",
                  f"  outer CV behavior {np.mean(beh_acc):.4f} (all-correct prior {np.mean(pri_acc):.4f}); "
                  f"test-odor rule used in {used:.0%} of outer folds",
                  f"  final cue views: {', '.join(pick)}",
                  f"  final behavior: rule {'ON' if info['used'] else 'OFF (all correct)'}; test inner acc "
                  f"{info['test_inner_acc']:.3f}, p={info['p']:.2g}, T={info['T']}, "
                  f"miss {info['miss']:.2f}, FA {info['fa']:.2f}",
                  f"  query predictions: cue8 fraction {qc.mean():.3f}, predicted correct {qb.mean():.3f}"]
        prev = read_previous(args.previous, c['stem'])
        if prev is not None:
            order = np.argsort(c['qry']['id'])
            lines.append(f"  agreement with v3 on query: cue {np.mean(prev[0] == qc[order]):.3f}, "
                         f"behavior {np.mean(prev[1] == qb[order]):.3f}")
        lines.append('')
    score = .5 * (total['cue'] + total['beh'])
    lines += [f"Weighted outer-CV: cue {total['cue']:.4f}, behavior {total['beh']:.4f} "
              f"(prior {total['prior']:.4f}); score {score:.4f}",
              f"Runtime {time.time() - t0:.0f} s with {args.workers} workers."]
    path = write_zip(cs, preds, args.out)
    with open(os.path.join(args.out, 'validation.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Wrote', path)


if __name__ == '__main__':
    main()
