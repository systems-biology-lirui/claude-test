"""v4 模型：同 session 无标签试次参与的 semi-supervised 解码。

cue（sample odor）：每个特征 view 先按 delay 时长中心化、在全部试次（有标签+无标签）上做 PCA，
再跑两类共享协方差的 GMM-EM，有标签试次的类别固定。最终取 5 个通用 view
加上 inner CV 最好的 3 个 view 的平均。

behavior：错误几乎全是 non-paired 试次上的 false alarm。delay 末尾约 -800 ms 处的神经活动
携带 test odor 信息，因此用 P(sample) 和 P(test) 得到 P(paired)，再结合本 session 的
miss / FA 率计算 P(correct)。只有 test odor 在 inner CV 中显著可解码 (p<0.01) 才启用，
否则全部预测 correct（此时这是 Bayes 最优）。
"""
import numpy as np
from scipy.stats import binomtest

from data import win

CORE = ('e05', 'e1', 'e15', 'traj', 'full')
CUE_KS = (15, 20, 30)
TEST_KS = (10, 20, 30)
TGRID = (.5, 1, 2, 4, 8, 16, np.inf)


# ---------- features ----------

def cue_views(a, q, min_units=30):
    def both(fn):
        return np.vstack([fn(a), fn(q)])
    base = {'e05': both(lambda d: win(d, 0, .5)),
            'e1': both(lambda d: win(d, 0, 1)),
            'e15': both(lambda d: win(d, 0, 1.5)),
            'mid': both(lambda d: win(d, 1, 3)),
            'full': both(lambda d: win(d, 0, 6)),
            'late': both(lambda d: win(d, -1, 0, True))}
    base['traj'] = np.hstack([both(lambda d: win(d, 0, .5)), both(lambda d: win(d, .5, 1.5)),
                              base['mid'], base['late']])
    views = dict(base)
    seen = []
    for lv in (5, 6, 7):
        reg = a['u_reg%d' % lv]
        for r in np.unique(reg):
            m = reg == r
            if r == '' or m.sum() < min_units or m.all() or any((m == s).all() for s in seen):
                continue
            seen.append(m)
            for w in ('e15', 'full', 'traj'):
                views[f'{w}_{r}'] = base[w][:, np.tile(m, base[w].shape[1] // len(m))]
    return views


def test_views(a, q):
    def tv(d):
        return {'t_bins': np.hstack([win(d, -.85, -.75, True), win(d, -.75, -.65, True),
                                     win(d, -.65, -.4, True), win(d, -.4, 0, True)]),
                't_fine': np.hstack([win(d, -.9 + .05 * i, -.85 + .05 * i, True) for i in range(6)])}
    x, y = tv(a), tv(q)
    return {k: np.vstack([x[k], y[k]]) for k in x}


# ---------- semi-supervised EM ----------

def center_by(X, g):
    X = X.astype(float).copy()
    for v in np.unique(g):
        X[g == v] -= X[g == v].mean(0)
    return X


def zs_all(X):
    sd = X.std(0)
    sd = np.maximum(sd, max(np.median(sd) * .3, 1e-6))
    return np.clip((X - X.mean(0)) / sd, -6, 6)


def pca_basis(X, kmax):
    Z = zs_all(X)
    Z = Z - Z.mean(0)
    u, s, _ = np.linalg.svd(Z, full_matrices=False)
    P = u[:, :kmax] * s[:kmax]
    return P / P.std(0)


def em(P, L, yL, iters=40, shrink=.3):
    """两类等先验、共享协方差的 GMM；L 为有标签行（类别固定）。返回所有行的 log-odds。"""
    n, k = P.shape
    m1, m0 = P[L][yL == 1].mean(0), P[L][yL == 0].mean(0)
    S = np.cov(P.T) * (1 - shrink) + shrink * np.eye(k)
    Si = np.linalg.inv(S)
    lo = P @ Si @ (m1 - m0) - .5 * (m1 @ Si @ m1 - m0 @ Si @ m0)
    for _ in range(iters):
        r = 1 / (1 + np.exp(-np.clip(lo, -30, 30)))
        r[L] = yL
        m1 = r @ P / r.sum()
        m0 = (1 - r) @ P / (n - r.sum())
        E1, E0 = P - m1, P - m0
        S = ((r[:, None] * E1).T @ E1 + ((1 - r)[:, None] * E0).T @ E0) / n
        S = (1 - shrink) * S + shrink * np.eye(k) * np.trace(S) / k
        Si = np.linalg.inv(S)
        new = P @ Si @ (m1 - m0) - .5 * (m1 @ Si @ m1 - m0 @ Si @ m0)
        done = np.max(np.abs(new - lo)) < 1e-4
        lo = new
        if done:
            break
    return lo


class ViewBank:
    """每个 (view, k) 的 PCA 只依赖无标签信息，可在所有 fold 间复用。"""

    def __init__(self, views, ks, delay):
        self.items = []
        for name, X in views.items():
            X = center_by(X, delay)
            kmax = min(max(ks), X.shape[1] - 1, X.shape[0] - 2)
            P = pca_basis(X, kmax)
            self.items += [(name, k, P[:, :k]) for k in ks if k <= kmax]

    def names(self):
        return sorted({n for n, _, _ in self.items})

    def squashed(self, L, yL):
        out, cnt = {}, {}
        for name, k, P in self.items:
            s = np.tanh(np.clip(em(P, L, yL), -30, 30) / 4)
            out[name] = out.get(name, 0) + s
            cnt[name] = cnt.get(name, 0) + 1
        return {n: out[n] / cnt[n] for n in out}

    def logodds(self, L, yL):
        return np.mean([np.clip(em(P, L, yL), -20, 20) for _, _, P in self.items], 0)


def folds(y, nf, seed):
    rng = np.random.default_rng(seed)
    f = np.zeros(len(y), int)
    off = 0
    for c in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == c))
        f[ids] = (off + np.arange(len(ids))) % nf
        off += len(ids)
    return f


def inner_quality(bank, L, yL, seed, nf=5, reps=2):
    names = bank.names()
    acc = {n: [] for n in names}
    marg = {n: [] for n in names}
    for r in range(reps):
        f = folds(yL, nf, seed + 1000 * r)
        pred = {n: np.zeros(len(L)) for n in names}
        for k in range(nf):
            tr = f != k
            s = bank.squashed(L[tr], yL[tr])
            for n in names:
                pred[n][~tr] = s[n][L[~tr]]
        for n in names:
            acc[n].append(((pred[n] > 0) == yL).mean())
            marg[n].append(np.mean((2 * yL - 1) * pred[n]))
    return {n: np.mean(acc[n]) + .1 * np.mean(marg[n]) for n in names}


def predict_cue(bank, L, yL, seed=0):
    s = bank.squashed(L, yL)
    q = inner_quality(bank, L, yL, seed)
    pick = sorted(set(CORE) | set(sorted(q, key=q.get, reverse=True)[:3]))
    return np.mean([s[n] for n in pick], 0), pick


# ---------- behavior ----------

def sig(x):
    return 1 / (1 + np.exp(-x))


def test_calibration(bank, L, yL, seed, reps=2):
    """inner CV：选 temperature（log-loss），并给出 test odor 解码的单侧 binomial p 值。"""
    held = []
    for r in range(reps):
        fo = folds(yL, 5, seed + 17 * r)
        h = np.zeros(len(L))
        for k in range(5):
            tr = fo != k
            h[~tr] = bank.logodds(L[tr], yL[tr])[L[~tr]]
        held.append(h)
    hh, yy = np.concatenate(held), np.tile(yL, reps)
    best, T = np.inf, np.inf
    for t in TGRID:
        p = np.full(len(hh), .5) if np.isinf(t) else np.clip(sig(hh / t), 1e-4, 1 - 1e-4)
        ll = -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
        if ll < best:
            best, T = ll, t
    acc = np.mean([((h > 0) == yL).mean() for h in held])
    p = binomtest(int(round(acc * len(L))), len(L), .5, alternative='greater').pvalue
    return T, acc, p


def predict_behavior(test_bank, L, cue, test, cor, cue_hat, seed=0, alpha=.01):
    """cue_hat: 所有行的 cue 预测 (0/1)，有标签行用真值。返回 (correct 预测, 诊断信息)。"""
    paired = cue[L] != test[L]
    miss = 1 - cor[L][paired].mean()
    fa = 1 - cor[L][~paired].mean()
    T, tacc, p = test_calibration(test_bank, L, test[L], seed)
    n = len(cue_hat)
    info = {'miss': miss, 'fa': fa, 'T': T, 'test_inner_acc': tacc, 'p': p, 'used': False}
    if p >= alpha or np.isinf(T):
        return np.full(n, cor[L].mean() >= .5), info
    pt = sig(test_bank.logodds(L, test[L]) / T)
    pc = cue_hat.astype(float)
    pp = pc * (1 - pt) + (1 - pc) * pt
    pcor = pp * (1 - miss) + (1 - pp) * (1 - fa)
    info['used'] = True
    return pcor >= .5, info
