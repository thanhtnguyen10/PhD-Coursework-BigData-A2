"""Score = (macro-F1 + micro-F1) / 2, as in the course evaluate.py, and the plug-in decision: per activity, the
top-m users, with m maximising the expected Score under the predicted probabilities (no labels needed)."""
import numpy as np


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def score(pred, truth):
    tp = ((pred == 1) & (truth == 1)).sum(0)
    fp = ((pred == 1) & (truth == 0)).sum(0)
    fn = ((pred == 0) & (truth == 1)).sum(0)
    per = [_f1(*c) for c in zip(tp, fp, fn)]
    macro, micro = float(np.mean(per)), _f1(tp.sum(), fp.sum(), fn.sum())
    return dict(score=(macro + micro) / 2, macro=macro, micro=micro, per_activity=per)


def expected_f1_predict(p, sweeps=3):
    n, k = p.shape
    cs = np.vstack([np.zeros(k), np.cumsum(-np.sort(-p, 0), 0)])  # cs[m, j] = expected TP of the top-m users
    m_ = np.arange(n + 1)[:, None]
    total = cs[-1]                                                # expected number of positives
    f1 = 2 * cs / (m_ + total + 1e-9)                             # expected F1 of activity j with the top-m users
    m = f1.argmax(0)
    for _ in range(sweeps):  # coordinate ascent on (macro + micro) / 2
        for j in range(k):
            o = [i for i in range(k) if i != j]
            tp_o = sum(cs[m[i], i] for i in o)
            fp_o = sum(m[i] - cs[m[i], i] for i in o)
            fn_o = sum(total[i] - cs[m[i], i] for i in o)
            macro = (sum(f1[m[i], i] for i in o) + f1[:, j]) / k
            tp = tp_o + cs[:, j]
            micro = 2 * tp / (2 * tp + fp_o + (m_[:, 0] - cs[:, j]) + fn_o + (total[j] - cs[:, j]) + 1e-9)
            m[j] = np.argmax(macro + micro)
    rank = np.argsort(np.argsort(-p, 0), 0)
    return (rank < m[None, :]).astype(int)
