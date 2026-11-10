"""Seed-paired inference. Test examples are never treated as independent runs."""
import itertools
import numpy as np
from scipy import stats


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need >= 2 finite observations')
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(stats.t.ppf(.975, len(x) - 1) * sd / np.sqrt(len(x)))
    return {'mean': mean, 'sample_sd': sd, 'ci95_low': mean-half, 'ci95_high': mean+half}


def paired_stats(differences):
    d = np.asarray(differences, dtype=float)
    out = mean_ci(d)
    # Exact paired label swaps; absolute mean gives the two-sided statistic.
    null = np.asarray(list(itertools.product((-1, 1), repeat=len(d)))) @ d / len(d)
    out['p_signflip_exact'] = float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-12))
    if out['sample_sd'] == 0:
        out['p_t_approx'] = 1.0 if out['mean'] == 0 else 0.0
    else:
        out['p_t_approx'] = float(stats.ttest_1samp(d, 0).pvalue)
    out.update(n_seeds=len(d), wins=int((d > 1e-12).sum()),
               ties=int((abs(d) <= 1e-12).sum()), losses=int((d < -1e-12).sum()))
    return out


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p))))
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1)
    return out.tolist()


def probability_metrics(labels, preds, probabilities, bins=15):
    y, pred, p = np.asarray(labels), np.asarray(preds), np.asarray(probabilities)
    f1, recall = [], []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        f1.append(2*tp/max(2*tp+fp+fn, 1))
        recall.append(tp/max(tp+fn, 1))
    q = np.clip(p, 1e-12, 1-1e-12)
    confidence = np.maximum(p, 1-p)
    bin_ids = np.minimum((confidence*bins).astype(int), bins-1)
    ece = 0.0
    for b in range(bins):
        mask = bin_ids == b
        if mask.any():
            ece += mask.mean()*abs((pred[mask] == y[mask]).mean()-confidence[mask].mean())
    return {'accuracy': float((pred == y).mean()), 'macro_f1': float(np.mean(f1)),
            'balanced_accuracy': float(np.mean(recall)), 'brier': float(np.mean((p-y)**2)),
            'nll': float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))), 'ece15': float(ece)}
