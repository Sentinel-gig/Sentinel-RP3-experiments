"""
Experiment C: risk field recovery and risk conditioned detection.

Sentinel Working Paper No. 3.

This is a Monte Carlo simulation study. Every number it produces is a model
output, not a measurement of any deployed system or of any real city.

PART 1: FIELD RECOVERY
A synthetic ground truth intensity is defined over a tile: corridor shaped
ridges plus a point hotspot over a uniform floor, with self exciting
(Hawkes style) triggering. Twelve months of incidents are generated. An
observation process then degrades them the way real reporting does: spatially
biased under reporting, and Gaussian geocoding noise of the magnitude a
landmark based resolution chain produces. Fields are fitted to the DEGRADED
observations and evaluated against the withheld final month, using the
predictive accuracy index (share of incidents captured in the top k percent of
cells, divided by k percent).

The question this answers: can a risk field built from extraction quality data
outperform the district aggregate resolution that public data actually offers?

PART 2: RISK CONDITIONED DETECTION
Sensor anomalies are simulated with a location dependent probability of
reflecting a true incident. Two decision rules are compared: a fixed threshold
on the sensor score, and a Bayesian rule that adds the log prior odds from the
RECOVERED field (not the ground truth field, since a deployed system would only
have the recovered one). Both are evaluated at matched recall.

The question this answers: how much does a spatial prior buy, and where?
"""
import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------ parameters
GRID = 40                 # cells per side
CELL_M = 100.0            # cell size in metres, so a 4 km by 4 km tile
MONTHS = 12
BASE_EVENTS_PER_MONTH = 140
BRANCHING_RATIO = 0.30    # Hawkes triggering: expected offspring per event
TRIGGER_SIGMA_CELLS = 1.5 # spatial spread of triggered events, in cells
REPORT_P_HIGH = 0.55      # reporting probability outside the under reported quadrant
REPORT_P_LOW = 0.35       # reporting probability inside it
GEOCODE_NOISE_CELLS = 1.5 # 150 m at 100 m cells
KDE_BANDWIDTH_CELLS = 2.0
ETAS_RECENT_WEIGHT = 0.25
ETAS_RECENT_MONTHS = 2
TOP_FRACTION = 0.05       # PAI evaluated at the top 5 percent of cells

# detection experiment
N_ANOMALIES = 40000
PRIOR_MIX = 0.7           # anomalies skew toward risky cells but occur everywhere
PI_SCALE = 4.0            # maps relative intensity to P(true incident | anomaly)
SENSOR_CONDITIONS = {'weak_sensor': 1.2, 'strong_sensor': 2.0}   # d prime
RECALL_TARGETS = (0.80, 0.90)


def ground_truth_intensity():
    """Corridor ridges plus a hotspot over a uniform floor. Deterministic."""
    yy, xx = np.mgrid[0:GRID, 0:GRID]

    def ridge(x0, y0, x1, y1, width, amp):
        px, py = xx - x0, yy - y0
        vx, vy = x1 - x0, y1 - y0
        t = np.clip((px * vx + py * vy) / (vx * vx + vy * vy), 0, 1)
        d2 = (px - t * vx) ** 2 + (py - t * vy) ** 2
        return amp * np.exp(-d2 / (2 * width * width))

    mu = 0.15
    mu = mu + ridge(2, 8, 38, 12, 1.6, 1.0)
    mu = mu + ridge(20, 0, 24, 39, 1.8, 0.9)
    mu = mu + ridge(5, 30, 35, 34, 1.5, 0.7)
    mu = mu + 0.8 * np.exp(-(((xx - 30) ** 2 + (yy - 8) ** 2)) / (2 * 2.5 ** 2))
    return mu / mu.sum()


def generate_events(mu, rng):
    """Thin a Poisson process against the intensity, then add one generation of
    Hawkes style triggering."""
    flat = mu.ravel()
    events = []
    for m in range(MONTHS):
        n = rng.poisson(BASE_EVENTS_PER_MONTH)
        idx = rng.choice(GRID * GRID, size=n, p=flat)
        parents = [(m, i % GRID + rng.uniform(-0.5, 0.5), i // GRID + rng.uniform(-0.5, 0.5))
                   for i in idx]
        children = []
        for (mm, x, y) in parents:
            for _ in range(rng.poisson(BRANCHING_RATIO)):
                children.append((mm + int(rng.random() < 0.5),
                                 x + rng.normal(0, TRIGGER_SIGMA_CELLS),
                                 y + rng.normal(0, TRIGGER_SIGMA_CELLS)))
        events.extend(parents)
        events.extend([c for c in children
                       if 0 <= c[1] < GRID and 0 <= c[2] < GRID and c[0] < MONTHS])
    return np.array(events)


def observe(events, rng):
    """Spatially biased under reporting plus geocoding noise."""
    p = np.where((events[:, 1] < GRID / 2) & (events[:, 2] < GRID / 2),
                 REPORT_P_LOW, REPORT_P_HIGH)
    obs = events[rng.random(len(events)) < p].copy()
    obs[:, 1] += rng.normal(0, GEOCODE_NOISE_CELLS, len(obs))
    obs[:, 2] += rng.normal(0, GEOCODE_NOISE_CELLS, len(obs))
    keep = (obs[:, 1] >= 0) & (obs[:, 1] < GRID) & (obs[:, 2] >= 0) & (obs[:, 2] < GRID)
    return obs[keep]


def fit_fields(obs):
    """KDE and an ETAS style variant that upweights recent months."""
    train = obs[obs[:, 0] < MONTHS - 1]
    H = np.zeros((GRID, GRID))
    for (_, x, y) in train:
        H[int(y), int(x)] += 1
    kde = gaussian_filter(H, KDE_BANDWIDTH_CELLS)
    kde = kde / kde.sum()

    recent = obs[(obs[:, 0] >= MONTHS - 1 - ETAS_RECENT_MONTHS) & (obs[:, 0] < MONTHS - 1)]
    Hr = np.zeros((GRID, GRID))
    for (_, x, y) in recent:
        Hr[int(y), int(x)] += 1
    if Hr.sum() > 0:
        etas = (1 - ETAS_RECENT_WEIGHT) * kde + ETAS_RECENT_WEIGHT * gaussian_filter(Hr, 1.5) / Hr.sum()
    else:
        etas = kde.copy()
    etas = etas / etas.sum()
    return kde, etas


def pai(model, test_events, frac=TOP_FRACTION):
    """Predictive accuracy index: hit rate in the top cells over the area share."""
    k = int(GRID * GRID * frac)
    top = np.argsort(model.ravel())[::-1][:k]
    mask = np.zeros(GRID * GRID, dtype=bool)
    mask[top] = True
    hits = sum(1 for (_, x, y) in test_events if mask[int(y) * GRID + int(x)])
    hit_rate = hits / len(test_events)
    return hit_rate, hit_rate / frac


def detection_experiment(mu, recovered, rng):
    """Fixed threshold versus a rule that adds the log prior odds from the
    RECOVERED field, which is what a deployed system would actually have."""
    flat_true = mu.ravel() / mu.ravel().sum()
    out = {}
    for label, d_prime in SENSOR_CONDITIONS.items():
        cells = rng.choice(GRID * GRID, size=N_ANOMALIES,
                           p=flat_true * PRIOR_MIX + (1 - PRIOR_MIX) / (GRID * GRID))
        lam_true = (mu.ravel() / mu.ravel().mean())[cells]
        pi_true = lam_true / (lam_true + PI_SCALE)
        is_true = rng.random(N_ANOMALIES) < pi_true
        s = np.where(is_true, rng.normal(d_prime, 1.0, N_ANOMALIES),
                     rng.normal(0.0, 1.0, N_ANOMALIES))

        lam_hat = (recovered.ravel() / recovered.ravel().mean())[cells]
        pi_hat = np.clip(lam_hat / (lam_hat + PI_SCALE), 1e-6, 1 - 1e-6)
        posterior = np.log(pi_hat / (1 - pi_hat)) + s * d_prime - d_prime * d_prime / 2

        def fpr_at_recall(score, target):
            ts = np.sort(score[is_true])
            thr = ts[int((1 - target) * len(ts))]
            return float((score[~is_true] >= thr).mean())

        def auc(score):
            ranks = np.empty(N_ANOMALIES)
            ranks[np.argsort(score)] = np.arange(1, N_ANOMALIES + 1)
            n1 = int(is_true.sum())
            n0 = N_ANOMALIES - n1
            return float((ranks[is_true].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

        res = {'d_prime': d_prime, 'auc_fixed': auc(s), 'auc_risk_conditioned': auc(posterior),
               'n_true': int(is_true.sum())}
        for t in RECALL_TARGETS:
            f1, f2 = fpr_at_recall(s, t), fpr_at_recall(posterior, t)
            key = int(t * 100)
            res[f'fpr_fixed_{key}'] = f1
            res[f'fpr_risk_conditioned_{key}'] = f2
            res[f'reduction_pct_{key}'] = 100.0 * (1 - f2 / f1) if f1 > 0 else None
        out[label] = res
    return out


def environment_record():
    import scipy
    return {'generated_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version.split()[0], 'numpy': np.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--replicates', type=int, default=20,
                    help='independent replications, for uncertainty intervals')
    args = ap.parse_args()

    mu = ground_truth_intensity()
    reps = []
    print(f'running {args.replicates} independent replications', flush=True)
    for r in range(args.replicates):
        rng = np.random.default_rng(args.seed + r)
        events = generate_events(mu, rng)
        obs = observe(events, rng)
        kde, etas = fit_fields(obs)
        test = events[events[:, 0] == MONTHS - 1]
        uniform = np.ones((GRID, GRID)) / (GRID * GRID)
        rec = {
            'n_true_events': int(len(events)),
            'n_observed': int(len(obs)),
            'reporting_fraction': float(len(obs) / len(events)),
            'pai_uniform': pai(uniform, test)[1],
            'pai_kde': pai(kde, test)[1],
            'pai_etas': pai(etas, test)[1],
            'hit_rate_etas': pai(etas, test)[0],
            'corr_truth_recovered': float(np.corrcoef(mu.ravel(), etas.ravel())[0, 1]),
            'detection': detection_experiment(mu, etas, rng),
        }
        reps.append(rec)
        if r == 0:
            np.save(os.path.join('results', 'mu_true.npy'), mu)
            np.save(os.path.join('results', 'mu_recovered.npy'), etas)
        print(f"  rep {r+1:2d}: reporting {100*rec['reporting_fraction']:.0f}%  "
              f"PAI etas {rec['pai_etas']:.2f}  corr {rec['corr_truth_recovered']:.3f}",
              flush=True)

    def summarize(key_fn):
        vals = np.array([key_fn(r) for r in reps], dtype=float)
        return {'mean': float(vals.mean()), 'sd': float(vals.std(ddof=1)),
                'ci95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                'min': float(vals.min()), 'max': float(vals.max())}

    summary = {
        'reporting_fraction': summarize(lambda r: r['reporting_fraction']),
        'pai_uniform': summarize(lambda r: r['pai_uniform']),
        'pai_kde': summarize(lambda r: r['pai_kde']),
        'pai_etas': summarize(lambda r: r['pai_etas']),
        'corr_truth_recovered': summarize(lambda r: r['corr_truth_recovered']),
    }
    for cond in SENSOR_CONDITIONS:
        for t in RECALL_TARGETS:
            k = int(t * 100)
            summary[f'{cond}_reduction_pct_{k}'] = summarize(
                lambda r, c=cond, kk=k: r['detection'][c][f'reduction_pct_{kk}'])
        summary[f'{cond}_auc_fixed'] = summarize(lambda r, c=cond: r['detection'][c]['auc_fixed'])
        summary[f'{cond}_auc_rc'] = summarize(lambda r, c=cond: r['detection'][c]['auc_risk_conditioned'])

    out = {'meta': {'seed': args.seed, 'replicates': args.replicates,
                    'parameters': {k: v for k, v in globals().items()
                                   if k.isupper() and isinstance(v, (int, float, str, dict, tuple))},
                    'environment': environment_record()},
           'replications': reps, 'summary': summary}
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'experiment_c.json'), 'w') as f:
        json.dump(out, f, indent=1)

    print('\n=== summary across replications ===')
    print(f"reporting fraction     {summary['reporting_fraction']['mean']:.3f}")
    print(f"PAI district uniform   {summary['pai_uniform']['mean']:.2f} "
          f"[{summary['pai_uniform']['ci95'][0]:.2f}, {summary['pai_uniform']['ci95'][1]:.2f}]")
    print(f"PAI KDE                {summary['pai_kde']['mean']:.2f} "
          f"[{summary['pai_kde']['ci95'][0]:.2f}, {summary['pai_kde']['ci95'][1]:.2f}]")
    print(f"PAI ETAS style         {summary['pai_etas']['mean']:.2f} "
          f"[{summary['pai_etas']['ci95'][0]:.2f}, {summary['pai_etas']['ci95'][1]:.2f}]")
    print(f"corr(truth, recovered) {summary['corr_truth_recovered']['mean']:.3f}")
    for cond in SENSOR_CONDITIONS:
        r80 = summary[f'{cond}_reduction_pct_80']
        r90 = summary[f'{cond}_reduction_pct_90']
        print(f"{cond:14s} false alert reduction  80% recall {r80['mean']:.1f}% "
              f"[{r80['ci95'][0]:.1f}, {r80['ci95'][1]:.1f}]   "
              f"90% recall {r90['mean']:.1f}%")
    print('\nwrote results/experiment_c.json')


if __name__ == '__main__':
    main()
