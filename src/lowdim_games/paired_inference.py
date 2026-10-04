"""Inference over independent episodes paired across deterministic methods.

Inputs to the scalar API are already paired differences (method minus control).
No round is counted as an independent replicate, and curve bootstraps resample
complete episode rows with the same weights at every sampled time. Intervals
for curves are pointwise, not simultaneous confidence bands.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special, stats


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if parsed != value or parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return parsed


def _episodes(values, ndim):
    values = np.asarray(values, dtype=float)
    if values.ndim != ndim or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"At least two finite episode observations in a {ndim}D array are required.")
    if ndim == 2 and values.shape[1] < 1:
        raise ValueError("The sampled time grid must be nonempty.")
    return values


def _bootstrap_means(values, n_resamples, rng, batch_size=128):
    """Resample episode indices; matrix multiplication avoids a B*N*grid cube."""
    generator = np.random.default_rng(rng)
    n = len(values)
    result = np.empty((n_resamples,) + values.shape[1:], dtype=float)
    # Each row is a bootstrap draw of n complete paired episodes. Counts are
    # exactly the multiplicities of sampled row indices; all time points share
    # their counts. Limiting the batch bounds both weights and result temporaries.
    for start in range(0, n_resamples, batch_size):
        size = min(batch_size, n_resamples - start)
        indices = generator.integers(0, n, size=(size, n))
        offsets = np.arange(size, dtype=np.int64)[:, None] * n
        counts = np.bincount((indices + offsets).ravel(), minlength=size * n).reshape(size, n)
        result[start:start + size] = (counts / n) @ values
    return result


def _log_two_sided_t_p(log_abs_t, df, absolute_t):
    """Stable log p, including where SciPy's t survival function underflows.

    The fallback uses p=I_x(df/2,1/2), x=df/(df+t**2), and the incomplete-beta
    hypergeometric expression. It is used only in the small-tail regime where
    direct t.logsf returned -inf. log(x) avoids forming t**2.
    """
    log_p = float(math.log(2.0) + stats.t.logsf(absolute_t, df))
    if math.isfinite(log_p):
        return min(0.0, log_p)
    log_df = math.log(df)
    log_x = log_df - float(np.logaddexp(log_df, 2.0 * log_abs_t))
    x = math.exp(log_x)
    a = df / 2.0
    correction = float(special.hyp2f1(a, 0.5, a + 1.0, x))
    log_p = a * log_x - math.log(a) - float(special.betaln(a, 0.5)) + math.log(correction)
    if not math.isfinite(log_p):
        raise ValueError("The paired t tail could not be represented even on the log scale.")
    return min(0.0, log_p)


def paired_difference_summary(diff, bootstrap_resamples=9999, rng=0,
                              *, family_size=3, confidence_level=0.95):
    """JSON-safe paired mean inference, with negative differences favoring method.

    Family intervals use a Bonferroni allocation of 1-confidence_level across
    family_size comparisons; by default this is a 98.333333% marginal t interval
    for a three-comparison family. Holm p-values are computed separately.

    Constant samples have undefined t statistics and d_z, so p_value is None.
    Their empirical bootstrap/t-formula intervals collapse to a point and are
    explicitly flagged as degenerate rather than presented as exact certainty
    about the underlying population. Exact observed signs define win/tie/loss.
    """
    values = _episodes(diff, 1)
    bootstrap_resamples = _integer(bootstrap_resamples, "bootstrap_resamples", 2)
    family_size = _integer(family_size, "family_size")
    if not np.isscalar(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    confidence_level = float(confidence_level)
    n, df = len(values), len(values) - 1
    mean = float(np.mean(values))
    centered = values - mean
    scale = float(np.max(np.abs(centered)))
    # The scale normalization avoids squaring extremely small differences and
    # mistaking numerical variance underflow for a truly constant sample.
    constant = bool(np.all(values == values[0]))
    sd = (0.0 if constant else
          scale * math.sqrt(float(np.sum((centered / scale) ** 2)) / df))
    se = sd / math.sqrt(n)
    if not all(math.isfinite(x) for x in (mean, sd, se)):
        raise ValueError("Episode differences must permit finite mean and variance summaries.")
    family_level = 1.0 - (1.0 - confidence_level) / family_size
    critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df))
    family_critical = float(stats.t.ppf((1.0 + family_level) / 2.0, df))
    t_ci = [mean - critical * se, mean + critical * se]
    family_ci = [mean - family_critical * se, mean + family_critical * se]
    bootstrap = _bootstrap_means(values, bootstrap_resamples, rng)
    alpha = 1.0 - confidence_level
    bootstrap_ci = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist()
    bootstrap_family_ci = np.quantile(
        bootstrap, [alpha / (2.0 * family_size), 1.0 - alpha / (2.0 * family_size)]).tolist()
    if constant:
        statistic, effect, p_value, log_p, negative_log10_p = None, None, None, None, None
        underflow = False
        status = "constant_zero" if mean == 0.0 else "constant_nonzero"
    else:
        if mean == 0.0:
            statistic, effect, p_value, log_p, negative_log10_p = 0.0, 0.0, 1.0, 0.0, 0.0
        else:
            log_abs_t = math.log(abs(mean)) - math.log(sd) + 0.5 * math.log(n)
            abs_t = (math.exp(log_abs_t) if log_abs_t <= math.log(np.finfo(float).max)
                     else math.inf)
            statistic = math.copysign(abs_t, mean) if math.isfinite(abs_t) else None
            effect_value = mean / sd
            effect = float(effect_value) if math.isfinite(effect_value) else None
            log_p = _log_two_sided_t_p(log_abs_t, df, abs_t)
            p_value = math.exp(log_p)
            negative_log10_p = -log_p / math.log(10.0)
        underflow = bool(p_value == 0.0)
        status = "ok"
    wins, ties, losses = int(np.sum(values < 0)), int(np.sum(values == 0)), int(np.sum(values > 0))
    return {
        "n": int(n), "degrees_of_freedom": int(df),
        "mean_difference": mean, "sample_standard_deviation": sd,
        "standard_error": se, "t_statistic": statistic,
        "t_ci_95": [float(x) for x in t_ci],
        "t_ci_family": [float(x) for x in family_ci],
        "confidence_level": confidence_level, "family_size": family_size,
        "family_interval_confidence_level": family_level,
        "p_value": p_value, "log_p_value": log_p,
        "negative_log10_p_value": negative_log10_p,
        "p_value_underflow": underflow, "inference_status": status,
        "degenerate_empirical_intervals": constant,
        "d_z": effect,
        "bootstrap_mean_ci_95": [float(x) for x in bootstrap_ci],
        "bootstrap_mean_ci_family": [float(x) for x in bootstrap_family_ci],
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_method": "percentile, independent complete paired episodes",
        "wins": wins, "ties": ties, "losses": losses,
        "win_fraction": wins / n, "tie_fraction": ties / n, "loss_fraction": losses / n,
        "difference_direction": "method minus control; negative favors method",
        "inference_unit": "independent complete episode",
    }


def holm_adjust(pvalues):
    """Holm adjusted p-values in input order, counting every planned contrast.

    Undefined tests (None) count in the family and conservatively receive an
    adjusted value of one. Their raw p-values remain None in scalar summaries;
    this convention does not assert that an undefined test was actually run.
    """
    raw = list(pvalues)
    values = np.asarray([1.0 if value is None else value for value in raw], dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must be a 1D sequence of probabilities or None.")
    if len(values) == 0:
        return values
    order = np.argsort(values, kind="stable")
    ordered = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(ordered, 1.0)
    return adjusted


def bootstrap_mean_band(values, n_resamples=2000, rng=0,
                        *, confidence_level=0.95):
    """Mean and pointwise percentile limits for episode-by-time observations.

    Row resampling keeps every trajectory intact and uses identical sampled
    episodes for every time column. Supply paired difference rows to estimate a
    paired mean-difference curve. Same rng across calls gives matching weights.
    """
    values = _episodes(values, 2)
    n_resamples = _integer(n_resamples, "n_resamples", 2)
    if not np.isscalar(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    means = _bootstrap_means(values, n_resamples, rng)
    alpha = 1.0 - float(confidence_level)
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return np.mean(values, axis=0), lower, upper
