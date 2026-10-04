"""Full realized-hull target for the resource impulse game.

The origin (no demand) must have been observed. The other observed opponent
actions are distinct unit requests e_i. With epsilon=1/sqrt(2), inactive
payoff (epsilon*r, epsilon*ell, 0), active energy gamma0-(gamma0-gamma1)*r,
and the pure minimum equal-weight-loss response, put
    r_star = gamma0 / (2*epsilon + gamma0-gamma1), beta = 2*epsilon*r_star.
This module assumes 0 < r_star < 1 and the supplied beta from that game.

The *entire* hull contains r*q for 0<=r<=1 and mixtures q of visited requests.
Inactive response images approach (beta/2)*(e_aggregate+e_i) at the tie;
active response images are on the energy axis up to beta. Tie closure and
convexification therefore give, after a first request,
    S(Q) = { (A,w,E): w>=0, sum(w)=A, E>=0, 2*A+E<=beta },
with w supported on observed resources. Before any request S(Q)={0}.
This proof covers unobserved interior mixtures, not just endpoint responses.

All realized prefix means in this game satisfy A=sum(w), w>=0 and E>=0.
Dropping resource coordinates gives a 2D triangle and a distance lower bound.
Scaling the actual resource vector to the projected aggregate gives a
feasible point in the full target and an upper bound. Exact projection uses
the KKT water-filling equations, with a directly recomputed full-target
support gap. No approximate oracle or empirical target is substituted.
"""
from __future__ import annotations

import numpy as np


def _beta(beta):
    value = float(beta)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("beta must be finite and positive.")
    return value


def target_bounds(aggregate, energy, resource_sq_norm, observed_count, beta):
    """Vectorized certified distances enclosing the full-target distance.

    aggregate and energy are *prefix means*. resource_sq_norm is sum_i w_i^2
    for the resource prefix means, not cumulative sums. observed_count counts
    previously revealed request types, including fully served zero deficits.
    Scalar inputs return NumPy scalar-shaped outputs; array inputs broadcast.
    This evaluator requires the previously observed origin and A=sum_i w_i.
    """
    beta = _beta(beta)
    A, E, sq, count = np.broadcast_arrays(
        np.asarray(aggregate, dtype=float), np.asarray(energy, dtype=float),
        np.asarray(resource_sq_norm, dtype=float), np.asarray(observed_count))
    if (not np.isfinite(A).all() or not np.isfinite(E).all()
            or not np.isfinite(sq).all() or not np.isfinite(count).all()
            or np.any(A < 0) or np.any(E < 0) or np.any(sq < 0)
            or np.any(count < 0) or np.any(count != np.floor(count))):
        raise ValueError("Prefix means, squared norm and request counts must be finite and nonnegative; counts are integers.")
    if np.any((count == 0) & ((A != 0) | (sq != 0))):
        raise ValueError("No observed request requires zero aggregate and resource squared norm.")
    # Nonnegative resource coordinates with sum A have these norm bounds.
    tol = 2e-10 * np.maximum(1.0, A*A)
    if np.any(sq > A*A + tol):
        raise ValueError("resource_sq_norm is incompatible with nonnegative resources summing to aggregate.")
    lower_sq = np.divide(A*A, count, out=np.zeros_like(A), where=count > 0)
    if np.any((count > 0) & (sq + tol < lower_sq)):
        raise ValueError("resource_sq_norm is below the count-constrained resource norm.")

    membership = (count > 0) & (2*A + E <= beta)
    # Projection onto the oblique triangle edge, clipped to either endpoint.
    position = np.clip((-0.5*A + 0.25*beta + E)/(1.25*beta), 0.0, 1.0)
    candidate_A = np.where(membership, A, 0.5*beta*(1-position))
    candidate_E = np.where(membership, E, beta*position)
    no_request = count == 0
    candidate_A = np.where(no_request, 0.0, candidate_A)
    candidate_E = np.where(no_request, 0.0, candidate_E)
    lower = np.hypot(A-candidate_A, E-candidate_E)
    ratio = np.divide(candidate_A, A, out=np.zeros_like(A), where=A > 0)
    upper = np.sqrt(lower*lower + (1-ratio)**2*sq)
    upper = np.maximum(upper, lower)
    exact_zero = membership | (no_request & (E == 0))
    return {
        "lower": lower, "upper": upper, "width": upper-lower,
        "candidate_aggregate": candidate_A, "candidate_energy": candidate_E,
        "candidate_A": candidate_A, "candidate_E": candidate_E,
        "exact_membership": exact_zero, "exact_zero": exact_zero,
        "exactflag": no_request | (upper == lower),
    }


def exact_target_projection(raw_resource_payoffs, cumulative_aggregate,
                            cumulative_energy, t, beta, observed_count=None,
                            *, counts=None, return_resource_projection=False):
    """Exact full-target prefix projection using raw resource cumulative sums.

    The array may contain one value per resource or weighted histogram levels:
    pass integer multiplicities in counts for the latter. Zero coordinates may
    be omitted, but observed_count must still count all revealed request types.
    When observed_count is omitted, represented multiplicities are the count.
    This is a numerical analytical oracle: returned feasibility_error and
    support_gap expose floating arithmetic; they are not formal certificates.
    Sorting costs O(m log m) for m array values or histogram levels. Thus exact
    terminal/landmark evaluation avoids all-round sorting of long episodes.
    """
    beta = _beta(beta)
    if isinstance(t, bool) or int(t) != t or t < 1:
        raise ValueError("t must be a positive integer.")
    t = int(t)
    raw = np.asarray(raw_resource_payoffs, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("raw_resource_payoffs must be a finite nonnegative vector.")
    if counts is None:
        multiplicity = np.ones(raw.shape, dtype=np.int64)
    else:
        provided = np.asarray(counts)
        if (provided.shape != raw.shape or not np.isfinite(provided).all()
                or np.any(provided < 0) or np.any(provided != np.floor(provided))):
            raise ValueError("counts must be nonnegative integers matching resource values.")
        multiplicity = provided.astype(np.int64)
    if observed_count is None:
        observed_count = int(np.sum(multiplicity))
    if (isinstance(observed_count, bool) or int(observed_count) != observed_count
            or observed_count < 0):
        raise ValueError("observed_count must be a nonnegative integer.")
    observed_count = int(observed_count)
    positive = (raw > 0) & (multiplicity > 0)
    if int(np.sum(multiplicity[positive])) > observed_count:
        raise ValueError("More positive resource coordinates than observed requests.")
    A, E = float(cumulative_aggregate)/t, float(cumulative_energy)/t
    if not np.isfinite(A) or not np.isfinite(E) or A < 0 or E < 0:
        raise ValueError("Cumulative aggregate and energy must be finite and nonnegative.")
    w = raw/t
    represented_A = float(np.dot(w, multiplicity))
    if abs(A-represented_A) > 2e-10*max(1.0, A):
        raise ValueError("Cumulative aggregate must equal the represented resource sum.")
    if observed_count == 0 and (A != 0 or np.any(positive)):
        raise ValueError("No observed request requires zero aggregate and resource payoffs.")
    sq = float(np.dot(w*w, multiplicity))

    def result(Ap, Ep, threshold, multiplier):
        wp = np.maximum(w-threshold, 0.0) if observed_count else np.zeros_like(w)
        # Compute the actual projected aggregate from coordinates, independently
        # of the threshold equations, so feasibility is directly auditable.
        if observed_count:
            Ap = float(np.dot(wp, multiplicity))
        dA, dE = A-Ap, E-Ep
        diff = w-wp
        distance_sq = dA*dA+dE*dE+float(np.dot(diff*diff, multiplicity))
        if observed_count:
            max_diff = float(np.max(diff[multiplicity > 0], initial=0.0))
            support = max(0.0, beta*(dA+max_diff)/2, beta*dE)
            dot_projection = dA*Ap+dE*Ep+float(np.dot(diff*wp, multiplicity))
            gap = max(0.0, 2*(support-dot_projection))
            feasibility = max(0.0, -Ap, -Ep, 2*Ap+Ep-beta,
                              float(-np.min(wp, initial=0.0)))
        else:
            gap, feasibility = 0.0, max(abs(Ap), abs(Ep), float(np.max(np.abs(wp), initial=0.0)))
        output = {
            "distance": float(np.sqrt(max(0.0, distance_sq))),
            "distance_squared": float(max(0.0, distance_sq)),
            "aggregate": Ap, "energy": float(Ep),
            "threshold": float(threshold), "multiplier": float(multiplier),
            "support_gap": float(gap), "projection_gap": float(gap),
            "feasibility_error": float(feasibility),
            "observed_count": observed_count,
        }
        if return_resource_projection:
            output["resource_projection"] = wp
        return output

    if observed_count == 0:
        return result(0.0, 0.0, 0.0, 0.0)
    if 2*A+E <= beta:
        return result(A, E, 0.0, 0.0)
    if not np.any(positive):
        return result(0.0, min(E, beta), 0.0, max(E-beta, 0.0))

    values = w[positive]
    weights = multiplicity[positive]
    order = np.argsort(-values, kind="stable")
    values, weights = values[order], weights[order]
    sizes = np.cumsum(weights).astype(float)
    sums = np.cumsum(values*weights)
    next_values = np.r_[values[1:], 0.0]
    tolerance = 8*np.finfo(float).eps*max(1.0, A, E, beta)

    # k=0: projection is the energy corner and every resource is clipped.
    multiplier = E-beta
    threshold = 2*multiplier-A
    if multiplier >= 0 and threshold >= values[0]-tolerance:
        return result(0.0, beta, max(threshold, 0.0), multiplier)

    # KKT: w'_i=(w_i-z)_+, A'=sum(w'), E'=(E-mu)_+,
    # z=A'-A+2*mu and 2*A'+E'=beta. For each possible active
    # group these equations are linear; weighted histograms preserve them.
    mu = (2*sums+2*sizes*A+(sizes+1)*(E-beta))/(5*sizes+1)
    z = (sums-A+2*mu)/(sizes+1)
    valid = ((mu >= -tolerance) & (mu <= E+tolerance)
             & (z >= -tolerance) & (values >= z-tolerance)
             & (z >= next_values-tolerance))
    indices = np.flatnonzero(valid)
    if len(indices):
        i = int(indices[0])
        return result(0.0, max(E-float(mu[i]), 0.0), max(float(z[i]), 0.0), max(float(mu[i]), 0.0))

    # Energy-clipped boundary: A'=beta/2, E'=0.
    z = (sums-beta/2)/sizes
    mu = (z-beta/2+A)/2
    valid = ((mu >= E-tolerance) & (z >= -tolerance)
             & (values >= z-tolerance) & (z >= next_values-tolerance))
    indices = np.flatnonzero(valid)
    if len(indices):
        i = int(indices[0])
        return result(beta/2, 0.0, max(float(z[i]), 0.0), float(mu[i]))
    raise ArithmeticError("Water-filling active-set equations did not yield a feasible target projection.")
