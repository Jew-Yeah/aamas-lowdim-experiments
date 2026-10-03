[English](cage_restart_theory_en.md) | [Русский](cage_restart_theory_ru.md)

# Scalar-aware saddle selection and block restarts

This note separates an admissible choice inside the manuscript's oracle from
changes to its one-switch algorithm. It states mathematical bounds, not new
experimental results. The original manuscript is unchanged. For the CAGE
model and its feedback assumptions, see [the benchmark description](cage_en.md)
and [the extended study](cage_study_en.md).

## What the experiment measures

The primary experimental loss is the sum of four native simulator costs per
simulator step. The theorem instead controls the Euclidean distance of the
average vector payoff from the response target

\[
S(Q)=\operatorname{cl}\operatorname{conv}
\{u(p^\star(z),z):z\in Q\}.
\]

These are different criteria. Improving a bound on target distance does not
by itself establish lower native loss. The calibrated payoff tensor and
response rule are fixed before evaluation. An attacker mode is disclosed
after the defender chooses its policy; the learner does not infer that mode
from native cyber telemetry.

In the current 50-step calibration, one meta-round selects policies for an
entire simulator episode. A block of 1,000 meta-rounds therefore represents
50,000 simulator steps in separately initialized episodes. A budget of 1,000
native simulator steps would represent 20 such meta-rounds. These two block
sizes must not be conflated.

## An admissible scalar-aware oracle

At fast round \(t\ge2\), keep the complete past hull
\(H_t=\operatorname{conv}\{\ell_s:s<t\}\). For its stored vertices, form the
matrix \(B_{a,j}=\langle\lambda_t,u(e_a,\ell_j)\rangle\). Obtain an exact
optimal dual mixture \(z^\star\) and let

\[
v=\min_a(Bz^\star)_a.
\]

Let \(c_{t,W}\) be predicted scalar costs using only the fixed training tensor
and the previous \(W\) disclosed attacker modes. For example, use the response
baseline's empirical mixture of the last \(\min(W,t-1)\) modes. Choose

\[
\begin{aligned}
\min_{p\in\Delta_K}\quad&c_{t,W}^{\mathsf T}p,\\
\text{subject to}\quad&B^{\mathsf T}p
   \le(v+\rho t^{-1/2})\mathbf1,\qquad 0\le\rho\le1.
\end{aligned}
\]

An exact minimax primal policy makes this program feasible. Its chosen policy
and the unchanged dual mixture satisfy

\[
\max_j(p^{\mathsf T}B)_j-\min_a(Bz^\star)_a
\le\rho t^{-1/2}\le t^{-1/2}.
\]

Thus this is a causal specialization of the existing saddle interface, with
nominal \(\beta_t=0\). It preserves the full hull, the response witness
\(p^\star(\ell_t^\star)\), hull projection and direction update. At
\(\rho=0\) it selects among exact minimax primal policies; positive \(\rho\)
uses part of the tolerance already permitted by the manuscript. The
approximation is an intentional choice of policy, not estimation noise.

The actual primal-dual gap must be checked after solving. With an approximate
dual, feasibility is not automatic at every requested tolerance. A failed
secondary solve should fall back to the original checked saddle pair. A
reported gap above \(t^{-1/2}\) contributes
\(\beta_t=\max\{0,\text{gap}_t-t^{-1/2}\}\), as in the manuscript's
finite-accuracy analysis. Floating-point checks are numerical diagnostics;
they do not establish an exact-real-arithmetic certificate on their own.

No theorem change is needed for this oracle choice under the same certified
interfaces. An experimental description must disclose \(W,\rho\), the
selection rule, numerical checks and tuning protocol. This construction
does not promise an advantage in native loss. Forgetting old hull vertices
would be a separate change requiring a different argument.

## Restart definitions

Partition the meta-game into blocks of actual lengths \(n_1,\ldots,n_N\),
with \(\sum_i n_i=T\); the final block can be shorter than the announced
block length \(L\). Both restart variants reset the direction, residual
counter, local fast clock, one-switch state and safe instance at each block.
They use the unchanged certified budget
\(G_{n_i}=\max_{0\le h\le n_i}B_0(h)\). A threshold-crossing round remains
fast, and any safe continuation is a fresh run for the remaining block length.
The first local round follows the original arbitrary-start convention;
the current implementation uses a uniform defender mixture.

- **Fresh history:** the hull and scalar forecasting history contain only
  observations from the current block.
- **Retained history:** the hull and forecasting history contain every
  previously disclosed global observation, including observations collected
  in earlier safe continuations. Only the optimization state and local clocks
  restart. After the first local observation, the next saddle call uses this
  complete preceding history.

Scalar selection inside a restarted block uses the local fast clock in its
tolerance and the chosen forecasting history. It is covered by the segment
argument below, not by identifying the restarted trajectory with the
manuscript's original uninterrupted algorithm.

## Fresh-history bound

Write \(Q_i\) for the hull of observations within block \(i\), and
\(V_i^{\mathrm{fresh}}\) for its cumulative distances to the local past hull,
excluding its first observation. Apply the one-switch theorem separately to
each fresh block. Since \(Q_i\subseteq Q_T\), all block targets lie in
\(S(Q_T)\). Convex averaging and the triangle inequality give

\[
T\,\operatorname{dist}(\bar u_T,S(Q_T))
\le\sum_{i=1}^N
\left[10\sqrt{n_i}
+2\min\{K_uV_i^{\mathrm{fresh}},G_{n_i}\}+2\right].
\]

Local novelty can repeat after each reset. In general,
\(\sum_iV_i^{\mathrm{fresh}}\le V_T\) is false: the global hull may already
contain an attacker mode that is new to the current block. If every block
stays fast, the sharper bound is
\(10\sum_i\sqrt{n_i}+\sum_i E_i\).

## Retained-history bound

A retained block can choose a response witness from an earlier block. That
witness need not belong to \(S(Q_i)\), so the local-block target version of
the one-switch theorem cannot simply be cited. The fast-prefix proof still
works relative to the global target \(S(Q_T)\): all witnesses belong to that
convex set, the direction starts at zero, and the update uses local indices.
The first local round costs at most 2. A fresh safe tail has a target contained
in \(S(Q_T)\) as well.

For each block, let \(V_i^{\mathrm{ret}}\) sum distances to the complete global
past hull on its rounds after the first local round. Then
\(\sum_iV_i^{\mathrm{ret}}\le V_T\), and the same crossing proof yields

\[
\begin{aligned}
T\,\operatorname{dist}(\bar u_T,S(Q_T))
&\le10\sum_i\sqrt{n_i}
 +2\sum_i\min\{K_uV_i^{\mathrm{ret}},G_{n_i}\}+2N\\
&\le10\sqrt{NT}
 +2\min\{K_uV_T,\sum_iG_{n_i}\}+2N.
\end{aligned}
\]

If every block stays fast, the stronger bound is
\(10\sum_i\sqrt{n_i}+\sum_iE_i\), with
\(\sum_iE_i\le K_uV_T\). Keeping observations from safe tails is essential
to the identification of each later hull with the complete global past hull.

All displayed restart bounds are nominal, with exact hull projection and
saddle gap within the stated local tolerance. Finite oracle errors can be
charged block by block using the manuscript's \(\alpha\) and \(\beta\) terms.

## What repeated restarts cost

With the explicit safe base,
\(G_n=6\sqrt{k}\,n^{3/4}\). For equal blocks \(T=NL\), either generic
restart bound implies

\[
\operatorname{dist}(\bar u_T,S(Q_T))
\le\frac{10}{\sqrt L}
 +\frac{12\sqrt{k}}{L^{1/4}}+\frac{2}{L}.
\]

For a fixed number of blocks the original asymptotic exponents are preserved,
with different constants. For fixed \(L\) and growing \(T\), this upper bound
does not tend to zero. That is a limitation of the guarantee, not a claim
that every restarted experiment fails to converge. Growing block lengths
recover decreasing bounds; resetting a direction can still help or hurt
finite-horizon native loss.

In the three-mode CAGE meta-game, an uninterrupted full-history fast run can
accumulate nonzero exact projection residual only at the first appearance of
each previously unseen pure mode. Thus its residual is at most \(2(M-1)=4\)
when \(M=3\). Retained-history restarts cannot make an already seen mode novel
again; fresh-history restarts can. The standard certified budget at a
1,000-round block is much larger than 4, so ordinary complete blocks cannot
trigger the default safe switch in this particular finite-mode setup. This
argument does not apply to arbitrary continuous opponent actions.

Resetting **only** the residual counter, while retaining the direction,
history and clock, leaves actions unchanged whenever neither version
switches. A full block restart is different because it also changes the
direction and local update schedule.

## Reporting and tuning

The two restart variants are exploratory extensions, not the original
single-switch algorithm over horizon \(T\). Testing them does not require
editing the current manuscript. Presenting them as a theorem-backed final
method would require their definition and the corresponding segment bound.

Select block length, forecasting window and \(\rho\) using training and a
separate validation stage. Freeze the selected configuration before a new
untouched test; previously inspected test outcomes cannot serve as fresh
confirmation. Give window and Hedge baselines a comparable tuning budget.
Retain the same declared native-loss endpoint and disclose whether histories
are retained. Exploration can suggest a useful variant; its improvement
still requires independent evaluation.
