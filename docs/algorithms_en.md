# Algorithms and experimental protocol

[Русская версия](algorithms_ru.md) · [Data and preprocessing](data_en.md)

## Finite game and response rule

The learner chooses a distribution $p_t\in\Delta_K$ over $K$ fixed
allocation schedules. The opponent chooses a distribution
$\ell_t\in\Delta_M$ over $M$ fixed demand profiles. The public tensor
$A\in\mathbb R^{K\times M\times d}$ defines the biaffine payoff

$$
u(p,\ell)=\sum_{a=1}^K\sum_{j=1}^M p_a\ell_j A_{a,j,:}.
$$

The allocation benchmark has seven loss coordinates: five borough
shortfalls, resource cost, and disparity between the largest and smallest
borough shortfalls. Shortfalls and disparity are divided by baseline
capacity $C$. Resource cost is $0.25\sum_b s_{a,b}/C$, where $s_a$
is schedule $a$. These are stated model choices. The first seven schedules
allocate $C$; the eighth offers a reserve option allocating $1.25C$.
Thus $C$ denotes baseline capacity, and the feasible menu includes the
specified reserve option.

All tensor entries are divided by a single
$b=\max\{1,\max_{a,j}\|u_{\rm raw}(e_a,e_j)\|_2\}$. Consequently
$\|u(p,\ell)\|_2\le1$ for every pair of mixtures. Every method uses
the same tensor, menu, profiles, weights, and feedback. The response rule is

$$
p^\star(\ell)=e_{\min\operatorname{argmin}_a
 w^\top u(e_a,\ell)},\qquad
w=\frac{(1,1,1,1,1,1,0.5)}{6.5}.
$$

Here the minimum selects the smallest schedule index among exact ties in
the supplied floating-point scores. The common weight normalization leaves
the minimizing schedule unchanged. The response is computed by scanning
all schedules.

## Compared methods

| Run identifier | Action rule | Role |
|---|---|---|
| `one_switch` | Past-hull policy followed, if its residual threshold is crossed, by a fresh safe run | Manuscript's master algorithm |
| `shared_past_hull` | Past-hull policy throughout the horizon | Shared geometric fast policy |
| `block_safe` | Known-horizon block routine throughout the run | Manuscript's explicit safe routine |
| `uniform` | Fixed schedule assigning $C/5$ to each borough | Allocation reference |
| `historical_share` | Fixed schedule assigning $C$ in training-period demand proportions | Allocation reference |
| `reserve` | Fixed training-share schedule with the permitted $1.25C$ reserve | Capacity-control reference |
| `last_week` | Weighted best response to the mean of up to seven previously revealed profiles | Causal allocation heuristic |

The `uniform` reference is a single equal-allocation schedule. The first
round of each mathematical learner instead uses the uniform distribution
over the entire schedule menu. `last_week` starts with the equal-allocation
schedule. Its seven-round window is a fixed design choice. The allocation
references have no asserted opportunistic approachability bound.

The past-hull policy is shared with the geometric method in
[Marinov et al., *Efficient Opportunistic Approachability*](https://proceedings.mlr.press/v313/marinov26a.html),
Section 5. The runs use one shared implementation; assigning an additional
prior-work label to it would duplicate the same policy. The explicit safe
routine instantiates the epoch construction underlying that paper's
Theorem 14. Theorem 14 gives a general $\widetilde O(T^{-1/4})$ rate;
Theorem 21 gives $\widetilde O(T^{-1/3})$ using an exponential-size
expert cover. The latter algorithm remains outside this implementation.
These comparisons refer to the
[full proceedings paper](https://raw.githubusercontent.com/mlresearch/v313/main/assets/marinov26a/marinov26a.pdf).

## Shared past-hull policy

Let $H_t=\operatorname{conv}\{\ell_1,\ldots,\ell_{t-1}\}$.
Round 1 uses any permissible initial mixture, chosen here as uniform.
After observing $\ell_1$, set
$s_1=u(p^\star(\ell_1),\ell_1)$, $E_1=0$, and $\lambda_2=0$.

For each $t\ge2$, before observing $\ell_t$, form the scalar matrix
$B_{a,s}=\langle\lambda_t,u(e_a,\ell_s)\rangle$, $s<t$.
The row player minimizes and the column player maximizes this matrix game.
Two linear programs give mixtures $p_t$ and $\zeta_t$, with
$\ell_t^\star=\sum_{s<t}\zeta_{t,s}\ell_s$. The implementation checks
the independently recomputed saddle gap

$$
g_t=\max_s(p_t^\top B)_s-\min_a(B\zeta_t)_a.
$$

After the opponent action is revealed, project it onto the past hull and
set

$$
\begin{aligned}
\widehat\ell_t&=\operatorname{proj}_{H_t}(\ell_t),&
s_t&=u(p^\star(\ell_t^\star),\ell_t^\star),\\
a_t&=u(p_t,\widehat\ell_t)-s_t,&
r_t&=u(p_t,\ell_t)-u(p_t,\widehat\ell_t),\\
E_t&=E_{t-1}+\|r_t\|_2,&
\lambda_{t+1}&=\operatorname{proj}_{\mathbb B_d(1)}
  \left(\lambda_t+\frac{a_t}{2\sqrt t}\right).
\end{aligned}
$$

Both the saddle mixture and its response witness lie in the past hull.
Under exact hull projection,
$E_T\le K_u V_T$, where
$V_T=\sum_{t=2}^T\operatorname{dist}(\ell_t,H_t)$.
The recorded `h_t` is the computed hull distance. The runner sums it only
on rounds when the fast policy is active.

## Safe block routine

Write $k=K-1$ for the learner's affine dimension. A fresh run of known
length $h$ uses

$$
m_h=\max\{1,\lfloor\sqrt h/k\rfloor\},\qquad
n_h=\lfloor h/m_h\rfloor,\qquad r_h=h-m_hn_h.
$$

The outer direction starts at zero and remains fixed for each block of
$n_h$ rounds. Within a block, initialize the learner mixture to uniform,
play the current mixture before seeing that round's opponent action, and
then perform

$$
p_{t+1}=\operatorname{proj}_{\Delta_K}
 \left(p_t-\frac{c_t}{K\sqrt{n_h}}\right),\qquad
(c_t)_a=\langle\lambda_e,u(e_a,\ell_t)\rangle.
$$

This probability-coordinate step follows from the manuscript's
$k/\sqrt{n_h}$ step in its well-rounded affine coordinates. Centering
the regular simplex at its uniform mixture and scaling the tangent
coordinates by $\sqrt{K(K-1)}$ gives
$\mathbb B_k(1)\subseteq P\subseteq\mathbb B_k(k)$.
Projection and the chain rule then give the factor $1/K$ above.

At the end of block $e$, compute its opponent mean $\bar\ell_e$,
the response witness $s_e=u(p^\star(\bar\ell_e),\bar\ell_e)$, and
$v_e=n_h^{-1}\sum_{t\in I_e}u(p_t,\ell_t)-s_e$. Update

$$
\lambda_{e+1}=\operatorname{proj}_{\mathbb B_d(1)}
 \left(\lambda_e+\frac{v_e}{2\sqrt{m_h}}\right).
$$

Reset the inner mixture at the next block. The $r_h$ remainder rounds
use the uniform mixture, corresponding to the origin in the affine
coordinates. In the manuscript's exact oracle model the cumulative
approachability error is at most

$$
B_0(h)=6\sqrt{k}\,h^{3/4}.
$$

The supplied allocation menu has $K=8$, hence $k=7$. The implementation
also handles the singleton menu separately.

## One-switch master

The horizon $T$ is supplied before play. The experimental threshold is
fixed by the stated safe guarantee:

$$
G_T=\max_{0\le h\le T}B_0(h)=6\sqrt{k}\,T^{3/4}.
$$

At the first round $\tau$ satisfying $E_\tau>G_T$, retain that round
in the fast prefix. If $\tau<T$, start a fresh safe routine of length
$T-\tau$ on the next round; its block indices are local to this new
run. The master never returns to fast mode. Its exact-oracle bound is

$$
T\delta_T\le10\sqrt T+2\min\{K_uV_T,G_T\}+2.
$$

The default threshold is conservative. Normalization implies
$E_T\le2(T-1)$; therefore a crossing is impossible for
$T\le81k^2$. With $K=8$, this covers all horizons up to 3969,
including the 730-day NYC holdout. Those runs necessarily have identical
`one_switch` and `shared_past_hull` actions and measure the fast branch.
The switch logic is separately checked on a constructed game with a valid
zero-error safe base. Experimental runs use the stated default budget.

## Causal protocol and target evaluation

Every run follows `choose()` and then `observe(ell)`. Profile fitting,
baseline capacity, historical shares, the action menu, normalization, and
weights are fixed from training data or a generated static game before test
play. Evaluation may process checkpoint targets ahead of the simulation;
these targets are never supplied to a learner.

At each checkpoint $t$, the evaluated target is

$$
Q_t=\operatorname{conv}\{\ell_1,\ldots,\ell_t\},\qquad
S(Q_t)=\operatorname{cl}\operatorname{conv}
 \{u(p^\star(z),z):z\in Q_t\},\qquad
\delta_t=\operatorname{dist}\left(\frac1t\sum_{s=1}^t
 u(p_s,\ell_s),S(Q_t)\right).
$$

The target oracle intersects the full opponent hull with response cells.
It respects the fixed tie rule, retaining closures of actual winner cells.
Projection uses either their payoff vertices or support-LP column
generation. Independent support LPs over the full cells check the resulting
projection gap. This includes responses to unobserved mixtures in $Q_t$.

The Euclidean Lipschitz bound recorded in each manifest is
$K_u=\max_a\|A_a^\top(I-\mathbf1\mathbf1^\top/M)\|_{\rm op}$.
It bounds payoff changes uniformly over learner mixtures on the opponent
simplex. The realized dimension is a numerical rank with a stated tolerance.

On the NYC benchmark, $\delta_t$ concerns the game built from training
centroids. Raw held-out request counts provide separate shortfall, cost,
disparity, and service-fraction diagnostics. Quantization errors are
reported in the data artifacts; the game's bound applies to its centroid
payoffs. Controlled regime paths study adaptation at a known realized
dimension, while the separate geometric constructions study hull growth.
Neither protocol by itself establishes a minimax exponent.

## Numerical evidence and recorded results

The solver uses floating-point LP/QP calculations. Feasibility checks,
saddle gaps, and squared-distance primal–dual gaps are recorded as numerical
diagnostics. For a past-hull projection gap $g$, the implementation
records $\alpha_t=\sqrt{\max\{g,0\}}$; it also records
$\beta_t=\max\{0,g_t-t^{-1/2}\}$. The exact-arithmetic theorem and
these numerical checks have distinct scopes, especially near response
ties and nearly degenerate hulls. A failed required oracle check stops the
run.

Result manifests contain configurations, seeds supplied by the caller,
software versions, a normalized-tensor hash, target-oracle mode, final and
checkpoint distances, applied metrics, switch round, and oracle residuals.
Action arrays are saved separately. Reported wall time includes each
method's evaluation; shared target construction is timed separately. This
timing is an end-to-end measurement, so evaluation costs must be accounted
for when interpreting computational comparisons.
