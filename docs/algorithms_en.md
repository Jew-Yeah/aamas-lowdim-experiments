[English](algorithms_en.md) | [Русский](algorithms_ru.md) | [Home](../README.md)

# Mathematical implementation

The recommended implementation is the one-switch master with a scalar-aware
choice among its admissible saddle responses. The original core learners remain
as implementation dependencies and controls. Validation selected `W = 16` and
`rho = 0.25` for the CAGE application; these are empirical defaults for this
benchmark, rather than universal theorem constants.

## Finite game and target

The learner chooses $p_t\in\Delta_K$ before observing
$\ell_t\in\Delta_M$. A fixed known tensor defines

$$
u(p,\ell)=\sum_{a=1}^K\sum_{j=1}^M p_a\ell_j A_{a,j,:},
\qquad \|u(p,\ell)\|_2\le1.
$$

CAGE uses six defensive policies, three attack modes, and four loss coordinates.
Its tensor normalization is fixed before validation or testing. The response map
is fixed throughout play:

$$
p^\star(\ell)=e_{\min\operatorname{argmin}_a w^\top u(e_a,\ell)}.
$$

The smallest action index resolves exact ties in supplied floating-point scores.
CAGE gives the four coordinates equal weight. The manuscript's target is

$$
Q_t=\operatorname{conv}\{\ell_1,\ldots,\ell_t\},\qquad
S(Q_t)=\operatorname{cl}\operatorname{conv}
\{u(p^\star(z),z):z\in Q_t\},
$$

and its error is the distance of the average vector payoff to $S(Q_t)$.
The full hull includes previously unobserved mixtures. The final scalar-loss
experiment does not measure this target distance.

## Fast hull policy and scalar-aware oracle

Round 1 uses a uniform mixture. After observing $\ell_1$, initialize
$s_1=u(p^\star(\ell_1),\ell_1)$, $E_1=0$, and $\lambda_2=0$.
At $t\ge2$, set $H_t=\operatorname{conv}\{\ell_s:s<t\}$ and
$B_{a,s}=\langle\lambda_t,u(e_a,\ell_s)\rangle$.
The learner minimizes and the opponent maximizes this matrix game.

First compute its original minimax dual distribution $z^*$ and
$L=\min_a(Bz^*)_a$. Let $c_t$ contain each defense's calibrated scalar loss
against the empirical distribution of the previous at most $W$ revealed modes.
The recommended oracle solves

$$
\min_{p\in\Delta_K}c_t^\top p\quad\text{subject to}\quad
B^\top p\le\bigl(L+\rho/\sqrt t\bigr)\mathbf1,
\qquad 0\le\rho\le1.
$$

The same dual distribution defines $\ell_t^\star$ and its response witness.
The actual saddle gap is independently recomputed:

$$
g_t=\max_s(p_t^\top B)_s-\min_a(Bz^*)_a.
$$

The allowed slack lies within the manuscript's $t^{-1/2}$ oracle allowance.
An invalid or failed forecast LP falls back to the original saddle pair.
Repeated observations can be removed from the hull representation without
changing its geometry; the forecast retains their chronological frequencies.

After revealing the current opponent action, update

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

Under exact projection, $E_T\le K_uV_T$, where
$V_T=\sum_{t=2}^T\operatorname{dist}(\ell_t,H_t)$.
Forecast updates occur only after the current action has been chosen.

## One-switch master and safe continuation

Write $k=K-1$. The known-horizon safe routine has nominal cumulative bound
$B_0(h)=6\sqrt{k}\,h^{3/4}$. The master uses the unchanged threshold

$$
G_T=6\sqrt{k}\,T^{3/4}.
$$

The first round $\tau$ satisfying $E_\tau>G_T$ remains in the fast prefix.
If $\tau<T$, a fresh safe routine of length $T-\tau$ starts next round;
there is no return to fast mode. Its exact-oracle bound is

$$
T\delta_T\le10\sqrt T+2\min\{K_uV_T,G_T\}+2.
$$

For a safe run of length $h$, the block schedule is
$m_h=\max\{1,\lfloor\sqrt h/k\rfloor\}$,
$n_h=\lfloor h/m_h\rfloor$, and $r_h=h-m_hn_h$.
The outer direction is fixed within each block. The inner mixture starts
uniformly and updates after observing the current mode:

$$
p_{t+1}=\operatorname{proj}_{\Delta_K}
\left(p_t-\frac{f_t}{K\sqrt{n_h}}\right),\qquad
(f_t)_a=\langle\lambda_e,u(e_a,\ell_t)\rangle.
$$

Here $f_t$ is the safe routine's direction-weighted vector. At the block end,
with opponent mean
$\bar\ell_e$ and witness $s_e=u(p^\star(\bar\ell_e),\bar\ell_e)$,
set $v_e=n_h^{-1}\sum_{t\in I_e}u(p_t,\ell_t)-s_e$ and
$\lambda_{e+1}=\operatorname{proj}_{\mathbb B_d(1)}
(\lambda_e+v_e/(2\sqrt{m_h}))$. Remainder rounds use the uniform mixture.
The factor $1/K$ follows from the regular-simplex affine-coordinate transform.

No safe switches occur in the recorded three-mode CAGE paths: only first
appearances are novel, so $E_T\le2(M-1)=4<G_T$. The recommended method
therefore evaluates the fast policy on these inputs. Exploratory repeated
restarts are separate algorithms and are preserved in the
[full-study archive](https://github.com/Jew-Yeah/aamas-lowdim-experiments/blob/e8c85cb2e7be49032c1c2d014b6f19084ea064a8/docs/cage_restart_theory_en.md).

## Numerical and statistical scope

Every learner follows `choose()` then `observe(ell)`. For a projection gap $g$,
the implementation records $\alpha_t=\sqrt{\max\{g,0\}}$; for a saddle gap
it records $\beta_t=\max\{0,g_t-t^{-1/2}\}$. Diagnostics include feasibility,
actual gap, fallback, residual, and switch counts. Floating-point checks do not
constitute exact-arithmetic certificates.

The target oracle, when requested by the core experiment API, intersects the
full opponent hull with the response cells and checks projection gaps using
independent support LPs. It includes responses at unobserved mixtures. The
primary CAGE report uses held-out scalar loss and conditional bootstrap
inference; favorable scalar loss does not establish a smaller vector target
distance or a guarantee for unknown exact simulator expectations.

The shared geometric fast policy is related to Section 5 of
[Marinov et al., Efficient Opportunistic Approachability](https://proceedings.mlr.press/v313/marinov26a.html).
The exponential expert-cover algorithm is not implemented. CAGE is the
simulator environment, not the name of an independent comparison algorithm.
