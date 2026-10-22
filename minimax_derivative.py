"""
Minimax (state-space) derivative estimation from noisy data.

Implements the derivative-estimation algorithm of the *quantum super-Krylov*
method, A. Byrne, W. Kirby, K. M. Soodhalter, S. Zhuk, arXiv:2412.17289,
Section III and Appendix B.

Problem (Sec. III).  We sample a smooth scalar signal z(t) -- for the
super-Krylov method z(t) = |<v|U(t)|v>|^2 is a recovery probability, but the
method is generic -- at D (roughly equispaced) times t_i in Omega = [0, T] and
obtain noisy measurements

    y_i = z(t_i) + eta_i ,      i = 1, ..., D                         (paper eq. 25)

We want smooth estimates of z and of its first M-1 derivatives
d^k/dt^k z(t), k = 0, ..., M-1, robust to the noise eta.  In the super-Krylov
method the first derivative gives the off-diagonal J-matrix entries via
J_{0k} = (i/k) z'(k dt) (eq. 21), so a noise-robust z' is the key deliverable.

State-space formulation (paper eqs. 26-31, B4-B13).  Stack the signal and its
derivatives into the state vector

    x(t) = ( z, z', z'', ..., z^{(M-1)} )^T ,   x_k = d^k/dt^k z ,

which obeys the integrator chain dx/dt = A x + f, with A the M x M shift matrix
(eq. 28) and the "input" f driving only the top derivative.  The estimate is the
regularized least-squares solution (paper eqs. 31 / B13)

    x_hat = argmin_w  ||y - C~ w||_2^2 + theta * ||L w||^2 ,   theta = theta_1/theta_2 ,

where (C~ w)_i = w_0(t_i) samples the signal at the measurement times and
L w = dw/dt - A w is the state operator (eq. B8).  Because the integrator chain
x_l' = x_{l+1} (l < M-1) is enforced by the dynamics and only the top derivative
is penalized, x_hat is an *M-th order smoothing spline*: x_hat_0 is a smoothed z
and x_hat_k = d^k/dt^k x_hat_0 are its analytic derivatives.  Equivalently, x_hat
solves the coupled ODE (paper eq. 40).

Min-max formulation (paper Sec. III A, eqs. 36-42).  With theta = theta_1/theta_2
and (theta_1, theta_2) satisfying the model bound (eq. 34/43), the minimizer above
*equals* the minimax estimate solving

    min_{q in G_y} max_{w in G_y} |(l, q - w)|                        (eq. 36)

over the admissible ellipsoid
    G_y = { w : theta_2 ||y - C~ w||^2 + theta_1 ||L w||^2 <= 1 } .   (eq. 37)

The associated *worst-case* ("minimax") estimation error of the point evaluation
l = delta(t - t*) e_k  (derivative order k at time t*) is

    Upsilon_{x_hat}(l) = max_{w in G_y} |(l, x_hat - w)| = sqrt((l, p_hat)) , (eqs. 38, 41)

with p_hat the solution of the covariance boundary-value problem (eq. 42).  It
upper-bounds the true error on x_hat_k(t*) (Example III.1, eq. 39) and is the
error bar drawn in the paper's Fig. 3.  Being a worst-case bound over the whole
admissible set, Upsilon is deliberately conservative -- typically several times
larger than the statistical standard deviation.

Discretization (this module).  On a uniform grid {s_p} of spacing h the signal
f = x_0 is represented by its nodal values.  The smoothness term
||L w||^2 = int (x_0^{(M)})^2 ds (paper eq. 32, after eliminating the exact
integrator chain) is the Riemann sum h * sum_p (D_M f)_p^2 with D_M the M-th
finite-difference operator; the derivatives are read off as x_k = D_k f with D_k
the k-th central-difference operator.  Writing S for the sampling operator, the
estimate solves the sparse, banded normal equations

    G f = theta_2 S^T y ,   G := theta_2 S^T S + theta_1 h D_M^T D_M ,

which is the discretization of the coupled ODE (eq. 40).  The data-independent
minimax error of the order-k derivative is the discrete Green's function
(eq. 41)

    Upsilon_k(s_p) = sqrt( [ D_k G^{-1} D_k^T ]_{pp} ) .

The estimate x_hat depends on the data only through theta = theta_1/theta_2; the
absolute scale of the error bars needs theta_1 and theta_2 individually (eq. 43).

Choosing the parameters (paper eq. 43):
    theta_2 = 1 / (2 * ||eta||^2)   ,   ||eta||^2 ~ D * sigma^2  (i.i.d. noise) ,
    theta_1 = 1 / (2 * ||z^{(M)}||^2_{L2}) .
Both can be passed explicitly; data-driven estimates are used otherwise.  (The
paper's eq. 43 states the smoothness bound on the highest penalized derivative;
we normalize by the same derivative that the penalty acts on, ||z^{(M)}||.)  For
the super-Krylov signal the bound |z'(t)| <= 2 m K (eq. 44) is a useful analytic
input.
"""

from dataclasses import dataclass
from math import comb
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

__all__ = [
    "MinimaxDerivativeResult",
    "minimax_derivative_estimate",
    "estimate_noise_std",
    "estimate_topderiv_l2",
    "krylov_JR_first_rows",
]


# --------------------------------------------------------------------------- #
#  Finite-difference operators
# --------------------------------------------------------------------------- #
def _first_derivative_operator(P, h):
    """Second-order first-derivative operator D1 (P x P) on a uniform grid.

    Central differences in the interior, one-sided at the two boundaries.
    """
    rows, cols, data = [], [], []
    for p in range(P):
        if p == 0:
            cols_p, coeff = (0, 1, 2), (-1.5, 2.0, -0.5)      # forward, O(h^2)
        elif p == P - 1:
            cols_p, coeff = (P - 3, P - 2, P - 1), (0.5, -2.0, 1.5)  # backward
        else:
            cols_p, coeff = (p - 1, p + 1), (-0.5, 0.5)       # central
        for c, a in zip(cols_p, coeff):
            rows.append(p)
            cols.append(c)
            data.append(a / h)
    return sparse.coo_matrix((data, (rows, cols)), shape=(P, P)).tocsr()


def _derivative_operator(k, P, h):
    """k-th derivative operator D_k (P x P), by composing D1 k times."""
    Dk = sparse.identity(P, format="csr")
    if k == 0:
        return Dk
    D1 = _first_derivative_operator(P, h)
    for _ in range(k):
        Dk = D1 @ Dk
    return Dk


def _mth_difference_operator(M, P, h):
    """M-th finite-difference penalty operator D_M, shape (P - M, P).

    Row r holds the scaled binomial coefficients of the M-th difference, so
    (D_M f)_r ~ f^{(M)}(s_{r + M/2}).  The L2 smoothness penalty int (f^{(M)})^2
    is then h * ||D_M f||^2 (paper eq. 32, top-derivative term).
    """
    if P - M < 1:
        raise ValueError(f"need more than M = {M} grid points for the penalty")
    coeffs = np.array([(-1) ** (M - j) * comb(M, j) for j in range(M + 1)]) / h ** M
    rows, cols, data = [], [], []
    for r in range(P - M):
        for j in range(M + 1):
            rows.append(r)
            cols.append(r + j)
            data.append(coeffs[j])
    return sparse.coo_matrix((data, (rows, cols)), shape=(P - M, P)).tocsr()


def _sampling_operator(sample_nodes, P):
    """Sampling operator S (D x P), (S f)_i = f(s_{node_i})  (paper eq. B10)."""
    sample_nodes = np.asarray(sample_nodes, dtype=int)
    D = sample_nodes.size
    return sparse.coo_matrix(
        (np.ones(D), (np.arange(D), sample_nodes)), shape=(D, P)
    ).tocsr()


def _build_grid(t, refine):
    """Uniform computational grid spanning [t[0], t[-1]].

    With ``refine = 1`` and equispaced samples the grid coincides with the
    sample times.  ``refine > 1`` inserts sub-nodes for smoother derivatives.
    Returns (grid, h, sample_nodes) with ``sample_nodes[i]`` the grid index
    nearest to t[i].
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array of sample times")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    refine = int(refine)
    if refine < 1:
        raise ValueError("refine must be a positive integer")

    P = refine * (len(t) - 1) + 1
    grid = np.linspace(t[0], t[-1], P)
    h = (t[-1] - t[0]) / (P - 1)
    sample_nodes = np.clip(np.rint((t - t[0]) / h).astype(int), 0, P - 1)
    snap_err = np.max(np.abs(grid[sample_nodes] - t))
    if snap_err > 0.25 * h:
        warnings.warn(
            "Sample times are not close to a uniform grid "
            f"(max snap error {snap_err:.3g} > h/4 = {0.25 * h:.3g}); "
            "consider a finer `refine` or resampling onto a uniform grid.",
            stacklevel=2,
        )
    return grid, h, sample_nodes


# --------------------------------------------------------------------------- #
#  Data-driven parameter estimates
# --------------------------------------------------------------------------- #
def estimate_noise_std(y):
    """Robust estimate of the per-sample noise std sigma from second differences.

    For a smooth signal plus i.i.d. noise, Var(Delta^2 y) ~ 6 sigma^2.  A
    median-absolute-deviation estimator is used for robustness.
    """
    y = np.asarray(y, dtype=float)
    d2 = np.diff(y, n=2)
    mad = np.median(np.abs(d2 - np.median(d2)))
    return 1.4826 * mad / np.sqrt(6.0)


def estimate_topderiv_l2(t, y, M, smooth_window=None):
    """Rough estimate of ||d^M/dt^M z||^2_{L2} over [t0, tD] (paper eq. 43).

    The signal is lightly smoothed (binomial window) before repeated numerical
    differentiation, then int (z^{(M)})^2 dt is evaluated by the trapezoidal
    rule.  Only a heuristic default -- passing an analytic value/bound via
    ``deriv_l2`` / ``deriv_bound`` is preferable.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if smooth_window is None:
        smooth_window = max(3, len(y) // 20)
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.array([1.0])
    for _ in range(smooth_window - 1):
        kernel = np.convolve(kernel, [0.5, 0.5])   # binomial (Gaussian-like)
    d = np.convolve(y, kernel, mode="same")
    for _ in range(M):
        d = np.gradient(d, t)
    return float(np.trapezoid(d ** 2, t))


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class MinimaxDerivativeResult:
    """Output of :func:`minimax_derivative_estimate`.

    Attributes
    ----------
    t : (P,) ndarray
        Uniform computational grid.
    estimate : (M, P) ndarray
        Row k is the estimate of d^k/dt^k z on the grid (x_hat, paper eq. 31);
        row 0 is the smoothed signal, row 1 its first derivative, etc.
    error : (M, P) ndarray or None
        Row k is the worst-case minimax error Upsilon_k on the grid (paper
        eqs. 38, 41), or ``None`` if not requested.  These are guaranteed
        (worst-case) error bars, not statistical standard deviations, and are
        deliberately conservative.
    M, theta1, theta2 : int, float, float
        Penalty order / state dimension and the regularization parameters used
        (theta = theta1 / theta2).
    """

    t: np.ndarray
    estimate: np.ndarray
    error: np.ndarray
    M: int
    theta1: float
    theta2: float

    @property
    def theta(self):
        return self.theta1 / self.theta2

    def evaluate(self, times, order=0):
        """Estimate of d^order/dt^order z at arbitrary ``times`` (linear interp)."""
        self._check_order(order)
        return np.interp(times, self.t, self.estimate[order])

    def evaluate_error(self, times, order=0):
        """Worst-case minimax error at arbitrary ``times`` (linear interp)."""
        if self.error is None:
            raise ValueError("error was not computed (compute_error=False)")
        self._check_order(order)
        return np.interp(times, self.t, self.error[order])

    def _check_order(self, order):
        if order < 0 or order > self.M - 1:
            raise ValueError(f"order must be in [0, {self.M - 1}]")

    def plot(self, y=None, sample_times=None, orders=(0, 1), axes=None):
        """Plot the estimate(s) with worst-case error bands (cf. paper Fig. 3)."""
        import matplotlib.pyplot as plt

        orders = tuple(orders)
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif", size=14)
        cmap = plt.get_cmap("Set2")
        if axes is None:
            _, axes = plt.subplots(len(orders), 1, sharex=True,
                                   figsize=(7, 3 * len(orders)), squeeze=False)
            axes = axes[:, 0]
        axes = np.atleast_1d(axes)
        for ax, order in zip(axes, orders):
            color = cmap(order)
            est = self.estimate[order]
            ax.plot(self.t, est, "-", color=color, lw=1.8,
                    label=rf"$\widehat{{x}}_{order}(t)$")
            if self.error is not None:
                err = self.error[order]
                ax.fill_between(self.t, est - err, est + err, color=color,
                                alpha=0.25, lw=0,
                                label=r"minimax error $\Upsilon$")
            if order == 0 and y is not None:
                st = self.t if sample_times is None else sample_times
                ax.plot(st, y, ".", color="0.35", ms=4, alpha=0.6, label=r"$y_i$")
            ax.set_ylabel(rf"$d^{order}z/dt^{order}$")
            ax.grid(color="gray", ls="dashdot", lw=0.8, alpha=0.5)
            ax.legend(fontsize=10, loc="best")
        axes[-1].set_xlabel(r"$t$")
        plt.tight_layout()
        plt.rcdefaults()
        return axes


# --------------------------------------------------------------------------- #
#  Main routine
# --------------------------------------------------------------------------- #
def minimax_derivative_estimate(
    t,
    y,
    M=3,
    theta1=None,
    theta2=None,
    theta=None,
    noise_std=None,
    deriv_l2=None,
    deriv_bound=None,
    refine=1,
    orders=None,
    compute_error=True,
    max_error_nodes=6000,
):
    """Minimax state-space derivative estimation (arXiv:2412.17289, Sec. III).

    Reconstructs a smooth signal and its first ``M - 1`` derivatives from noisy
    samples by solving the regularized problem (paper eq. 31) in its equivalent
    minimax form (eq. 36), and returns worst-case ("minimax") error bars
    (eqs. 38, 41).  The estimate is an M-th order smoothing spline, so the
    derivatives are the analytic derivatives of the smoothed signal and the
    structure x_hat_k = d/dt x_hat_{k-1} holds exactly (cf. eq. 40).

    Parameters
    ----------
    t, y : (D,) array_like
        Sample times (strictly increasing, roughly uniform) and noisy
        measurements y_i = z(t_i) + eta_i.
    M : int, default 3
        Penalty order / state dimension: penalizes int (z^{(M)})^2 and estimates
        z and its derivatives up to order M - 1.  M >= 2.  M = 3 matches the
        paper's numerical experiments.
    theta1, theta2 : float, optional
        Regularization parameters of the minimax formulation (paper eq. 43):
        theta2 weights the noise term, theta1 the smoothness term.  Used verbatim
        if both given.
    theta : float, optional
        Ratio theta1 / theta2 (paper eq. 31).  The *estimate* depends only on
        this ratio; supply it to control smoothing without fixing the error-bar
        scale.
    noise_std : float, optional
        Per-sample noise std sigma; sets theta2 = 1 / (2 D sigma^2) via
        ||eta||^2 ~ D sigma^2 (eq. 43).  Estimated from the data if omitted.
    deriv_l2 : float, optional
        Value of ||z^{(M)}||^2_{L2}; sets theta1 = 1 / (2 deriv_l2) (eq. 43).
    deriv_bound : float, optional
        Uniform bound on |z^{(M)}| (used as deriv_l2 = T * deriv_bound^2 when
        ``deriv_l2`` is not given).
    refine : int, default 1
        Sub-nodes per sample interval on the computational grid; refine > 1
        yields smoother, better-resolved derivatives at extra cost.
    orders : iterable of int, optional
        Derivative orders to return (default: 0, ..., M - 1).  Restricting this
        (e.g. ``orders=(0, 1)``) speeds up the error computation.
    compute_error : bool, default True
        Whether to compute the worst-case minimax error bars.
    max_error_nodes : int, default 6000
        Skip the error computation if the grid size P exceeds this, to bound cost.

    Returns
    -------
    MinimaxDerivativeResult

    Notes
    -----
    Parameter resolution:
      * theta2 from ``theta2`` else ``noise_std`` else estimated noise (eq. 43);
      * theta1 from ``theta1`` else ``deriv_l2``/``deriv_bound`` else estimated
        (eq. 43); or theta1 = theta * theta2 if ``theta`` is given.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape:
        raise ValueError("t and y must have the same shape")
    if M < 2:
        raise ValueError("M must be >= 2 (need at least z and its derivative)")
    if len(t) < M + 2:
        raise ValueError(f"need at least M + 2 = {M + 2} samples")

    if orders is None:
        orders = tuple(range(M))
    else:
        orders = tuple(sorted(set(int(o) for o in orders)))
        if orders[0] < 0 or orders[-1] > M - 1:
            raise ValueError(f"orders must lie in [0, {M - 1}]")

    grid, h, sample_nodes = _build_grid(t, refine)
    P = grid.size
    T = grid[-1] - grid[0]
    D = len(t)

    # ---- resolve theta2 (noise term, paper eq. 43) ----
    if theta2 is None:
        if noise_std is None:
            noise_std = estimate_noise_std(y)
            warnings.warn(
                f"noise_std not given; estimated sigma = {noise_std:.3g} from "
                "the data. Pass noise_std for reliable minimax error bars.",
                stacklevel=2,
            )
        theta2 = 1.0 / (2.0 * D * float(noise_std) ** 2)   # ||eta||^2 ~ D sigma^2

    # ---- resolve theta1 (smoothness term, paper eq. 43) ----
    if theta1 is None:
        if theta is not None:
            theta1 = float(theta) * theta2
        else:
            if deriv_l2 is None:
                if deriv_bound is not None:
                    deriv_l2 = T * float(deriv_bound) ** 2      # int <= T * bound^2
                else:
                    deriv_l2 = estimate_topderiv_l2(t, y, M)
                    warnings.warn(
                        "smoothness scale not given; estimated "
                        f"||z^(M)||^2 = {deriv_l2:.3g} from the data. Pass "
                        "deriv_l2 or deriv_bound for reliable results.",
                        stacklevel=2,
                    )
            theta1 = 1.0 / (2.0 * max(float(deriv_l2), 1e-300))

    theta1, theta2 = float(theta1), float(theta2)
    if theta1 <= 0 or theta2 <= 0:
        raise ValueError("theta1 and theta2 must be positive")
    theta_ratio = theta1 / theta2

    # ---- assemble and factorize the (banded) normal-equation matrix ----
    # G_full f = theta2 S^T y , with G_full = theta2 S^T S + theta1 h D_M^T D_M.
    # Factor out theta2: Gr := S^T S + theta_ratio h D_M^T D_M , f = Gr^{-1} S^T y.
    S = _sampling_operator(sample_nodes, P)
    DM = _mth_difference_operator(M, P, h)
    Gr = (S.T @ S + theta_ratio * h * (DM.T @ DM)).tocsc()
    lu = splu(Gr)

    f = lu.solve(np.asarray(S.T @ y).ravel())

    # ---- derivatives x_k = D_k f  and worst-case error Upsilon_k ----
    estimate = np.zeros((M, P))
    error = np.full((M, P), np.nan) if compute_error else None
    do_error = compute_error and P <= max_error_nodes
    if compute_error and not do_error:
        warnings.warn(
            f"skipping error computation: grid size P = {P} > max_error_nodes "
            f"= {max_error_nodes}. Increase max_error_nodes to force it.",
            stacklevel=2,
        )
        error = None

    for k in range(M):
        Dk = _derivative_operator(k, P, h)
        estimate[k] = Dk @ f                        # cheap: always fill all orders
        if do_error and k in orders:
            # Upsilon_k(s_p)^2 = [ D_k G_full^{-1} D_k^T ]_pp
            #                  = (1/theta2) [ D_k Gr^{-1} D_k^T ]_pp   (paper eq. 41)
            var = _propagated_inverse_diagonal(lu, Dk) / theta2
            error[k] = np.sqrt(np.clip(var, 0.0, None))

    return MinimaxDerivativeResult(
        t=grid, estimate=estimate, error=error, M=M,
        theta1=theta1, theta2=theta2,
    )


def _propagated_inverse_diagonal(lu, A, block=128):
    """diag(A G^{-1} A^T) from a factorization ``lu`` of G, in column blocks."""
    n = A.shape[0]
    AT = A.T.tocsc()
    out = np.zeros(n)
    for start in range(0, n, block):
        cols = np.arange(start, min(start + block, n))
        X = lu.solve(AT[:, cols].toarray())        # G^{-1} A^T[:, cols]
        AX = A[cols].dot(X)                          # (len(cols) x len(cols))
        out[cols] = np.diag(np.asarray(AX))
    return out


# --------------------------------------------------------------------------- #
#  Super-Krylov convenience: first rows of (J, R)  (paper Algorithm 1)
# --------------------------------------------------------------------------- #
def krylov_JR_first_rows(result, krylov_dt, krylov_dim):
    """First rows of the super-Krylov matrices (J, R) from the estimate.

    Implements step 3 of Algorithm 1 (paper eqs. 20-21):

        R_hat_{0k} = z_hat(k dt)                                       (eq. 20)
        J_hat_{0k} = (i / k) z_hat'(k dt)   for k >= 1                 (eq. 21)

    with the fixed values R_hat_{00} = 1 and J_hat_{00} = 0.  Feed the returned
    rows to a Toeplitz-Hermitian builder (e.g.
    ``hardware.utils.krylov.observables_to_matrix``) to impute the full matrices.

    Parameters
    ----------
    result : MinimaxDerivativeResult
        Output of :func:`minimax_derivative_estimate` (needs M >= 2 and order 1
        among the returned derivative orders).
    krylov_dt : float
        Krylov timestep dt.
    krylov_dim : int
        Krylov dimension m.

    Returns
    -------
    J0, R0 : (m,) complex ndarray
        First rows of J_hat and R_hat.
    """
    if result.M < 2:
        raise ValueError("need M >= 2 to estimate the first derivative for J")
    ks = np.arange(krylov_dim)
    tks = ks * krylov_dt
    R0 = result.evaluate(tks, order=0).astype(complex)
    dz = result.evaluate(tks, order=1)
    J0 = np.zeros(krylov_dim, dtype=complex)
    J0[1:] = (1j / ks[1:]) * dz[1:]     # eq. 21
    R0[0] = 1.0                          # R_hat_{00} = 1
    return J0, R0


# --------------------------------------------------------------------------- #
#  Self-test / demonstration on the paper's Fig. 2 signal
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Paper Fig. 2 test function: z(t) = cos(t) sin(3t), M = 3, Gaussian noise.
    rng = np.random.default_rng(0)
    T = 2 * np.pi
    D = 300
    sigma = 1e-2

    t = np.linspace(0, T, D)
    z = np.cos(t) * np.sin(3 * t)
    dz = -np.sin(t) * np.sin(3 * t) + 3 * np.cos(t) * np.cos(3 * t)
    y = z + rng.normal(0, sigma, size=D)

    # Analytic ||z'''||^2 for z = 0.5 (sin4t + sin2t): z''' = -32 cos4t - 4 cos2t.
    deriv_l2 = np.trapezoid((32 * np.cos(4 * t) + 4 * np.cos(2 * t)) ** 2, t)

    res = minimax_derivative_estimate(t, y, M=3, noise_std=sigma, deriv_l2=deriv_l2)

    z_hat = res.evaluate(t, order=0)
    dz_hat = res.evaluate(t, order=1)
    rel_z = np.linalg.norm(z_hat - z) / np.linalg.norm(z)
    rel_dz = np.linalg.norm(dz_hat - dz) / np.linalg.norm(dz)

    print(f"theta1 = {res.theta1:.4g}, theta2 = {res.theta2:.4g}, "
          f"theta = {res.theta:.4g}")
    print(f"relative L2 error, z   : {rel_z:.3e}")
    print(f"relative L2 error, z'  : {rel_dz:.3e}")
    if res.error is not None:
        err_dz = res.evaluate_error(t, order=1)
        inside = np.abs(dz_hat - dz) <= err_dz
        print(f"typical |z' - z'_hat|  : {np.abs(dz_hat - dz).mean():.3e}")
        print(f"mean minimax error z'  : {err_dz.mean():.3e} (worst-case bound)")
        print(f"true z' inside minimax band: {100 * inside.mean():.1f}% of points")
