import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

# tick is used for the baseline Hawkes model
from tick.hawkes import HawkesExpKern
import matplotlib.pyplot as plt
import os


# -----------------------------
# Data loading
# -----------------------------

def load_events_2d(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Load events from events_100k.json into tick's 2D format.

    Returns
    -------
    events_2d : list[list[float]]
        events_2d[dim] is a list of event times for dimension `dim`.
    T : float
        Observation horizon (max timestamp).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dim0: List[float] = []
    dim1: List[float] = []
    for e in raw:
        t = float(e["t"])  # event time
        i = int(e["i"])    # dimension index (0 or 1)
        if i == 0:
            dim0.append(t)
        elif i == 1:
            dim1.append(t)
        else:
            # Ignore unexpected dimensions
            continue

    # Ensure sorted order (should already be sorted)
    dim0.sort()
    dim1.sort()
    T = max(dim0[-1] if dim0 else 0.0, dim1[-1] if dim1 else 0.0)
    # Convert to numpy arrays as required by tick
    return [np.asarray(dim0, dtype=float), np.asarray(dim1, dtype=float)], T


# -----------------------------
# Simple baseline with tick
# -----------------------------

@dataclass
class BaselineResult:
    decay: float
    mu: np.ndarray
    adjacency: np.ndarray
    loglik: float


def fit_simple_baseline_tick(events_2d: List[np.ndarray], decay_grid: np.ndarray) -> BaselineResult:
    """
    Fit a 2D Hawkes with a single exponential kernel using tick, with a
    small grid search over the decay parameter. Returns the best model by log-likelihood.
    """
    best = None
    for decay in decay_grid:
        # This tick version expects a 2D decays matrix (n_nodes x n_nodes)
        decays_mat = np.full((2, 2), float(decay), dtype=float)
        learner = HawkesExpKern(decays=decays_mat, verbose=False)
        learner.fit([events_2d])

        # tick's score is the log-likelihood on provided realisations
        ll = float(learner.score())
        if (best is None) or (ll > best.loglik):
            best = BaselineResult(
                decay=float(decay),
                mu=learner.baseline.copy(),
                adjacency=learner.adjacency.copy(),
                loglik=ll,
            )
    assert best is not None
    return best


# -----------------------------
# Full model with exogenous sine term
# -----------------------------

@dataclass
class FullFitResult:
    params: np.ndarray
    loglik: float
    aic: float
    success: bool


class FullHawkesWithSine:
    """
    Full Hawkes model in 2D with a shared exponential decay beta and an
    additive exogenous term f_u(t) = beta0_u * sin(beta1_u * t).

    Intensity per dimension u in {0,1}:
        lambda_u(t) = mu_u
                      + beta0_u * sin(beta1_u * t)
                      + sum_j alpha_{u,j} * sum_{s in N_j, s < t} exp(-beta * (t - s))

    We optimize parameters by (numerical-gradient) L-BFGS-B.
    """

    def __init__(self, events_2d: List[List[float]], T: float, decay: float):
        self.events = [np.asarray(ev, dtype=float) for ev in events_2d]
        self.T = float(T)
        self.beta = float(decay)  # shared decay, fixed
        self.use_kernel = os.environ.get("USE_KERNEL", "1") != "0"

        # Regularization strengths (L2) for objective; default 0 (disabled)
        self.reg_l2_mu = float(os.environ.get("REG_L2_MU", "0.0"))
        self.reg_l2_A = float(os.environ.get("REG_L2_A", "0.0"))
        self.reg_l2_b0 = float(os.environ.get("REG_L2_B0", "0.0"))

        # Precompute terms used in the integral of the excitation part (if enabled):
        if self.use_kernel:
            # For each j, S_j = sum_{s in N_j} (1 - exp(-beta * (T - s)))
            self.S = [float(np.sum(1.0 - np.exp(-self.beta * (self.T - self.events[j])))) for j in range(2)]
        else:
            self.S = [0.0, 0.0]

        # Build merged chronological stream of events as (t, dim)
        merged = [(float(t), 0) for t in self.events[0]] + [(float(t), 1) for t in self.events[1]]
        merged.sort(key=lambda x: x[0])
        self.timeline = merged

    @staticmethod
    def _pack(mu: np.ndarray, A: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
        return np.concatenate([mu.ravel(), A.ravel(), b0.ravel(), b1.ravel()])

    @staticmethod
    def _unpack(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu = theta[0:2]
        A = theta[2:6].reshape(2, 2)
        b0 = theta[6:8]
        b1 = theta[8:10]
        return mu, A, b0, b1

    def log_likelihood(self, theta: np.ndarray) -> float:
        mu, A, b0, b1 = self._unpack(theta)

        # Boundaries are handled by optimizer; here we only ensure numerical safety
        eps = 1e-12

        # If no excitation kernel: simplify likelihood
        if not self.use_kernel:
            log_sum = 0.0
            for t, dim in self.timeline:
                lam = mu[dim] + b0[dim] * math.sin(b1[dim] * t)
                lam = max(lam, eps)
                log_sum += math.log(lam)

            integral_mu = float(np.sum(mu) * self.T)
            with np.errstate(divide='ignore', invalid='ignore'):
                sine_int = np.where(np.abs(b1) > 0, b0 * (1.0 - np.cos(b1 * self.T)) / b1, 0.0)
            integral_sine = float(np.sum(sine_int))
            return log_sum - (integral_mu + integral_sine)

        # 1) Sum over log intensities at event times (using recursive r_j updates)
        log_sum = 0.0
        r = np.array([0.0, 0.0], dtype=float)  # r[j] = sum exp(-beta (t - s)) for past events of j
        last_t = 0.0
        for t, dim in self.timeline:
            # decay the r's from last_t to t
            decay_factor = math.exp(-self.beta * (t - last_t)) if t > last_t else 1.0
            r *= decay_factor

            # intensity for both dims at current t
            sin_term = b0 * np.sin(b1 * t)
            lambdas = mu + A.dot(r) + sin_term

            lam_dim = max(lambdas[dim], eps)
            log_sum += math.log(lam_dim)

            # now include this event in its dimension's r
            r[dim] += 1.0
            last_t = t

        integral_mu = float(np.sum(mu) * self.T)
        with np.errstate(divide='ignore', invalid='ignore'):
            sine_int = np.where(np.abs(b1) > 0, b0 * (1.0 - np.cos(b1 * self.T)) / b1, 0.0)
        integral_sine = float(np.sum(sine_int))

        excitation_int_per_j = np.array(self.S, dtype=float) / self.beta
        integral_excitation = float(np.sum(self._unpack(theta)[1] * excitation_int_per_j))

        return log_sum - (integral_mu + integral_sine + integral_excitation)

    def _reg_penalty_full(self, theta: np.ndarray) -> float:
        """L2 penalties using current env-configured strengths."""
        mu, A, b0, _ = self._unpack(theta)
        pen = 0.0
        if self.reg_l2_mu > 0:
            pen += float(self.reg_l2_mu) * float(np.sum(mu * mu))
        if self.reg_l2_A > 0:
            pen += float(self.reg_l2_A) * float(np.sum(A * A))
        if self.reg_l2_b0 > 0:
            pen += float(self.reg_l2_b0) * float(np.sum(b0 * b0))
        return pen

    def log_likelihood_interval(self, theta: np.ndarray, t_start: float, t_end: float) -> float:
        """
        Log-likelihood restricted to [t_start, t_end], conditioning on full
        history before t_start. This is used for validation scoring.
        """
        mu, A, b0, b1 = self._unpack(theta)
        eps = 1e-12

        if not self.use_kernel:
            # No excitation path
            log_sum = 0.0
            integral_mu = float(np.sum(mu) * (t_end - t_start))
            with np.errstate(divide='ignore', invalid='ignore'):
                sine_int = np.where(np.abs(b1) > 0, b0 * (np.cos(b1 * t_start) - np.cos(b1 * t_end)) / b1, 0.0)
            integral_sine = float(np.sum(sine_int))

            for t, dim in self.timeline:
                if t <= t_start or t > t_end:
                    continue
                lam = mu[dim] + b0[dim] * math.sin(b1[dim] * t)
                lam = max(lam, eps)
                log_sum += math.log(lam)
            return log_sum - (integral_mu + integral_sine)

        # Initialize r at t_start by running through merged events until t_start
        r = np.array([0.0, 0.0], dtype=float)
        last_t = 0.0
        for t, dim in self.timeline:
            if t >= t_start:
                # decay up to t_start and stop
                r *= math.exp(-self.beta * (t_start - last_t)) if t_start > last_t else 1.0
                last_t = t_start
                break
            # decay to event time then add impulse
            r *= math.exp(-self.beta * (t - last_t)) if t > last_t else 1.0
            r[dim] += 1.0
            last_t = t
        else:
            # no events after last_t; just decay to t_start
            if t_start > last_t:
                r *= math.exp(-self.beta * (t_start - last_t))
                last_t = t_start

        # Now process within [t_start, t_end]
        log_sum = 0.0
        integral_mu = float(np.sum(mu) * (t_end - t_start))
        with np.errstate(divide='ignore', invalid='ignore'):
            sine_int = np.where(np.abs(b1) > 0, b0 * (np.cos(b1 * t_start) - np.cos(b1 * t_end)) / b1, 0.0)
            # since ∫ sin = -cos/b1, so from t_start to t_end: (-cos(b1 t_end) + cos(b1 t_start))/b1
        integral_sine = float(np.sum(sine_int))

        cur_t = t_start
        for t, dim in self.timeline:
            if t <= t_start:
                continue
            if t > t_end:
                break
            # integrate excitation over [cur_t, t)
            dt = t - cur_t
            if dt > 0:
                integral_exc_seg = float(np.sum(A.dot(r) * (1.0 - np.exp(-self.beta * dt)) / self.beta))
            else:
                integral_exc_seg = 0.0
            # decay r to t
            r *= math.exp(-self.beta * dt) if dt > 0 else 1.0

            # add log term at event
            sin_term = b0 * np.sin(b1 * t)
            lambdas = mu + A.dot(r) + sin_term
            lam_dim = max(lambdas[dim], eps)
            log_sum += math.log(lam_dim)

            # update r by impulse
            r[dim] += 1.0
            cur_t = t

        # integrate from last event to t_end
        dt_tail = t_end - cur_t
        integral_exc_tail = float(np.sum(A.dot(r) * (1.0 - np.exp(-self.beta * dt_tail)) / self.beta)) if dt_tail > 0 else 0.0

        return log_sum - (integral_mu + integral_sine + integral_exc_tail)

    def fit(self, init_mu: np.ndarray, init_A: np.ndarray) -> FullFitResult:
        # Parameter vector: [mu0, mu1, A00, A01, A10, A11, b0_0, b0_1, b1_0, b1_1]
        theta0 = self._pack(init_mu, init_A, np.array([0.1, 0.1], dtype=float), np.array([1.0, 1.0], dtype=float))

        # Bounds to keep intensities well-behaved
        bounds = [
            (1e-8, None), (1e-8, None),          # mu0, mu1 >= 0
            (0.0, None), (0.0, None), (0.0, None), (0.0, None),  # alpha matrix >= 0
            (-5.0, 5.0), (-5.0, 5.0),            # beta0 amplitudes (allow negative/positive)
            (1e-3, 10.0), (1e-3, 10.0),          # beta1 frequencies in rad / time unit
        ]

        def obj(theta: np.ndarray) -> float:
            return -self.log_likelihood(theta) + self._reg_penalty_full(theta)

        maxiter = int(os.environ.get("MLE_MAXITER", "500"))
        res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-6})
        k_params = 10  # number of estimated parameters in the full model
        ll = -float(res.fun)
        aic = 2 * k_params - 2 * ll
        return FullFitResult(params=res.x, loglik=ll, aic=aic, success=bool(res.success))

    def fit_with_fixed_b1(self, init_mu: np.ndarray, init_A: np.ndarray, fixed_b1_shared: float) -> FullFitResult:
        """
        Fit with b1 fixed (shared across two dimensions). Optimize mu (2), A (4), b0 (2).
        Parameter vector becomes [mu0, mu1, A00, A01, A10, A11, b0_0, b0_1].
        """
        b1_vec = np.array([fixed_b1_shared, fixed_b1_shared], dtype=float)

        def pack(mu: np.ndarray, A: np.ndarray, b0: np.ndarray) -> np.ndarray:
            return np.concatenate([mu.ravel(), A.ravel(), b0.ravel()])

        def unpack(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            mu = theta[0:2]
            A = theta[2:6].reshape(2, 2)
            b0 = theta[6:8]
            return mu, A, b0

        theta0 = pack(init_mu, init_A, np.array([0.1, 0.1], dtype=float))

        bounds = [
            (1e-8, None), (1e-8, None),          # mu
            (0.0, None), (0.0, None), (0.0, None), (0.0, None),  # A >= 0
            (-5.0, 5.0), (-5.0, 5.0),            # b0 amplitudes
        ]

        def obj(theta: np.ndarray) -> float:
            mu, A, b0 = unpack(theta)
            # reuse log-likelihood with fixed b1 & apply L2 penalties
            full_theta = self._pack(mu, A, b0, b1_vec)
            return -self.log_likelihood(full_theta) + self._reg_penalty_full(full_theta)

        maxiter = int(os.environ.get("MLE_MAXITER", "400"))
        res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-6})
        k_params = 8  # mu(2)+A(4)+b0(2)
        ll = -float(res.fun)
        aic = 2 * k_params - 2 * ll
        # Return a full-sized parameter vector for downstream usage
        mu_hat, A_hat, b0_hat = unpack(res.x)
        full_params = self._pack(mu_hat, A_hat, b0_hat, b1_vec)
        return FullFitResult(params=full_params, loglik=ll, aic=aic, success=bool(res.success))

    # -------- No-kernel (A=0) variant --------
    def _loglik_no_kernel(self, theta: np.ndarray) -> float:
        """theta = [mu0, mu1, b0_0, b0_1, b1_0, b1_1]"""
        mu = theta[0:2]
        b0 = theta[2:4]
        b1 = theta[4:6]
        eps = 1e-12
        log_sum = 0.0
        last_t = 0.0
        for t, dim in self.timeline:
            # intensity without excitation
            lam = mu + b0 * np.sin(b1 * t)
            lam_d = max(lam[dim], eps)
            log_sum += math.log(lam_d)
            last_t = t

        # integrals over [0, T]
        integral_mu = float(np.sum(mu) * self.T)
        with np.errstate(divide='ignore', invalid='ignore'):
            sine_int = np.where(np.abs(b1) > 0, b0 * (1.0 - np.cos(b1 * self.T)) / b1, 0.0)
        integral_sine = float(np.sum(sine_int))
        return log_sum - (integral_mu + integral_sine)

    def fit_no_kernel(self) -> FullFitResult:
        """Fit μ(2), b0(2), b1(2) with A fixed to 0 (no excitation)."""
        theta0 = np.array([0.5, 0.5, 0.1, 0.1, 1.0, 1.0], dtype=float)
        bounds = [
            (1e-8, None), (1e-8, None),  # mu
            (-5.0, 5.0), (-5.0, 5.0),    # b0
            (1e-3, 10.0), (1e-3, 10.0),  # b1
        ]
        maxiter = int(os.environ.get("MLE_MAXITER", "400"))
        def obj_no_kernel(th: np.ndarray) -> float:
            # Build a full-sized theta with A=0 and provided mu,b0,b1 to reuse penalty function
            mu = th[0:2]
            b0 = th[2:4]
            b1 = th[4:6]
            A = np.zeros((2, 2), dtype=float)
            full_theta = self._pack(mu, A, b0, b1)
            return -self._loglik_no_kernel(th) + self._reg_penalty_full(full_theta)

        res = minimize(obj_no_kernel, theta0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-6})
        ll = -float(res.fun)
        k_params = 6
        aic = 2 * k_params - 2 * ll
        mu = res.x[0:2]
        b0 = res.x[2:4]
        b1 = res.x[4:6]
        A = np.zeros((2, 2), dtype=float)
        full_params = self._pack(mu, A, b0, b1)
        return FullFitResult(params=full_params, loglik=ll, aic=aic, success=bool(res.success))


# -----------------------------
# GOF and validation utilities
# -----------------------------

def _sine_integral_segment(b0_u: float, b1_u: float, t0: float, t1: float) -> float:
    if abs(b1_u) < 1e-12:
        return 0.0
    return b0_u * (math.cos(b1_u * t0) - math.cos(b1_u * t1)) / b1_u


def compute_time_rescaling(model: FullHawkesWithSine, theta: np.ndarray, t_start: float = 0.0, t_end: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ogata time-rescaling residuals for each dimension over [t_start, t_end].
    """
    if t_end is None:
        t_end = model.T

    mu, A, b0, b1 = model._unpack(theta)
    beta = model.beta

    # Initialize r at t_start
    r = np.array([0.0, 0.0], dtype=float)
    last_t = 0.0
    for t, dim in model.timeline:
        if t >= t_start:
            if t_start > last_t:
                r *= math.exp(-beta * (t_start - last_t))
            last_t = t_start
            break
        r *= math.exp(-beta * (t - last_t)) if t > last_t else 1.0
        r[dim] += 1.0
        last_t = t
    else:
        if t_start > last_t:
            r *= math.exp(-beta * (t_start - last_t))
            last_t = t_start

    acc = np.array([0.0, 0.0], dtype=float)
    res0: List[float] = []
    res1: List[float] = []

    for t, dim in model.timeline:
        if t <= t_start:
            continue
        if t > t_end:
            break
        dt = t - last_t
        if dt > 0:
            decay_factor = math.exp(-beta * dt)
            for u in (0, 1):
                base_int = mu[u] * dt + _sine_integral_segment(b0[u], b1[u], last_t, t)
                exc = float(A[u].dot(r)) * (1.0 - decay_factor) / beta
                acc[u] += base_int + exc
            r *= decay_factor
        # event at t
        if dim == 0:
            res0.append(float(acc[0]))
            acc[0] = 0.0
        else:
            res1.append(float(acc[1]))
            acc[1] = 0.0
        r[dim] += 1.0
        last_t = t

    return np.asarray(res0, dtype=float), np.asarray(res1, dtype=float)


def plot_gof_time_rescaling(res0: np.ndarray, res1: np.ndarray, title_prefix: str = "", save_path: Optional[str] = None) -> None:
    def qq_plot(ax, res: np.ndarray, title: str):
        if len(res) == 0:
            ax.text(0.5, 0.5, "No events", ha='center')
            return
        x = np.sort(res)
        n = len(x)
        p = (np.arange(1, n + 1) - 0.5) / n
        q = -np.log(1.0 - p)
        ax.plot(q, x, 'o', ms=3, alpha=0.6)
        lim = max(q.max(), x.max())
        ax.plot([0, lim], [0, lim], 'k--', lw=1)
        ax.set_title(title)
        ax.set_xlabel('Exp(1) theoretical quantiles')
        ax.set_ylabel('Empirical quantiles')

    def hist_plot(ax, res: np.ndarray, title: str):
        if len(res) == 0:
            ax.text(0.5, 0.5, "No events", ha='center')
            return
        ax.hist(res, bins=50, density=True, alpha=0.4, color='C0', edgecolor='k')
        xs = np.linspace(0, max(np.percentile(res, 99), 1.0), 200)
        ax.plot(xs, np.exp(-xs), 'r-', lw=1.5, label='Exp(1) pdf')
        ax.set_title(title)
        ax.legend()

    def pp_plot(ax, res: np.ndarray, title: str):
        if len(res) == 0:
            ax.text(0.5, 0.5, "No events", ha='center')
            return
        u = 1.0 - np.exp(-res)
        u_sorted = np.sort(u)
        y = np.arange(1, len(u_sorted) + 1) / len(u_sorted)
        ax.plot(u_sorted, y, 'o', ms=3, alpha=0.6)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel('Uniform(0,1) theoretical')
        ax.set_ylabel('Empirical CDF')

    plt.figure(figsize=(12, 6))
    for row, res, name in [(0, res0, 'Dim 0'), (1, res1, 'Dim 1')]:
        ax1 = plt.subplot(2, 3, row * 3 + 1)
        hist_plot(ax1, res, f"{title_prefix} {name} — Residual hist vs Exp(1)")
        ax2 = plt.subplot(2, 3, row * 3 + 2)
        qq_plot(ax2, res, f"{title_prefix} {name} — QQ Exp(1)")
        ax3 = plt.subplot(2, 3, row * 3 + 3)
        pp_plot(ax3, res, f"{title_prefix} {name} — PP Uniform")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def simple_loglik_interval(events_2d: List[np.ndarray], T: float, mu: np.ndarray, A: np.ndarray, beta: float, t_start: float, t_end: float) -> float:
    """Validation log-likelihood for the simple Hawkes (no exogenous term)."""
    merged = [(float(t), 0) for t in events_2d[0]] + [(float(t), 1) for t in events_2d[1]]
    merged.sort(key=lambda x: x[0])
    eps = 1e-12

    # Initialize r at t_start
    r = np.array([0.0, 0.0], dtype=float)
    last_t = 0.0
    for t, dim in merged:
        if t >= t_start:
            if t_start > last_t:
                r *= math.exp(-beta * (t_start - last_t))
            last_t = t_start
            break
        r *= math.exp(-beta * (t - last_t)) if t > last_t else 1.0
        r[dim] += 1.0
        last_t = t
    else:
        if t_start > last_t:
            r *= math.exp(-beta * (t_start - last_t))
            last_t = t_start

    log_sum = 0.0
    integral_mu = float(np.sum(mu) * (t_end - t_start))
    cur_t = t_start
    for t, dim in merged:
        if t <= t_start:
            continue
        if t > t_end:
            break
        dt = t - cur_t
        if dt > 0:
            decay_factor = math.exp(-beta * dt)
            integral_exc_seg = float(np.sum(A.dot(r) * (1.0 - decay_factor) / beta))
            r *= decay_factor
        else:
            integral_exc_seg = 0.0
        lambdas = mu + A.dot(r)
        lam_dim = max(lambdas[dim], eps)
        log_sum += math.log(lam_dim)
        r[dim] += 1.0
        cur_t = t
    dt_tail = t_end - cur_t
    integral_exc_tail = float(np.sum(A.dot(r) * (1.0 - math.exp(-beta * dt_tail)) / beta)) if dt_tail > 0 else 0.0
    return log_sum - (integral_mu + integral_exc_tail)


def plot_pp_validation(model: FullHawkesWithSine, theta: np.ndarray, t_start: float, t_end: float, title_prefix: str = "", save_path: Optional[str] = None) -> None:
    res0, res1 = compute_time_rescaling(model, theta, t_start=t_start, t_end=t_end)
    plt.figure(figsize=(10, 4))
    for idx, (res, name) in enumerate([(res0, 'Dim 0'), (res1, 'Dim 1')], start=1):
        u = 1.0 - np.exp(-res)
        u_sorted = np.sort(u)
        y = np.arange(1, len(u_sorted) + 1) / max(len(u_sorted), 1)
        ax = plt.subplot(1, 2, idx)
        if len(u_sorted) > 0:
            ax.plot(u_sorted, y, 'o', ms=3, alpha=0.6)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title_prefix} {name} PP (val)")
        ax.set_xlabel('Uniform(0,1) theoretical')
        ax.set_ylabel('Empirical CDF')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def simulate_full_model(model: FullHawkesWithSine, theta: np.ndarray, horizon: float, max_events: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    mu, A, b0, b1 = model._unpack(theta)
    beta = model.beta
    t = 0.0
    r = np.array([0.0, 0.0], dtype=float)
    out = [[], []]

    while t < horizon and (len(out[0]) + len(out[1])) < max_events:
        # Upper bound for intensities using |sin|<=1
        lam_bar_vec = mu + np.abs(b0) + A.dot(r)
        lam_bar_vec = np.maximum(lam_bar_vec, 1e-8)
        lam_bar_sum = float(np.sum(lam_bar_vec))
        if lam_bar_sum <= 0:
            break
        dt = np.random.exponential(1.0 / lam_bar_sum)
        t_next = t + dt
        if t_next > horizon:
            break
        decay_factor = math.exp(-beta * dt)
        r *= decay_factor
        # Actual intensities at t_next
        lam_vec = mu + b0 * np.sin(b1 * t_next) + A.dot(r)
        lam_vec = np.maximum(lam_vec, 0.0)
        lam_sum = float(np.sum(lam_vec))
        u = np.random.rand()
        if u * lam_bar_sum <= lam_sum and lam_sum > 0:
            # Accept event, choose dimension
            v = np.random.rand() * lam_sum
            if v <= lam_vec[0]:
                out[0].append(t_next)
                r[0] += 1.0
            else:
                out[1].append(t_next)
                r[1] += 1.0
            t = t_next
        else:
            t = t_next
            continue

    return np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)


def plot_rate_comparison(true_events: List[np.ndarray], sim_events: Tuple[np.ndarray, np.ndarray], T: float, title_prefix: str = "", save_path: Optional[str] = None) -> None:
    bins = 100
    edges = np.linspace(0.0, T, bins + 1)
    width = edges[1] - edges[0]
    true_rates = [np.histogram(true_events[d], bins=edges)[0] / width for d in (0, 1)]
    sim_rates = [np.histogram(sim_events[d], bins=edges)[0] / width for d in (0, 1)]

    plt.figure(figsize=(12, 4))
    for d in (0, 1):
        ax = plt.subplot(1, 2, d + 1)
        ax.plot(edges[:-1], true_rates[d], label='True rate', lw=1.2)
        ax.plot(edges[:-1], sim_rates[d], label='Sim rate', lw=1.2)
        ax.set_title(f"{title_prefix} Dim {d} — rate comparison")
        ax.set_xlabel('time')
        ax.set_ylabel('events per unit time')
        ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# -----------------------------
# Beta selection utilities
# -----------------------------

def tune_beta_for_full(events_2d: List[np.ndarray], T: float, train_events: List[np.ndarray], t_split: float, candidate_betas: np.ndarray, init_mu: np.ndarray, init_A: np.ndarray) -> Tuple[float, FullFitResult, float]:
    """
    Grid search decay beta for the full model by training on [0, t_split]
    and selecting the beta with the highest validation log-likelihood on
    (t_split, T]. Returns (best_beta, best_train_fit, best_val_ll).
    """
    best_beta: Optional[float] = None
    best_fit: Optional[FullFitResult] = None
    best_ll: Optional[float] = None
    for beta in candidate_betas:
        # fit on train
        m_train = FullHawkesWithSine(train_events, t_split, decay=float(beta))
        fit_train = m_train.fit(init_mu=init_mu, init_A=init_A)
        # evaluate on validation using evaluator with full timeline
        evaluator = FullHawkesWithSine(events_2d, T, decay=float(beta))
        ll_val = evaluator.log_likelihood_interval(fit_train.params, t_split, T)
        if (best_ll is None) or (ll_val > best_ll):
            best_ll = float(ll_val)
            best_beta = float(beta)
            best_fit = fit_train
    assert best_beta is not None and best_fit is not None and best_ll is not None
    return best_beta, best_fit, best_ll


def grid_search_in_sample_full(events_2d: List[np.ndarray], T: float, beta_grid: np.ndarray, b1_grid: np.ndarray, init_mu: np.ndarray, init_A: np.ndarray) -> Tuple[float, float, FullFitResult]:
    """
    In-sample grid search for (beta, shared b1). For each pair, fit with b1 fixed
    over the full sample and select the best by full-sample log-likelihood.
    Returns (best_beta, best_b1, best_fit).
    """
    best: Optional[Tuple[float, float, FullFitResult]] = None
    for beta in beta_grid:
        model_beta = FullHawkesWithSine(events_2d, T, decay=float(beta))
        for b1 in b1_grid:
            fit_res = model_beta.fit_with_fixed_b1(init_mu=init_mu, init_A=init_A, fixed_b1_shared=float(b1))
            score = fit_res.loglik
            if (best is None) or (score > best[2].loglik):
                best = (float(beta), float(b1), fit_res)
    assert best is not None
    return best[0], best[1], best[2]


def in_sample_forecast_and_metrics(events_2d: List[np.ndarray], T: float, params: np.ndarray, beta: float, bins: int = 200, n_paths: int = 10, seed: int = 42, out_prefix: Optional[str] = None) -> dict:
    """
    Simulate n_paths using fitted params to estimate expected counts per bin.
    Compute MAE/RMSE/MAPE and Pearson r for counts. Save plot if out_prefix provided.
    """
    model = FullHawkesWithSine(events_2d, T, decay=beta)
    rng = np.random.RandomState(seed)
    edges = np.linspace(0.0, T, bins + 1)
    width = edges[1] - edges[0]
    true_counts = [np.histogram(events_2d[d], bins=edges)[0].astype(float) for d in (0, 1)]

    sim_counts_accum = [np.zeros(bins, dtype=float), np.zeros(bins, dtype=float)]
    for k in range(n_paths):
        # different seed per path
        np.random.seed(int(rng.randint(0, 2**31-1)))
        sim0, sim1 = simulate_full_model(model, params, horizon=T, max_events=1200000)
        sim_counts_accum[0] += np.histogram(sim0, bins=edges)[0]
        sim_counts_accum[1] += np.histogram(sim1, bins=edges)[0]

    pred_counts = [sim_counts_accum[d] / float(n_paths) for d in (0, 1)]

    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        denom = np.maximum(y_true, 1e-8)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)))
        # Pearson r
        if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
            r = 0.0
        else:
            r = float(np.corrcoef(y_true, y_pred)[0, 1])
        return {"mae": mae, "rmse": rmse, "mape": mape, "r": r}

    m0 = metrics(true_counts[0], pred_counts[0])
    m1 = metrics(true_counts[1], pred_counts[1])

    if out_prefix is not None:
        plt.figure(figsize=(12, 4))
        for d in (0, 1):
            ax = plt.subplot(1, 2, d + 1)
            ax.plot(edges[:-1], true_counts[d] / width, label='True rate', lw=1.2)
            ax.plot(edges[:-1], pred_counts[d] / width, label='Pred rate (MC mean)', lw=1.2)
            ax.set_title(f"In-sample forecast — Dim {d}")
            ax.set_xlabel('time')
            ax.set_ylabel('events per unit time')
            ax.legend()
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(os.path.join("results", f"{out_prefix}_insample_forecast.png"), dpi=200, bbox_inches='tight')
        plt.close()

    return {"dim0": m0, "dim1": m1}

# -----------------------------
# Orchestration
# -----------------------------

def run_comparison(data_path: str) -> None:
    events_2d, T = load_events_2d(data_path)

    # Baseline fit with a small decay grid
    decay_grid = np.logspace(-2, 0.7, 8)  # ~[0.01 .. 5]
    baseline = fit_simple_baseline_tick(events_2d, decay_grid)

    # Strategy for beta in the full model (ignored if EXCITATION=off)
    # Options: "baseline" use baseline.decay; "fixed" use FIXED_BETA; "grid" tune on 70/30 split
    BETA_STRATEGY = os.environ.get("BETA_STRATEGY", "baseline")
    EXCITATION = os.environ.get("EXCITATION", "on")  # "on" | "off"
    FIXED_BETA = float(os.environ.get("FIXED_BETA", baseline.decay))
    BETA_GRID_FULL = np.logspace(-2, 1.0, 10)

    # Prepare temporal split (used for grid/fixed reporting as well)
    t_split = 0.7 * T
    train_events = [ev[ev < t_split] for ev in events_2d]

    if EXCITATION == "off":
        # no-kernel: beta irrelevant for fitting; keep for record using baseline.decay
        beta_full = float(baseline.decay)
        full = FullHawkesWithSine(events_2d, T, decay=beta_full)
        full_fit = full.fit_no_kernel()
        full_train_fit_best = full_fit  # for validation/plots reuse
        baseline_train = fit_simple_baseline_tick(train_events, decay_grid)
        best_val_ll = float("nan")
    elif BETA_STRATEGY == "grid":
        baseline_train = fit_simple_baseline_tick(train_events, decay_grid)
        tuned_beta, full_train_fit_best, best_val_ll = tune_beta_for_full(
            events_2d, T, train_events, t_split, BETA_GRID_FULL, baseline_train.mu, baseline_train.adjacency
        )
        beta_full = tuned_beta
    elif BETA_STRATEGY == "fixed":
        beta_full = FIXED_BETA
        baseline_train = fit_simple_baseline_tick(train_events, decay_grid)
        full_train = FullHawkesWithSine(train_events, t_split, decay=beta_full)
        full_train_fit_best = full_train.fit(init_mu=baseline_train.mu, init_A=baseline_train.adjacency)
        best_val_ll = FullHawkesWithSine(events_2d, T, decay=beta_full).log_likelihood_interval(full_train_fit_best.params, t_split, T)
    else:
        # default: baseline decay
        beta_full = baseline.decay
        baseline_train = fit_simple_baseline_tick(train_events, decay_grid)
        full_train = FullHawkesWithSine(train_events, t_split, decay=beta_full)
        full_train_fit_best = full_train.fit(init_mu=baseline_train.mu, init_A=baseline_train.adjacency)
        best_val_ll = FullHawkesWithSine(events_2d, T, decay=beta_full).log_likelihood_interval(full_train_fit_best.params, t_split, T)

    # Fit full model on the entire horizon with selected beta (if excitation on)
    # Helper: choose fitting logic depending on FIXED_B1 (optional shared frequency)
    def fit_with_optional_fixed_b1(model: FullHawkesWithSine, init_mu: np.ndarray, init_A: np.ndarray) -> FullFitResult:
        fixed_b1_env = os.environ.get("FIXED_B1")
        if fixed_b1_env is not None and len(str(fixed_b1_env).strip()) > 0:
            return model.fit_with_fixed_b1(init_mu=init_mu, init_A=init_A, fixed_b1_shared=float(fixed_b1_env))
        return model.fit(init_mu=init_mu, init_A=init_A)

    if EXCITATION != "off":
        full = FullHawkesWithSine(events_2d, T, decay=beta_full)
        full_fit = fit_with_optional_fixed_b1(full, baseline.mu, baseline.adjacency)

    # Metrics
    # For the baseline, number of estimated params = 6 (mu:2, A:4). Decay is treated as tuned hyperparam
    k_simple = 6
    aic_simple = 2 * k_simple - 2 * baseline.loglik

    # Likelihood-ratio test
    lr = 2.0 * (full_fit.loglik - baseline.loglik)
    df = 4  # additional params: beta0(2) + beta1(2)
    p_value = float(chi2.sf(lr, df))

    # Report
    print("=== Data ===")
    print(f"Events dim0: {len(events_2d[0])}, dim1: {len(events_2d[1])}, T={T:.4f}")
    print()
    print("=== Simple Hawkes (tick.HawkesExpKern) ===")
    print(f"best decay: {baseline.decay:.6g}")
    print(f"mu: {baseline.mu}")
    print(f"adjacency (alpha):\n{baseline.adjacency}")
    print(f"log-likelihood: {baseline.loglik:.6f}")
    print(f"AIC: {aic_simple:.6f}")
    print()
    print("=== Full Hawkes + sine f(t) ===")
    print(f"excitation: {EXCITATION}, beta strategy: {BETA_STRATEGY}, beta used: {beta_full:.6g}")
    print(f"success: {full_fit.success}")
    mu_f, A_f, b0_f, b1_f = FullHawkesWithSine._unpack(full_fit.params)
    print(f"mu: {mu_f}")
    print(f"adjacency (alpha):\n{A_f}")
    print(f"beta0: {b0_f}, beta1: {b1_f}")
    print(f"log-likelihood: {full_fit.loglik:.6f}")
    print(f"AIC: {full_fit.aic:.6f}")
    print()

    # Summarize and persist parameters for comparison
    os.makedirs("results", exist_ok=True)
    def full_params_as_dict(tag: str, params: np.ndarray, beta_val: float) -> dict:
        mu_p, A_p, b0_p, b1_p = FullHawkesWithSine._unpack(params)
        return {
            "tag": tag,
            "beta": float(beta_val),
            "mu": mu_p.tolist(),
            "A": A_p.tolist(),
            "beta0": b0_p.tolist(),
            "beta1": b1_p.tolist(),
            "fixed_b1_used": float(os.environ["FIXED_B1"]) if os.environ.get("FIXED_B1") else None,
        }

    train_params_dict = full_params_as_dict("full_fit_all", full_fit.params, beta_full)
    with open(os.path.join("results", "params_full_train.json"), "w", encoding="utf-8") as f:
        json.dump(train_params_dict, f, ensure_ascii=False, indent=2)

    print("Parameters used for rate-comparison simulation (from full fit):")
    print(train_params_dict)
    if EXCITATION != "off":
        print("=== Likelihood-ratio test (full vs simple) ===")
        print(f"LR statistic: {lr:.6f}, df={df}, p-value: {p_value:.3e}")
        if p_value < 0.05:
            print("Result: Full model significantly improves over simple baseline (p < 0.05)")
        else:
            print("Result: No significant improvement detected at 5% level (consider different f(t) or decay grid)")

    # ------------------
    # GOF: time-rescaling + QQ & Uniform plots on full model
    # ------------------
    residuals0, residuals1 = compute_time_rescaling(full, full_fit.params)
    plot_gof_time_rescaling(residuals0, residuals1, title_prefix="Full model", save_path=os.path.join("results", "gof_full.png"))

    # ------------------
    # 70/30 temporal split validation
    # ------------------
    # Build val set (already have train split above)
    val_events = [ev[(ev >= t_split) & (ev <= T)] for ev in events_2d]

    # Evaluate on validation interval [t_split, T]
    # Baseline manual likelihood on val
    ll_simple_val = simple_loglik_interval(events_2d, T, baseline_train.mu, baseline_train.adjacency, baseline_train.decay, t_split, T)
    ll_full_val = full.log_likelihood_interval(full_train_fit_best.params, t_split, T) if EXCITATION != "off" else float('nan')
    print()
    print("=== Temporal validation (70/30 split) ===")
    print(f"Validation log-lik simple: {ll_simple_val:.6f}")
    print(f"Validation log-lik full  : {ll_full_val:.6f}")

    # PP-plot on validation for each dimension
    plot_pp_validation(full, full_train_fit_best.params, t_split, T, title_prefix="Full model (val)", save_path=os.path.join("results", "pp_validation_full.png"))

    # ------------------
    # Simulation-based rate check
    # ------------------
    sim_dim0, sim_dim1 = simulate_full_model(full, full_fit.params, horizon=T, max_events=1200000)
    plot_rate_comparison(events_2d, (sim_dim0, sim_dim1), T, title_prefix="Full model", save_path=os.path.join("results", "rate_comparison_full.png"))

    # ------------------
    # In-sample forecasting via grid search over (beta, b1)
    # 支持仅用前 INSAMPLE_FRAC * T 的数据进行快速选择
    # ------------------
    print()
    # Optionally run an in-sample grid search to approximate a "simulation-stage"
    # target parameter selection via quick grid over (beta, shared b1). This both
    # accentuates the sine exogenous term and provides parameters for MC-based
    # in-sample forecasting.
    INSAMPLE_ENABLE = os.environ.get("INSAMPLE", "1") != "0"
    insample_params_dict = None
    insample_metrics = None
    best_beta_is = None
    best_b1_is = None
    if INSAMPLE_ENABLE:
        print("=== In-sample forecasting (grid search over beta & b1) ===")
        insample_frac = float(os.environ.get("INSAMPLE_FRAC", "1.0"))
        beta_pts = int(os.environ.get("INSAMPLE_BETA_POINTS", "5"))
        b1_pts = int(os.environ.get("INSAMPLE_B1_POINTS", "5"))
        n_paths = int(os.environ.get("INSAMPLE_N_PATHS", "3"))
        beta_grid_insample = np.logspace(-2, 1.0, beta_pts)
        b1_grid_insample = np.linspace(0.5, 1.5, b1_pts)  # shared frequency candidates
        # Use baseline as initialization
        if insample_frac < 1.0:
            T_q = insample_frac * T
            events_q = [ev[ev <= T_q] for ev in events_2d]
            best_beta_is, best_b1_is, best_fit_is = grid_search_in_sample_full(
                events_q, T_q, beta_grid_insample, b1_grid_insample, baseline.mu, baseline.adjacency
            )
        else:
            best_beta_is, best_b1_is, best_fit_is = grid_search_in_sample_full(
                events_2d, T, beta_grid_insample, b1_grid_insample, baseline.mu, baseline.adjacency
            )
        print(f"best beta (in-sample): {best_beta_is:.6g}, best shared b1: {best_b1_is:.6g}")
        mu_is, A_is, b0_is, b1_is = FullHawkesWithSine._unpack(best_fit_is.params)
        print(f"in-sample mu: {mu_is}")
        print(f"in-sample A: \n{A_is}")
        print(f"in-sample beta0: {b0_is}, b1: {b1_is}")

        # Forecast by MC simulation and compute errors
        metrics = in_sample_forecast_and_metrics(
            events_2d, T, best_fit_is.params, beta=best_beta_is, bins=200, n_paths=n_paths, out_prefix="full"
        )
        print("In-sample errors:")
        print(f"Dim0: {metrics['dim0']}")
        print(f"Dim1: {metrics['dim1']}")

        # Save and print parameters used for in-sample forecast simulation
        insample_params_dict = full_params_as_dict("full_fit_insample_grid", best_fit_is.params, best_beta_is)
        insample_metrics = metrics
        with open(os.path.join("results", "params_full_insample.json"), "w", encoding="utf-8") as f:
            json.dump(insample_params_dict, f, ensure_ascii=False, indent=2)
        print("Parameters used for in-sample forecast simulation (from grid-search fit):")
        print(insample_params_dict)

    # Combined comparison file, clearly contrasting the "simulation-stage" (grid) target
    # parameters and the full training-stage fitted parameters.
    compare_all = {
        "baseline": {
            "decay_beta": float(baseline.decay),
            "mu": baseline.mu.tolist(),
            "A": baseline.adjacency.tolist(),
            "loglik": float(baseline.loglik),
            "aic": float(2 * 6 - 2 * baseline.loglik),
        },
        "full_training": {
            **train_params_dict,
            "loglik": float(full_fit.loglik),
            "aic": float(full_fit.aic),
        },
    }
    if insample_params_dict is not None:
        compare_all["simulation_stage"] = {
            **insample_params_dict,
            "selection": {
                "method": "grid_search_in_sample",
                "best_beta": float(best_beta_is),
                "best_shared_b1": float(best_b1_is),
            },
            "forecast_metrics": insample_metrics,
        }
    with open(os.path.join("results", "params_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(compare_all, f, ensure_ascii=False, indent=2)

    # Extra concise report file for downstream consumption
    with open(os.path.join("results", "report.json"), "w", encoding="utf-8") as f:
        json.dump(compare_all, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Default relative path; adjust if needed
    run_comparison("events_100k.json")


