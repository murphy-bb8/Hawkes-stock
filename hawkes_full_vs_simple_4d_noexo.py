import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from tick.hawkes import HawkesExpKern


def load_events_4d(path: str) -> Tuple[List[np.ndarray], float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dims = [[] for _ in range(4)]
    for e in raw:
        t = float(e["t"])
        i = int(e["i"])
        if 0 <= i < 4:
            dims[i].append(t)
    for d in range(4):
        dims[d].sort()
    T = max([dims[d][-1] if len(dims[d]) > 0 else 0.0 for d in range(4)])
    return [np.asarray(dims[d], dtype=float) for d in range(4)], T


@dataclass
class BaselineResult:
    decay: float
    mu: np.ndarray
    adjacency: np.ndarray
    loglik: float


def fit_simple_baseline_tick_4d(events_4d: List[np.ndarray], decay_grid: np.ndarray) -> BaselineResult:
    best = None
    for decay in decay_grid:
        decays_mat = np.full((4, 4), float(decay), dtype=float)
        learner = HawkesExpKern(decays=decays_mat, verbose=False)
        learner.fit([events_4d])
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


@dataclass
class FullFitResult:
    params: np.ndarray
    loglik: float
    aic: float
    success: bool


class FullHawkes4DNoExo:
    def __init__(self, events_4d: List[np.ndarray], T: float, decay: float):
        self.events = [np.asarray(ev, dtype=float) for ev in events_4d]
        self.T = float(T)
        self.beta = float(decay)
        self.reg_l2_mu = float(os.environ.get("REG_L2_MU", "0.0"))
        self.reg_l2_A = float(os.environ.get("REG_L2_A", "0.0"))

        self.S = [float(np.sum(1.0 - np.exp(-self.beta * (self.T - self.events[j])))) for j in range(4)]

        merged = []
        for d in range(4):
            merged.extend([(float(t), d) for t in self.events[d]])
        merged.sort(key=lambda x: x[0])
        self.timeline = merged

    @staticmethod
    def _pack(mu: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.concatenate([mu.ravel(), A.ravel()])

    @staticmethod
    def _unpack(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = theta[0:4]
        A = theta[4:20].reshape(4, 4)
        return mu, A

    def log_likelihood(self, theta: np.ndarray) -> float:
        mu, A = self._unpack(theta)
        eps = 1e-12

        log_sum = 0.0
        r = np.zeros(4, dtype=float)
        last_t = 0.0
        for t, dim in self.timeline:
            decay_factor = math.exp(-self.beta * (t - last_t)) if t > last_t else 1.0
            r *= decay_factor
            lambdas = mu + A.dot(r)
            lam_dim = max(lambdas[dim], eps)
            log_sum += math.log(lam_dim)
            r[dim] += 1.0
            last_t = t

        integral_mu = float(np.sum(mu) * self.T)
        excitation_int = np.array(self.S, dtype=float) / self.beta
        integral_exc = float(np.sum(A * excitation_int[np.newaxis, :]))

        return log_sum - (integral_mu + integral_exc)

    def _reg_penalty_full(self, theta: np.ndarray) -> float:
        mu, A = self._unpack(theta)
        pen = 0.0
        if self.reg_l2_mu > 0:
            pen += float(self.reg_l2_mu) * float(np.sum(mu * mu))
        if self.reg_l2_A > 0:
            pen += float(self.reg_l2_A) * float(np.sum(A * A))
        return pen

    def log_likelihood_interval(self, theta: np.ndarray, t_start: float, t_end: float) -> float:
        mu, A = self._unpack(theta)
        eps = 1e-12

        r = np.zeros(4, dtype=float)
        last_t = 0.0
        for t, dim in self.timeline:
            if t >= t_start:
                if t_start > last_t:
                    r *= math.exp(-self.beta * (t_start - last_t))
                last_t = t_start
                break
            r *= math.exp(-self.beta * (t - last_t)) if t > last_t else 1.0
            r[dim] += 1.0
            last_t = t
        else:
            if t_start > last_t:
                r *= math.exp(-self.beta * (t_start - last_t))
                last_t = t_start

        log_sum = 0.0
        integral_mu = float(np.sum(mu) * (t_end - t_start))
        cur_t = t_start
        integral_exc = 0.0
        for t, dim in self.timeline:
            if t <= t_start:
                continue
            if t > t_end:
                break
            dt = t - cur_t
            if dt > 0:
                integral_exc += float(np.sum(A.dot(r) * (1.0 - np.exp(-self.beta * dt)) / self.beta))
            r *= math.exp(-self.beta * dt) if dt > 0 else 1.0

            lambdas = mu + A.dot(r)
            lam_dim = max(lambdas[dim], eps)
            log_sum += math.log(lam_dim)
            r[dim] += 1.0
            cur_t = t

        dt_tail = t_end - cur_t
        if dt_tail > 0:
            integral_exc += float(np.sum(A.dot(r) * (1.0 - math.exp(-self.beta * dt_tail)) / self.beta))

        return log_sum - (integral_mu + integral_exc)

    def fit(self, init_mu: np.ndarray, init_A: np.ndarray) -> FullFitResult:
        theta0 = self._pack(init_mu, init_A)
        bounds = []
        bounds.extend([(1e-8, None)] * 4)
        bounds.extend([(0.0, None)] * 16)

        def obj(theta: np.ndarray) -> float:
            return -self.log_likelihood(theta) + self._reg_penalty_full(theta)

        maxiter = int(os.environ.get("MLE_MAXITER", "500"))
        res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-6})
        k_params = 20
        ll = -float(res.fun)
        aic = 2 * k_params - 2 * ll
        return FullFitResult(params=res.x, loglik=ll, aic=aic, success=bool(res.success))


def _spectral_radius(mat: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(mat)
    return float(np.max(np.real(eigvals)))


def tune_beta_for_full_4d(events_4d: List[np.ndarray],
                          T: float,
                          train_events: List[np.ndarray],
                          t_split: float,
                          candidate_betas: np.ndarray,
                          init_mu: np.ndarray,
                          init_A: np.ndarray) -> Tuple[float, FullFitResult, float]:
    best_beta: Optional[float] = None
    best_fit: Optional[FullFitResult] = None
    best_ll: Optional[float] = None
    for beta in candidate_betas:
        m_train = FullHawkes4DNoExo(train_events, t_split, decay=float(beta))
        fit_train = m_train.fit(init_mu=init_mu, init_A=init_A)
        evaluator = FullHawkes4DNoExo(events_4d, T, decay=float(beta))
        ll_val = evaluator.log_likelihood_interval(fit_train.params, t_split, T)
        if (best_ll is None) or (ll_val > best_ll):
            best_ll = float(ll_val)
            best_beta = float(beta)
            best_fit = fit_train
    assert best_beta is not None and best_fit is not None and best_ll is not None
    return best_beta, best_fit, best_ll


def simple_loglik_interval_4d(events_4d: List[np.ndarray], T: float, mu: np.ndarray, A: np.ndarray, beta: float, t_start: float, t_end: float) -> float:
    merged = []
    for d in range(4):
        merged.extend([(float(t), d) for t in events_4d[d]])
    merged.sort(key=lambda x: x[0])
    eps = 1e-12

    r = np.zeros(4, dtype=float)
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
    integral_exc = 0.0
    for t, dim in merged:
        if t <= t_start:
            continue
        if t > t_end:
            break
        dt = t - cur_t
        if dt > 0:
            decay_factor = math.exp(-beta * dt)
            integral_exc += float(np.sum(A.dot(r) * (1.0 - decay_factor) / beta))
            r *= decay_factor
        lambdas = mu + A.dot(r)
        lam_dim = max(lambdas[dim], eps)
        log_sum += math.log(lam_dim)
        r[dim] += 1.0
        cur_t = t
    dt_tail = t_end - cur_t
    if dt_tail > 0:
        integral_exc += float(np.sum(A.dot(r) * (1.0 - math.exp(-beta * dt_tail)) / beta))
    return log_sum - (integral_mu + integral_exc)


def run_comparison_4d_noexo(data_path: str) -> None:
    events_4d, T = load_events_4d(data_path)

    dmin = float(os.environ.get("DECAY_MIN", "0.1"))
    dmax = float(os.environ.get("DECAY_MAX", "2.0"))
    decay_grid = np.logspace(math.log10(dmin), math.log10(dmax), 8)
    baseline = fit_simple_baseline_tick_4d(events_4d, decay_grid)

    t_split = 0.7 * T
    train_events = [ev[ev < t_split] for ev in events_4d]
    baseline_train = fit_simple_baseline_tick_4d(train_events, decay_grid) if sum(len(ev) for ev in train_events) > 4 else baseline

    beta_strategy = os.environ.get("BETA_STRATEGY", "baseline")
    fixed_beta = float(os.environ.get("FIXED_BETA", baseline.decay))

    if beta_strategy == "grid" and sum(len(ev) for ev in train_events) > 4:
        bmin = float(os.environ.get("BETA_GRID_MIN", str(dmin)))
        bmax = float(os.environ.get("BETA_GRID_MAX", str(dmax)))
        bpts = int(os.environ.get("BETA_GRID_POINTS", "8"))
        beta_grid = np.logspace(math.log10(bmin), math.log10(bmax), bpts)
        beta_used, _, _ = tune_beta_for_full_4d(events_4d, T, train_events, t_split, beta_grid, baseline_train.mu, baseline_train.adjacency)
    else:
        beta_used = baseline.decay if beta_strategy != "fixed" else fixed_beta

    full = FullHawkes4DNoExo(events_4d, T, decay=beta_used)
    full_fit = full.fit(init_mu=baseline.mu, init_A=baseline.adjacency)

    ll_simple_val = simple_loglik_interval_4d(events_4d, T, baseline_train.mu, baseline_train.adjacency, baseline_train.decay, t_split, T)
    ll_full_val = full.log_likelihood_interval(full_fit.params, t_split, T)

    k_simple = 4 + 16
    aic_simple = 2 * k_simple - 2 * baseline.loglik

    mu_f, A_f = FullHawkes4DNoExo._unpack(full_fit.params)
    branching_ratio = float(_spectral_radius(A_f) / beta_used) if beta_used > 0 else float("inf")
    constraint_ok = bool(branching_ratio < 1.0)

    results = {
        "baseline": {
            "decay": float(baseline.decay),
            "mu": baseline.mu.tolist(),
            "A": baseline.adjacency.tolist(),
            "loglik": float(baseline.loglik),
            "aic": float(aic_simple),
        },
        "full": {
            "beta": float(beta_used),
            "mu": mu_f.tolist(),
            "A": A_f.tolist(),
            "loglik": float(full_fit.loglik),
            "aic": float(full_fit.aic),
            "branching_ratio": branching_ratio,
            "constraint_ok": constraint_ok,
        },
        "validation": {
            "t_split": float(t_split),
            "ll_simple": float(ll_simple_val),
            "ll_full": float(ll_full_val),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/comparison_4d_noexo.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    try:
        out_tag = os.environ.get("OUT_TAG")
        if not out_tag:
            base = os.path.basename(data_path)
            tag = os.path.splitext(base)[0]
            if tag.startswith("events_"):
                out_tag = tag[len("events_"):]
            else:
                out_tag = tag
        with open(f"results/comparison_4d_noexo_{out_tag}.json", "w", encoding="utf-8") as f2:
            json.dump(results, f2, ensure_ascii=False, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    run_comparison_4d_noexo("events_100k.json")
