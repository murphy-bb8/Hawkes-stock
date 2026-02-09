# =============================================
# hawkes_full_1d.py
# 一维 Hawkes 模型 + 正弦外生项 f(Xt)=beta0*sin(beta1*t)
# =============================================

import os
import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize
from scipy.stats import chi2
from tick.hawkes import HawkesExpKern
import matplotlib.pyplot as plt


# -----------------------------
# 数据加载（单维）
# -----------------------------
def load_events_1d(path: str):
    """
    从 json 加载事件，兼容两种格式：
    1) [{"t": ...}] 仅含时间；
    2) [{"t": ..., "i": 0}] 含维度标签（只取 i==0）。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 兼容 t-only 与 t+i 两种结构
    events = []
    for e in raw:
        try:
            if "i" in e and int(e.get("i", 0)) != 0:
                continue
            events.append(float(e["t"]))
        except Exception:
            continue
    events.sort()
    T = events[-1] if len(events) > 0 else 0.0
    return np.asarray(events, dtype=float), T


# -----------------------------
# Baseline 模型（tick）
# -----------------------------
@dataclass
class Baseline1D:
    decay: float
    mu: float
    alpha: float
    loglik: float


def fit_baseline_tick(events: np.ndarray, decay_grid: np.ndarray) -> Baseline1D:
    best = None
    for decay in decay_grid:
        learner = HawkesExpKern(decays=float(decay), verbose=False)
        learner.fit([[events]])  # tick 期望 [list of realizations]
        ll = float(learner.score())
        if best is None or ll > best.loglik:
            best = Baseline1D(decay=float(decay),
                              mu=float(learner.baseline[0]),
                              alpha=float(learner.adjacency[0, 0]),
                              loglik=ll)
    return best


# -----------------------------
# Full 模型：含正弦外生项
# -----------------------------
@dataclass
class FullFitResult:
    params: np.ndarray
    loglik: float
    aic: float
    success: bool


class FullHawkes1DWithSine:
    """
    λ(t) = μ + α ∑ exp(-β(t-ti)) + β0 sin(β1 t)
    """

    def __init__(self, events: np.ndarray, T: float, beta: float):
        self.events = np.asarray(events, dtype=float)
        self.T = float(T)
        self.beta = float(beta)
        self.S = float(np.sum(1.0 - np.exp(-self.beta * (self.T - self.events))))  # ∫kernel
        # Optional L2 regularization strengths from env (default off)
        self.reg_l2_mu = float(os.environ.get("REG_L2_MU", "0.0"))
        self.reg_l2_alpha = float(os.environ.get("REG_L2_A", "0.0"))
        self.reg_l2_b0 = float(os.environ.get("REG_L2_B0", "0.0"))
        # Stability soft constraint: hinge penalty on alpha <= stab_ratio * beta
        self.reg_stab = float(os.environ.get("REG_STAB", "0.0"))
        self.stab_ratio = float(os.environ.get("STAB_RATIO", "1.0"))
        # Stationarity margin: enforce alpha <= (1 - margin) * beta
        # Set STAB_MARGIN to 0.01 or 0.0 to relax constraint (keep alpha < beta)
        self.stab_margin = float(os.environ.get("STAB_MARGIN", "0.01"))

    def log_likelihood(self, theta: np.ndarray) -> float:
        mu, alpha, beta0, beta1 = theta
        eps = 1e-12
        log_sum = 0.0
        r = 0.0
        last_t = 0.0

        for t in self.events:
            # 更新r
            r *= math.exp(-self.beta * (t - last_t))
            lam = mu + alpha * r + beta0 * math.sin(beta1 * t)
            lam = max(lam, eps)
            log_sum += math.log(lam)
            r += 1.0
            last_t = t

        # 积分项
        integral_mu = mu * self.T
        with np.errstate(divide='ignore', invalid='ignore'):
            integral_sine = beta0 * (1.0 - math.cos(beta1 * self.T)) / beta1 if abs(beta1) > 1e-12 else 0.0
        integral_exc = alpha * self.S / self.beta
        return log_sum - (integral_mu + integral_sine + integral_exc)

    def _reg_penalty(self, theta: np.ndarray) -> float:
        mu, alpha, beta0, _ = theta
        pen = 0.0
        if self.reg_l2_mu > 0:
            pen += self.reg_l2_mu * (mu * mu)
        if self.reg_l2_alpha > 0:
            pen += self.reg_l2_alpha * (alpha * alpha)
        if self.reg_l2_b0 > 0:
            pen += self.reg_l2_b0 * (beta0 * beta0)
        # stability hinge
        if self.reg_stab > 0:
            margin = alpha - self.stab_ratio * self.beta
            if margin > 0:
                pen += self.reg_stab * (margin * margin)
        return float(pen)

    def log_likelihood_interval(self, theta: np.ndarray, t_start: float, t_end: float) -> float:
        """LL on (t_start, t_end], conditioning on history up to t_start."""
        mu, alpha, beta0, beta1 = theta
        eps = 1e-12
        # Initialize r at t_start
        r = 0.0
        last_t = 0.0
        for t in self.events:
            if t >= t_start:
                if t_start > last_t:
                    r *= math.exp(-self.beta * (t_start - last_t))
                last_t = t_start
                break
            r *= math.exp(-self.beta * (t - last_t)) if t > last_t else 1.0
            r += 1.0
            last_t = t
        else:
            if t_start > last_t:
                r *= math.exp(-self.beta * (t_start - last_t))
                last_t = t_start

        log_sum = 0.0
        integral_mu = mu * (t_end - t_start)
        with np.errstate(divide='ignore', invalid='ignore'):
            sine_int = beta0 * (math.cos(beta1 * t_start) - math.cos(beta1 * t_end)) / beta1 if abs(beta1) > 1e-12 else 0.0

        cur_t = t_start
        integral_exc = 0.0
        for t in self.events:
            if t <= t_start:
                continue
            if t > t_end:
                break
            dt = t - cur_t
            if dt > 0:
                decay_factor = math.exp(-self.beta * dt)
                integral_exc_seg = alpha * r * (1.0 - decay_factor) / self.beta
                integral_exc += integral_exc_seg
            else:
                decay_factor = 1.0
                integral_exc_seg = 0.0
            # decay r
            r *= decay_factor

            lam = mu + alpha * r + beta0 * math.sin(beta1 * t)
            lam = max(lam, eps)
            log_sum += math.log(lam)
            r += 1.0
            sine_int += 0.0  # already accounted via closed form above
            cur_t = t

        # tail integral of excitation
        dt_tail = t_end - cur_t
        integral_exc_tail = alpha * r * (1.0 - math.exp(-self.beta * dt_tail)) / self.beta if dt_tail > 0 else 0.0
        integral_exc += integral_exc_tail

        return log_sum - (integral_mu + sine_int + integral_exc)

    def fit(self, init_mu: float, init_alpha: float) -> FullFitResult:
        # Initialization
        b1_init_env = os.environ.get("B1_INIT")
        b1_init = float(b1_init_env) if (b1_init_env is not None and str(b1_init_env).strip() != "") else 1.0
        alpha_cap = (1.0 - self.stab_margin) * self.beta
        # Keep strict alpha < beta when margin == 0.0
        if self.stab_margin <= 0.0:
            alpha_cap = max(0.0, self.beta - 1e-8)
        theta0 = np.array([init_mu, min(init_alpha, alpha_cap), 0.1, b1_init], dtype=float)
        # Stationarity-aware bounds: alpha <= (1 - margin) * beta
        alpha_upper = max(0.0, alpha_cap)
        bounds = [(1e-8, None), (0.0, alpha_upper), (-5.0, 5.0), (1e-3, 10.0)]
        # If FIXED_B1 is set, lock b1 during optimization
        fixed_b1_env = os.environ.get("FIXED_B1")
        if fixed_b1_env is not None and str(fixed_b1_env).strip() != "":
            b1_fixed = float(fixed_b1_env)
            theta0[3] = b1_fixed
            bounds[3] = (b1_fixed, b1_fixed)
        else:
            # Optional bounded search region for b1
            b1_min_env = os.environ.get("B1_MIN")
            b1_max_env = os.environ.get("B1_MAX")
            if b1_min_env is not None and b1_max_env is not None and str(b1_min_env).strip() != "" and str(b1_max_env).strip() != "":
                b1_min = float(b1_min_env)
                b1_max = float(b1_max_env)
                if b1_min > 0 and b1_max > b1_min:
                    bounds[3] = (b1_min, b1_max)

        res = minimize(lambda th: -self.log_likelihood(th) + self._reg_penalty(th),
                       theta0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-6})

        ll = -float(res.fun)
        aic = 2 * 4 - 2 * ll
        return FullFitResult(params=res.x, loglik=ll, aic=aic, success=bool(res.success))


# -----------------------------
# 主流程
# -----------------------------
def _tune_beta_grid(events: np.ndarray,
                    T: float,
                    t_split: float,
                    beta_grid: np.ndarray,
                    baseline_train: Baseline1D) -> Tuple[float, FullFitResult, float]:
    """Grid search beta by fitting on [0, t_split] and selecting best val LL."""
    train_events = events[events < t_split]
    best_beta = None
    best_fit = None
    best_ll = None
    for beta in beta_grid:
        model_train = FullHawkes1DWithSine(train_events, t_split, beta=float(beta))
        fit_train = model_train.fit(init_mu=baseline_train.mu, init_alpha=baseline_train.alpha)
        evaluator = FullHawkes1DWithSine(events, T, beta=float(beta))
        ll_val = evaluator.log_likelihood_interval(fit_train.params, t_split, T)
        if best_ll is None or ll_val > best_ll:
            best_beta = float(beta)
            best_fit = fit_train
            best_ll = float(ll_val)
    return float(best_beta), best_fit, float(best_ll)


def run_comparison_1d(data_path: str, true_params: dict = None):
    events, T = load_events_1d(data_path)
    print(f"Loaded {len(events)} events, horizon T={T:.2f}")

    # baseline拟合：限制在相对合理的区间（默认 0.1 ~ 2.0），可通过 DECAY_MIN/MAX 覆盖
    dmin = float(os.environ.get("DECAY_MIN", "0.1"))
    dmax = float(os.environ.get("DECAY_MAX", "2.0"))
    decay_grid = np.logspace(math.log10(dmin), math.log10(dmax), 8)
    baseline = fit_baseline_tick(events, decay_grid)
    print("\n=== Baseline (tick) ===")
    print(f"decay={baseline.decay:.4f}, mu={baseline.mu:.4f}, alpha={baseline.alpha:.4f}")
    print(f"loglik={baseline.loglik:.4f}")

    # full模型拟合：beta 策略（baseline / fixed / grid）
    beta_strategy = os.environ.get("BETA_STRATEGY", "baseline")
    fixed_beta = float(os.environ.get("FIXED_BETA", baseline.decay))
    t_split = 0.7 * T
    train_events = events[events < t_split]
    baseline_train = fit_baseline_tick(train_events, decay_grid) if len(train_events) > 1 else baseline

    if beta_strategy == "grid" and len(train_events) > 1:
        bmin = float(os.environ.get("BETA_GRID_MIN", str(dmin)))
        bmax = float(os.environ.get("BETA_GRID_MAX", str(dmax)))
        bpts = int(os.environ.get("BETA_GRID_POINTS", "8"))
        beta_grid = np.logspace(math.log10(bmin), math.log10(bmax), bpts)
        beta_used, full_train_fit, _ = _tune_beta_grid(
            events, T, t_split, beta_grid, baseline_train
        )
    else:
        beta_used = baseline.decay if beta_strategy != "fixed" else fixed_beta
        full_train_fit = None

    full = FullHawkes1DWithSine(events, T, beta=beta_used)
    full_fit = full.fit(init_mu=baseline.mu, init_alpha=baseline.alpha)
    print("\n=== Full model (with sine) ===")
    mu, alpha, b0, b1 = full_fit.params
    print(f"beta_used={beta_used:.6g}")
    print(f"mu={mu:.4f}, alpha={alpha:.4f}, beta0={b0:.4f}, beta1={b1:.4f}")
    print(f"loglik={full_fit.loglik:.4f}, AIC={full_fit.aic:.4f}, success={full_fit.success}")

    # LRT检验
    lr = 2 * (full_fit.loglik - baseline.loglik)
    pval = chi2.sf(lr, 2)
    print(f"\nLikelihood-ratio test: LR={lr:.4f}, p={pval:.4e}")

    # 统一验证口径：70/30 时间切分
    def simple_ll_interval(mu_s: float, alpha_s: float, beta_s: float, t0: float, t1: float) -> float:
        model_tmp = FullHawkes1DWithSine(events, T, beta=beta_s)
        theta_tmp = np.array([mu_s, alpha_s, 0.0, 1.0], dtype=float)
        return model_tmp.log_likelihood_interval(theta_tmp, t0, t1)

    ll_val_simple = simple_ll_interval(baseline.mu, baseline.alpha, baseline.decay, t_split, T)
    ll_val_full = full.log_likelihood_interval(full_fit.params, t_split, T)
    print("\n=== Temporal validation (70/30) ===")
    print(f"Validation log-lik simple: {ll_val_simple:.4f}")
    print(f"Validation log-lik full  : {ll_val_full:.4f}")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    k_simple = 2  # mu, alpha (decay treated as tuned hyperparam)
    baseline_aic = 2 * k_simple - 2 * baseline.loglik
    branching_ratio = float(alpha / beta_used) if beta_used > 0 else float("inf")
    constraint_ok = bool(branching_ratio < 1.0)
    results = {
        "baseline": {
            "decay": float(baseline.decay),
            "mu": float(baseline.mu),
            "alpha": float(baseline.alpha),
            "loglik": float(baseline.loglik),
            "aic": float(baseline_aic),
            "branching_ratio": float(baseline.alpha / baseline.decay) if baseline.decay > 0 else float("inf"),
        },
        "full": {
            "beta": float(beta_used),
            "mu": float(mu), "alpha": float(alpha),
            "beta0": float(b0), "beta1": float(b1),
            "loglik": float(full_fit.loglik), "aic": float(full_fit.aic),
            "branching_ratio": branching_ratio,
            "constraint_ok": constraint_ok,
        },
        "lrt": {"stat": float(lr), "p_value": float(pval)},
        "validation": {
            "t_split": float(t_split),
            "ll_simple": float(ll_val_simple),
            "ll_full": float(ll_val_full)
        }
    }

    # 若提供真值，计算误差
    if true_params:
        diffs = {}
        for k in ["mu", "alpha", "beta0", "beta1"]:
            if k in true_params:
                diffs[k + "_error"] = abs(results["full"][k] - true_params[k])
        results["param_diff"] = diffs
        print("\nParameter errors:", diffs)

    with open("results/comparison_1d.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # Also save a tagged copy to avoid being overwritten in batch runs
    try:
        out_tag = os.environ.get("OUT_TAG")
        if not out_tag:
            # derive from data filename, e.g., events_600519.json -> 600519
            base = os.path.basename(data_path)
            tag = os.path.splitext(base)[0]
            if tag.startswith("events_"):
                out_tag = tag[len("events_"):]
            else:
                out_tag = tag
        with open(f"results/comparison_1d_{out_tag}.json", "w", encoding="utf-8") as f2:
            json.dump(results, f2, ensure_ascii=False, indent=2)
        print(f"\nResults saved to results/comparison_1d.json and results/comparison_1d_{out_tag}.json")
    except Exception:
        print("\nResults saved to results/comparison_1d.json")


# -----------------------------
# 可选：直接仿真 + 拟合
# -----------------------------
if __name__ == "__main__":
    # 1. 如果没有数据，先用 mhp.py 生成一份
    data_file = "events_100k.json"
    if not os.path.exists(data_file):
        from mhp import MHP
        true_mu, true_alpha, true_beta, T = 0.2, 0.5, 1.0, 100.0
        true_b0, true_b1 = 0.3, 1.2
        sim = MHP(lambda_0=true_mu, alpha=true_alpha, beta=true_beta, T=T)
        events = sim.simulate()
        json.dump([{"t": float(t), "i": 0} for t in events],
                  open(data_file, "w", encoding="utf-8"))
        print(f"Simulated {len(events)} events saved to {data_file}")

        true_params = {"mu": true_mu, "alpha": true_alpha,
                       "beta0": true_b0, "beta1": true_b1}
    else:
        true_params = None

    # 2. 拟合并输出结果
    run_comparison_1d(data_file, true_params)
