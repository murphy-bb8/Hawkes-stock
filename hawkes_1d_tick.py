"""
1D Hawkes 模型（无外生项）- 使用 tick 实现。
"""
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from tick.hawkes import HawkesExpKern


def load_events_1d(path: str) -> Tuple[np.ndarray, float]:
    """加载1D事件数据"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    events = []
    for e in raw:
        t = float(e["t"])
        events.append(t)
    events.sort()
    T = events[-1] if len(events) > 0 else 0.0
    return np.asarray(events, dtype=float), T


@dataclass
class TickFitResult:
    decay: float
    mu: float
    alpha: float
    loglik: float
    aic: float
    branching_ratio: float


def fit_hawkes_1d_tick(events: np.ndarray, decay: float) -> TickFitResult:
    """用 tick 拟合 1D Hawkes 模型（给定 decay）"""
    learner = HawkesExpKern(decays=float(decay), verbose=False)
    learner.fit([[events]])
    ll = float(learner.score())
    
    mu = float(learner.baseline[0])
    alpha = float(learner.adjacency[0, 0])
    
    # 分枝比 = alpha / decay
    branching_ratio = alpha / decay
    
    # AIC: k = 2 (mu, alpha)
    k_params = 2
    aic = 2 * k_params - 2 * ll
    
    return TickFitResult(
        decay=decay,
        mu=mu,
        alpha=alpha,
        loglik=ll,
        aic=aic,
        branching_ratio=branching_ratio,
    )


def grid_search_decay_1d(events: np.ndarray, decay_grid: np.ndarray,
                         require_stable: bool = True) -> TickFitResult:
    """
    网格搜索最优 decay，优先选择稳定（分枝比 < 1）的结果
    """
    all_results = []
    stable_results = []
    
    for decay in decay_grid:
        result = fit_hawkes_1d_tick(events, decay)
        all_results.append(result)
        if result.branching_ratio < 1.0:
            stable_results.append(result)
        print(f"  decay={decay:.4f}, loglik={result.loglik:.4f}, branching_ratio={result.branching_ratio:.4f}")
    
    if require_stable and len(stable_results) > 0:
        best = max(stable_results, key=lambda r: r.loglik)
        print(f"  -> Selected stable solution: decay={best.decay:.4f}, branching_ratio={best.branching_ratio:.4f}")
    else:
        best = min(all_results, key=lambda r: r.branching_ratio)
        print(f"  -> No stable solution found, selecting min branching_ratio: decay={best.decay:.4f}, branching_ratio={best.branching_ratio:.4f}")
    
    return best


def compute_time_rescaling_residuals_1d(events: np.ndarray, T: float,
                                         mu: float, alpha: float, 
                                         decay: float) -> dict:
    """
    计算1D Hawkes模型的时间重标定残差（Ogata检验）
    
    强度为 λ(t) = μ + alpha * Σ_k exp(-decay*(t-t_k))
    """
    from scipy.stats import kstest
    
    residuals = []
    r = 0.0  # r = Σ_k exp(-decay*(t - t_k))
    last_t = 0.0
    Lambda_accum = 0.0
    last_event_time = 0.0
    
    for t in events:
        dt = t - last_t
        if dt > 0:
            decay_factor = math.exp(-decay * dt)
            # 基底积分
            base_int = mu * dt
            # 激励积分：alpha * r * (1 - exp(-decay*dt)) / decay
            exc_int = alpha * r * (1.0 - decay_factor) / decay
            Lambda_accum += base_int + exc_int
            r *= decay_factor
        
        if last_event_time > 0:  # 跳过第一个事件
            residuals.append(float(Lambda_accum))
        last_event_time = t
        Lambda_accum = 0.0
        r += 1.0
        last_t = t
    
    res = np.array(residuals, dtype=float)
    if len(res) > 10:
        ks_stat, ks_pval = kstest(res, 'expon', args=(0, 1))
        return {
            "n_residuals": len(res),
            "mean": float(np.mean(res)),
            "std": float(np.std(res)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "gof_pass": bool(ks_pval > 0.05),
        }
    else:
        return {
            "n_residuals": len(res),
            "error": "insufficient_residuals",
        }


def compute_validation_ll_1d(events: np.ndarray, decay: float,
                              t_start: float, t_end: float) -> float:
    """
    计算验证集上的对数似然（使用 tick 的内置评分）
    """
    # 提取验证集事件
    val_events = events[(events >= t_start) & (events <= t_end)] - t_start
    
    # 如果验证集事件太少，返回 NaN
    if len(val_events) < 4:
        return float("nan")
    
    # 用 tick 重新 fit 并返回 score
    learner = HawkesExpKern(decays=float(decay), verbose=False)
    learner.fit([[val_events]])
    return float(learner.score())


def run_comparison_1d_tick(data_path: str) -> dict:
    """运行 1D Hawkes 对比实验（全部使用 tick）"""
    events, T = load_events_1d(data_path)
    total_events = len(events)
    print(f"Loaded 1D events: {total_events}, T={T:.2f}")
    
    # Decay 网格：在 0.6–2.0 区间加密
    dmin = float(os.environ.get("DECAY_MIN", "0.3"))
    dmax = float(os.environ.get("DECAY_MAX", "10.0"))
    
    grid_low = np.linspace(dmin, 0.6, 3, endpoint=False)
    grid_dense = np.linspace(0.6, 2.0, 12)
    grid_high = np.linspace(2.5, dmax, 4)
    decay_grid = np.unique(np.concatenate([grid_low, grid_dense, grid_high]))
    n_points = len(decay_grid)
    print(f"Decay grid ({n_points} points): {decay_grid[:5].round(3)}...{decay_grid[-3:].round(3)}")
    
    require_stable = os.environ.get("REQUIRE_STABLE", "1") != "0"
    
    # 70/30 时间切分
    t_split = 0.7 * T
    train_events = events[events < t_split]
    
    beta_strategy = os.environ.get("BETA_STRATEGY", "grid")
    
    if beta_strategy == "grid":
        print(f"Grid search decay on training set ({n_points} points, require_stable={require_stable})...")
        train_result = grid_search_decay_1d(train_events, decay_grid, require_stable=require_stable)
        best_decay = train_result.decay
        print(f"Best decay from grid search: {best_decay:.4f}")
    elif beta_strategy == "fixed":
        best_decay = float(os.environ.get("FIXED_BETA", "0.1"))
        print(f"Using fixed decay: {best_decay:.4f}")
    else:
        full_result = grid_search_decay_1d(events, decay_grid, require_stable=require_stable)
        best_decay = full_result.decay
        print(f"Baseline decay: {best_decay:.4f}")
    
    # 全量数据拟合
    print(f"Fitting on full data with decay={best_decay:.4f}...")
    full_result = fit_hawkes_1d_tick(events, best_decay)
    
    # 训练集拟合
    train_result = fit_hawkes_1d_tick(train_events, best_decay)
    
    # 验证集评分（简化：在验证集上单独 fit 后取 score）
    val_events = events[(events >= t_split) & (events <= T)]
    val_total = len(val_events)
    
    if val_total >= 4:
        val_result = fit_hawkes_1d_tick(val_events - t_split, best_decay)
        ll_val = val_result.loglik
    else:
        ll_val = float("nan")
    
    # GOF 检验
    print("Computing time-rescaling residuals (GOF test)...")
    gof_results = compute_time_rescaling_residuals_1d(
        events, T, full_result.mu, full_result.alpha, full_result.decay
    )
    gof_pass = gof_results.get("gof_pass", False)
    print(f"  GOF pass: {gof_pass}")
    
    results = {
        "full": {
            "decay": float(full_result.decay),
            "mu": float(full_result.mu),
            "alpha": float(full_result.alpha),
            "loglik": float(full_result.loglik),
            "aic": float(full_result.aic),
            "branching_ratio": float(full_result.branching_ratio),
            "constraint_ok": bool(full_result.branching_ratio < 1.0),
        },
        "train": {
            "decay": float(train_result.decay),
            "mu": float(train_result.mu),
            "alpha": float(train_result.alpha),
            "loglik": float(train_result.loglik),
        },
        "validation": {
            "t_split": float(t_split),
            "ll_val": float(ll_val),
            "n_val_events": int(val_total),
        },
        "gof": gof_results,
        "config": {
            "beta_strategy": beta_strategy,
            "decay_grid": decay_grid.tolist(),
            "total_events": int(total_events),
        },
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_1d_tick.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    out_tag = os.environ.get("OUT_TAG", "")
    if out_tag:
        with open(f"results/comparison_1d_tick_{out_tag}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "events_sim_1d.json"
    run_comparison_1d_tick(data_file)
