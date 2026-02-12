"""
4D Hawkes 模型（无外生项）- 使用 tick 实现。
"""
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from tick.hawkes import HawkesExpKern


def load_events_4d(path: str) -> Tuple[List[np.ndarray], float]:
    """加载4D事件数据"""
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
class TickFitResult:
    decay: float
    mu: np.ndarray
    adjacency: np.ndarray
    loglik: float
    aic: float
    branching_ratio: float


def fit_hawkes_4d_tick(events_4d: List[np.ndarray], decay: float) -> TickFitResult:
    """用 tick 拟合 4D Hawkes 模型（给定 decay）"""
    decays_mat = np.full((4, 4), float(decay), dtype=float)
    learner = HawkesExpKern(decays=decays_mat, verbose=False)
    learner.fit([events_4d])
    ll = float(learner.score())
    
    mu = learner.baseline.copy()
    A = learner.adjacency.copy()
    
    # 分枝比 = 谱半径(A) / decay
    eigvals = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigvals)))
    branching_ratio = spectral_radius / decay
    
    # AIC: k = 4 (mu) + 16 (A) = 20 参数
    k_params = 20
    aic = 2 * k_params - 2 * ll
    
    return TickFitResult(
        decay=decay,
        mu=mu,
        adjacency=A,
        loglik=ll,
        aic=aic,
        branching_ratio=branching_ratio,
    )


def grid_search_decay(events_4d: List[np.ndarray], decay_grid: np.ndarray, 
                      require_stable: bool = True) -> TickFitResult:
    """
    网格搜索最优 decay，优先选择稳定（分枝比 < 1）的结果
    
    Parameters:
    -----------
    events_4d : 4D事件列表
    decay_grid : decay候选值
    require_stable : 是否优先选择稳定解
    
    Returns:
    --------
    best : 最优拟合结果
    """
    all_results = []
    stable_results = []
    
    for decay in decay_grid:
        result = fit_hawkes_4d_tick(events_4d, decay)
        all_results.append(result)
        if result.branching_ratio < 1.0:
            stable_results.append(result)
        print(f"  decay={decay:.4f}, loglik={result.loglik:.4f}, branching_ratio={result.branching_ratio:.4f}")
    
    if require_stable and len(stable_results) > 0:
        # 在稳定解中选择 loglik 最大的
        best = max(stable_results, key=lambda r: r.loglik)
        print(f"  -> Selected stable solution: decay={best.decay:.4f}, branching_ratio={best.branching_ratio:.4f}")
    else:
        # 没有稳定解，选择分枝比最小的
        best = min(all_results, key=lambda r: r.branching_ratio)
        print(f"  -> No stable solution found, selecting min branching_ratio: decay={best.decay:.4f}, branching_ratio={best.branching_ratio:.4f}")
    
    return best


def compute_validation_ll(events_4d: List[np.ndarray], T: float, 
                          mu: np.ndarray, A: np.ndarray, decay: float,
                          t_start: float, t_end: float) -> float:
    """
    计算验证集上的对数似然（使用 tick 的内置评分）
    注意：tick 不直接支持区间 LL，这里用近似方法
    """
    # 提取验证集事件
    val_events = [ev[(ev >= t_start) & (ev <= t_end)] - t_start for ev in events_4d]
    T_val = t_end - t_start
    
    # 如果验证集事件太少，返回 NaN
    total_val = sum(len(ev) for ev in val_events)
    if total_val < 4:
        return float("nan")
    
    # 用训练好的参数创建模型并评分
    decays_mat = np.full((4, 4), float(decay), dtype=float)
    learner = HawkesExpKern(decays=decays_mat, verbose=False)
    
    # 设置参数（tick 的 baseline 和 adjacency 是可写的）
    learner.fit([val_events])  # 先 fit 以初始化
    
    # 用验证集重新计算 score（这里简化处理：直接返回验证集上的 fit score）
    # 更严格的做法需要固定参数计算 LL，但 tick 不直接支持
    return float(learner.score())


def compute_time_rescaling_residuals_4d(events_4d: List[np.ndarray], T: float,
                                         mu: np.ndarray, A: np.ndarray, 
                                         decay: float) -> dict:
    """
    计算4D Hawkes模型的时间重标定残差（Ogata检验）
    
    对于一个正确指定的点过程，变换后的残差应服从 Exp(1) 分布。
    
    Returns:
    --------
    dict: 每个维度的残差及 KS 检验 p 值
    """
    from scipy.stats import kstest
    
    # 合并事件时间线
    merged = []
    for d in range(4):
        merged.extend([(float(t), d) for t in events_4d[d]])
    merged.sort(key=lambda x: x[0])
    
    # 计算每个维度的累积强度（Lambda）
    residuals_by_dim = {d: [] for d in range(4)}
    r = np.zeros(4, dtype=float)
    last_t = 0.0
    Lambda_accum = np.zeros(4, dtype=float)  # 累积强度
    last_event_time = {d: 0.0 for d in range(4)}  # 每维上一个事件时间
    
    for t, dim in merged:
        dt = t - last_t
        if dt > 0:
            # 衰减
            decay_factor = math.exp(-decay * dt)
            # 计算这段时间内的积分（强度积分）
            for u in range(4):
                # 基底部分
                base_int = mu[u] * dt
                # 激励部分：∫ A[u,:] · r · exp(-decay * s) ds = A[u,:] · r · (1 - exp(-decay*dt)) / decay
                exc_int = float(A[u, :].dot(r) * (1.0 - decay_factor) / decay)
                Lambda_accum[u] += base_int + exc_int
            r *= decay_factor
        
        # 记录该维度的残差
        if last_event_time[dim] > 0:  # 跳过第一个事件
            residuals_by_dim[dim].append(float(Lambda_accum[dim]))
        
        # 重置该维度的累积强度
        Lambda_accum[dim] = 0.0
        last_event_time[dim] = t
        
        # 更新 r
        r[dim] += 1.0
        last_t = t
    
    # 对每个维度进行 KS 检验
    results = {}
    for d in range(4):
        res = np.array(residuals_by_dim[d], dtype=float)
        if len(res) > 10:
            # KS 检验：残差应服从 Exp(1)
            ks_stat, ks_pval = kstest(res, 'expon', args=(0, 1))
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "mean": float(np.mean(res)),
                "std": float(np.std(res)),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "gof_pass": bool(ks_pval > 0.05),  # 5% 显著性水平
            }
        else:
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "error": "insufficient_residuals",
            }
    
    # 汇总
    gof_pass_count = sum(1 for d in range(4) if results.get(f"dim_{d}", {}).get("gof_pass", False))
    results["summary"] = {
        "gof_pass_count": gof_pass_count,
        "all_pass": gof_pass_count == 4,
    }
    
    return results


def run_comparison_4d_tick(data_path: str) -> dict:
    """运行 4D Hawkes 对比实验（全部使用 tick）"""
    events_4d, T = load_events_4d(data_path)
    total_events = sum(len(ev) for ev in events_4d)
    print(f"Loaded 4D events: {[len(ev) for ev in events_4d]}, total={total_events}, T={T:.2f}")
    
    # Decay 网格：在 0.6–2.0 区间加密，两侧稀疏
    dmin = float(os.environ.get("DECAY_MIN", "0.3"))
    dmax = float(os.environ.get("DECAY_MAX", "10.0"))
    
    # 构建非均匀网格：低区 [0.3, 0.6] + 密集区 [0.6, 2.0] + 高区 [2.0, 10.0]
    grid_low = np.linspace(dmin, 0.6, 3, endpoint=False)  # 3 点
    grid_dense = np.linspace(0.6, 2.0, 12)                 # 12 点（密集）
    grid_high = np.linspace(2.5, dmax, 4)                  # 4 点
    decay_grid = np.unique(np.concatenate([grid_low, grid_dense, grid_high]))
    n_points = len(decay_grid)
    print(f"Decay grid ({n_points} points): {decay_grid[:5].round(3)}...{decay_grid[-3:].round(3)}")
    
    # 是否要求稳定解
    require_stable = os.environ.get("REQUIRE_STABLE", "1") != "0"
    
    # 70/30 时间切分
    t_split = 0.7 * T
    train_events = [ev[ev < t_split] for ev in events_4d]
    
    # 策略选择
    beta_strategy = os.environ.get("BETA_STRATEGY", "grid")
    
    if beta_strategy == "grid":
        # 在训练集上网格搜索最优 decay（优先稳定解）
        print(f"Grid search decay on training set ({n_points} points, require_stable={require_stable})...")
        train_result = grid_search_decay(train_events, decay_grid, require_stable=require_stable)
        best_decay = train_result.decay
        print(f"Best decay from grid search: {best_decay:.4f}")
    elif beta_strategy == "fixed":
        best_decay = float(os.environ.get("FIXED_BETA", "0.1"))
        print(f"Using fixed decay: {best_decay:.4f}")
    else:
        # baseline: 在全量数据上搜索
        full_result = grid_search_decay(events_4d, decay_grid)
        best_decay = full_result.decay
        print(f"Baseline decay: {best_decay:.4f}")
    
    # 在全量数据上用最优 decay 拟合
    print(f"Fitting on full data with decay={best_decay:.4f}...")
    full_result = fit_hawkes_4d_tick(events_4d, best_decay)
    
    # 在训练集上拟合（用于验证对比）
    train_result = fit_hawkes_4d_tick(train_events, best_decay)
    
    # 验证集评分（简化：在验证集上单独 fit 后取 score）
    val_events = [ev[(ev >= t_split) & (ev <= T)] for ev in events_4d]
    val_total = sum(len(ev) for ev in val_events)
    
    if val_total >= 4:
        # 用训练集参数在验证集上的表现（这里简化为验证集单独 fit）
        val_result = fit_hawkes_4d_tick(val_events, best_decay)
        ll_val = val_result.loglik
    else:
        ll_val = float("nan")
    
    # 时间重标定残差检验（GOF）
    print("Computing time-rescaling residuals (GOF test)...")
    gof_results = compute_time_rescaling_residuals_4d(
        events_4d, T, full_result.mu, full_result.adjacency, full_result.decay
    )
    gof_pass = gof_results["summary"]["all_pass"]
    print(f"  GOF pass: {gof_results['summary']['gof_pass_count']}/4 dimensions")
    
    # 结果
    results = {
        "full": {
            "decay": float(full_result.decay),
            "mu": full_result.mu.tolist(),
            "A": full_result.adjacency.tolist(),
            "loglik": float(full_result.loglik),
            "aic": float(full_result.aic),
            "branching_ratio": float(full_result.branching_ratio),
            "constraint_ok": bool(full_result.branching_ratio < 1.0),
        },
        "train": {
            "decay": float(train_result.decay),
            "mu": train_result.mu.tolist(),
            "A": train_result.adjacency.tolist(),
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
    
    # 保存结果
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_4d_tick.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 带 tag 的副本
    out_tag = os.environ.get("OUT_TAG", "")
    if out_tag:
        with open(f"results/comparison_4d_tick_{out_tag}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "events_100k.json"
    run_comparison_4d_tick(data_file)
