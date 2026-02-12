"""
4D Hawkes 模型 (带re_spread外生项) - 使用 tick 实现。
模型：λ_i(t) = μ_i(t) + Σ_j Σ_{t_k^j < t} A_{ij} e^{-β(t-t_k^j)} + γ_spread_i * re_spread(t)

GOF检验使用分段基准强度：μ_i(t) = μ_i * exp(γ_{i,o} * I_OPEN30(t) + γ_{i,m} * I_MID30(t) + γ_{i,c} * I_CLOSE30(t))
外生项：γ_spread_i * re_spread(t) 捕捉价差对事件强度的影响
"""
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.optimize import minimize
from tick.hawkes import HawkesExpKern


# A股交易时间常量（秒，从午夜开始）
MARKET_OPEN_AM = 34200    # 9:30
MARKET_CLOSE_AM = 41400   # 11:30
MARKET_OPEN_PM = 46800    # 13:00
MARKET_CLOSE_PM = 54000   # 15:00

# OPEN30: 9:30-10:00 (早盘开盘30分钟)
OPEN30_START = 34200      # 9:30
OPEN30_END = 36000        # 10:00

# MID30: 13:00-13:30 (午间开盘30分钟)
MID30_START = 46800       # 13:00
MID30_END = 48600         # 13:30

# CLOSE30: 14:30-15:00 (收盘前30分钟)
CLOSE30_START = 52200     # 14:30
CLOSE30_END = 54000       # 15:00


def get_intraday_time(t_original: float) -> float:
    """
    从原始时间戳获取日内时间（秒，从午夜开始）
    假设原始时间戳是秒级时间戳
    """
    if 0 <= t_original < 86400:
        return t_original
    return t_original % 86400


def is_open30(t_intraday: float) -> bool:
    """判断是否在OPEN30时段（9:30-10:00）"""
    return OPEN30_START <= t_intraday < OPEN30_END


def is_mid30(t_intraday: float) -> bool:
    """判断是否在MID30时段（13:00-13:30，午间开盘）"""
    return MID30_START <= t_intraday < MID30_END


def is_close30(t_intraday: float) -> bool:
    """判断是否在CLOSE30时段（14:30-15:00）"""
    return CLOSE30_START <= t_intraday < CLOSE30_END


def compute_indicators(t_intraday: float) -> Tuple[float, float, float]:
    """
    计算I_OPEN30、I_MID30和I_CLOSE30指示变量
    返回 (I_open, I_mid, I_close)
    """
    I_open = 1.0 if is_open30(t_intraday) else 0.0
    I_mid = 1.0 if is_mid30(t_intraday) else 0.0
    I_close = 1.0 if is_close30(t_intraday) else 0.0
    return I_open, I_mid, I_close


# 时段边界点（日内时间，秒）
PERIOD_BOUNDARIES = [
    OPEN30_END,      # 36000 (10:00) - OPEN30结束
    MARKET_CLOSE_AM, # 41400 (11:30) - 上午收盘
    MARKET_OPEN_PM,  # 46800 (13:00) - MID30开始
    MID30_END,       # 48600 (13:30) - MID30结束
    CLOSE30_START,   # 52200 (14:30) - CLOSE30开始
]


def get_period_boundaries_between(t1_intraday: float, t2_intraday: float) -> List[float]:
    """获取两个日内时间点之间的所有时段边界"""
    if t1_intraday >= t2_intraday:
        return []
    
    boundaries = []
    for b in PERIOD_BOUNDARIES:
        if t1_intraday < b < t2_intraday:
            boundaries.append(b)
    return sorted(boundaries)


def intraday_to_trading_time(t_intraday: float) -> float:
    """
    将日内时间转换为交易时间（修复版本）
    
    A股交易时间：
    - 上午：9:30-11:30 (34200-41400) -> 0-7200
    - 下午：13:00-15:00 (46800-54000) -> 7200-14400
    - 午休：11:30-13:00 不计入交易时间
    
    Returns:
    --------
    交易时间（秒），范围 [0, 14400]
    """
    if t_intraday < MARKET_OPEN_AM:
        return 0.0
    elif t_intraday <= MARKET_CLOSE_AM:
        # 上午交易时段
        return t_intraday - MARKET_OPEN_AM
    elif t_intraday < MARKET_OPEN_PM:
        # 午休时段，返回上午结束时间
        return MARKET_CLOSE_AM - MARKET_OPEN_AM  # 7200
    elif t_intraday <= MARKET_CLOSE_PM:
        # 下午交易时段
        return (MARKET_CLOSE_AM - MARKET_OPEN_AM) + (t_intraday - MARKET_OPEN_PM)
    else:
        return 14400.0


def is_in_trading_hours(t_intraday: float) -> bool:
    """判断日内时间是否在交易时段内"""
    return (MARKET_OPEN_AM <= t_intraday <= MARKET_CLOSE_AM or 
            MARKET_OPEN_PM <= t_intraday <= MARKET_CLOSE_PM)


def compute_segmented_baseline_integral(
    t1_intraday: float, t2_intraday: float, dt_trading: float,
    mu_base: float, gamma_open: float, gamma_mid: float, gamma_close: float
) -> float:
    """
    计算分段基准强度积分，正确处理跨时段边界的情况
    
    修复版本：使用端点映射直接计算每段的交易时间，
    而非使用比例分配（比例分配会错误地将午休时间计入）
    
    Parameters:
    -----------
    t1_intraday : float
        起始日内时间（秒，从午夜开始）
    t2_intraday : float
        结束日内时间（秒，从午夜开始）
    dt_trading : float
        交易时间间隔（已排除午休，用于验证）
    mu_base : float
        基准强度
    gamma_open, gamma_mid, gamma_close : float
        各时段的gamma系数
    
    Returns:
    --------
    float : 基准强度积分值
    """
    if dt_trading <= 0:
        return 0.0
    
    # 跨日情况：使用终点的indicator计算整个积分
    if t2_intraday <= t1_intraday:
        I_open, I_mid, I_close = compute_indicators(t2_intraday)
        mu_t = compute_time_varying_mu(mu_base, gamma_open, gamma_mid, gamma_close, 
                                        I_open, I_mid, I_close)
        return mu_t * dt_trading
    
    # 获取区间内的边界点
    boundaries = get_period_boundaries_between(t1_intraday, t2_intraday)
    
    if len(boundaries) == 0:
        # 没有跨越边界，整段使用相同的indicator
        t_mid = (t1_intraday + t2_intraday) / 2
        I_open, I_mid, I_close = compute_indicators(t_mid)
        mu_t = compute_time_varying_mu(mu_base, gamma_open, gamma_mid, gamma_close,
                                        I_open, I_mid, I_close)
        return mu_t * dt_trading
    
    # 有边界点，需要分段积分
    # 使用端点映射直接计算每段的交易时间
    segment_points = [t1_intraday] + boundaries + [t2_intraday]
    
    total_integral = 0.0
    for i in range(len(segment_points) - 1):
        seg_start = segment_points[i]
        seg_end = segment_points[i + 1]
        
        # 使用端点映射计算该段的实际交易时间
        # 这样午休时段（11:30-13:00）的交易时间将为0
        seg_trading_start = intraday_to_trading_time(seg_start)
        seg_trading_end = intraday_to_trading_time(seg_end)
        seg_trading_dt = seg_trading_end - seg_trading_start
        
        # 跳过非交易时段（如午休）
        if seg_trading_dt <= 0:
            continue
        
        # 使用段中点的indicator
        seg_mid = (seg_start + seg_end) / 2
        I_open, I_mid, I_close = compute_indicators(seg_mid)
        mu_t = compute_time_varying_mu(mu_base, gamma_open, gamma_mid, gamma_close,
                                        I_open, I_mid, I_close)
        total_integral += mu_t * seg_trading_dt
    
    return total_integral


def estimate_gamma_mle(events_4d: List[np.ndarray], 
                       events_4d_original: List[np.ndarray],
                       mu: np.ndarray, A: np.ndarray, decay: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    估计时变基准强度的gamma参数
    """
    gamma_open = np.zeros(4)
    gamma_mid = np.zeros(4)
    gamma_close = np.zeros(4)
    
    for d in range(4):
        if len(events_4d_original[d]) == 0:
            continue
        
        n_open = 0
        n_mid = 0
        n_close = 0
        n_other = 0
        
        for t_orig in events_4d_original[d]:
            t_intraday = get_intraday_time(t_orig)
            if is_open30(t_intraday):
                n_open += 1
            elif is_mid30(t_intraday):
                n_mid += 1
            elif is_close30(t_intraday):
                n_close += 1
            else:
                n_other += 1
        
        n_total = n_open + n_mid + n_close + n_other
        if n_total == 0:
            continue
        
        T_open = 30 * 60
        T_mid = 30 * 60
        T_close = 30 * 60
        T_other = 150 * 60
        
        rate_open = n_open / T_open if n_open > 0 else 0
        rate_mid = n_mid / T_mid if n_mid > 0 else 0
        rate_close = n_close / T_close if n_close > 0 else 0
        rate_other = n_other / T_other if n_other > 0 else 1e-10
        
        if rate_open > 0 and rate_other > 0:
            gamma_open[d] = np.log(rate_open / rate_other)
        if rate_mid > 0 and rate_other > 0:
            gamma_mid[d] = np.log(rate_mid / rate_other)
        if rate_close > 0 and rate_other > 0:
            gamma_close[d] = np.log(rate_close / rate_other)
        
        gamma_open[d] = np.clip(gamma_open[d], -3.0, 3.0)
        gamma_mid[d] = np.clip(gamma_mid[d], -3.0, 3.0)
        gamma_close[d] = np.clip(gamma_close[d], -3.0, 3.0)
    
    return gamma_open, gamma_mid, gamma_close


def estimate_gamma_spread(events_4d: List[np.ndarray],
                          re_spread_4d: List[np.ndarray],
                          mu: np.ndarray, T: float) -> np.ndarray:
    """
    估计re_spread外生项的gamma_spread参数
    
    使用简化的最小二乘估计：
    假设 λ_i ≈ μ_i + γ_spread_i * mean(re_spread_i)
    通过比较高/低spread时期的事件率来估计
    
    Parameters:
    -----------
    events_4d : 归一化后的事件时间
    re_spread_4d : 每个事件对应的re_spread值
    mu : tick估计的基准强度
    T : 总观测时间
    
    Returns:
    --------
    gamma_spread : shape (4,) 各维度的spread系数
    """
    gamma_spread = np.zeros(4)
    
    for d in range(4):
        if len(events_4d[d]) == 0 or len(re_spread_4d[d]) == 0:
            continue
        
        spreads = np.array(re_spread_4d[d])
        n_events = len(spreads)
        
        if n_events < 10:
            continue
        
        # 计算spread的均值和标准差
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        if std_spread < 1e-10:
            # spread没有变化，无法估计
            continue
        
        # 分成高/低spread两组
        median_spread = np.median(spreads)
        high_spread_mask = spreads >= median_spread
        low_spread_mask = spreads < median_spread
        
        n_high = np.sum(high_spread_mask)
        n_low = np.sum(low_spread_mask)
        
        if n_high < 5 or n_low < 5:
            continue
        
        # 计算两组的平均spread
        mean_spread_high = np.mean(spreads[high_spread_mask])
        mean_spread_low = np.mean(spreads[low_spread_mask])
        
        # 假设时间对半分（简化）
        # 事件率差异反映spread效应
        rate_high = n_high / (T / 2)
        rate_low = n_low / (T / 2)
        
        # γ_spread ≈ (rate_high - rate_low) / (mean_spread_high - mean_spread_low)
        spread_diff = mean_spread_high - mean_spread_low
        if abs(spread_diff) > 1e-10:
            gamma_spread[d] = (rate_high - rate_low) / spread_diff
        
        # 限制范围
        gamma_spread[d] = np.clip(gamma_spread[d], -1000.0, 1000.0)
    
    return gamma_spread


def compute_time_varying_mu(mu_base: float, gamma_open: float, gamma_mid: float, gamma_close: float,
                            I_open: float, I_mid: float, I_close: float) -> float:
    """
    计算时变基准强度
    μ(t) = μ * exp(γ_o * I_OPEN30 + γ_m * I_MID30 + γ_c * I_CLOSE30)
    """
    return mu_base * math.exp(gamma_open * I_open + gamma_mid * I_mid + gamma_close * I_close)


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
    """网格搜索最优 decay，优先选择稳定（分枝比 < 1）的结果"""
    all_results = []
    stable_results = []
    
    for decay in decay_grid:
        result = fit_hawkes_4d_tick(events_4d, decay)
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


def compute_time_rescaling_residuals_4d_exog(
    events_4d: List[np.ndarray], T: float,
    mu: np.ndarray, A: np.ndarray, decay: float,
    events_4d_original: Optional[List[np.ndarray]] = None,
    re_spread_4d: Optional[List[np.ndarray]] = None,
    gamma_open: Optional[np.ndarray] = None,
    gamma_mid: Optional[np.ndarray] = None,
    gamma_close: Optional[np.ndarray] = None,
    gamma_spread: Optional[np.ndarray] = None
) -> dict:
    """
    计算4D Hawkes模型的时间重标定残差（Ogata检验）- 带外生项版本
    
    模型：λ_i(t) = μ_i(t) + Σ_j Σ_{t_k^j < t} A_{ij} e^{-β(t-t_k^j)} + γ_spread_i * re_spread(t)
    
    Parameters:
    -----------
    events_4d : 归一化后的事件时间
    T : 总时间长度
    mu : 基准强度
    A : 激励矩阵
    decay : 衰减率
    events_4d_original : 原始事件时间（用于判断时段）
    re_spread_4d : 每个事件的re_spread值
    gamma_open, gamma_mid, gamma_close : 时段gamma系数
    gamma_spread : re_spread的系数
    
    Returns:
    --------
    dict: 每个维度的残差及 KS 检验 p 值
    """
    from scipy.stats import kstest
    
    use_time_varying = (events_4d_original is not None and 
                        gamma_open is not None and 
                        gamma_mid is not None and
                        gamma_close is not None)
    
    use_exog = (re_spread_4d is not None and gamma_spread is not None)
    
    if not use_time_varying:
        gamma_open = np.zeros(4)
        gamma_mid = np.zeros(4)
        gamma_close = np.zeros(4)
    
    if not use_exog:
        gamma_spread = np.zeros(4)
    
    # 合并事件时间线
    merged = []
    for d in range(4):
        for idx, t in enumerate(events_4d[d]):
            t_orig = events_4d_original[d][idx] if use_time_varying and idx < len(events_4d_original[d]) else t
            spread = re_spread_4d[d][idx] if use_exog and idx < len(re_spread_4d[d]) else 0.0
            merged.append((float(t), d, float(t_orig), float(spread)))
    merged.sort(key=lambda x: x[0])
    
    # 计算每个维度的累积强度
    residuals_by_dim = {d: [] for d in range(4)}
    r = np.zeros(4, dtype=float)
    last_t = 0.0
    last_t_orig = MARKET_OPEN_AM
    last_spread = {d: 0.0 for d in range(4)}  # 每维上一个事件的spread
    Lambda_accum = np.zeros(4, dtype=float)
    last_event_time = {d: 0.0 for d in range(4)}
    
    for t, dim, t_orig, spread in merged:
        dt = t - last_t
        if dt > 0:
            decay_factor = math.exp(-decay * dt)
            
            t1_intraday = get_intraday_time(last_t_orig)
            t2_intraday = get_intraday_time(t_orig)
            
            for u in range(4):
                # 基准强度积分
                base_int = compute_segmented_baseline_integral(
                    t1_intraday, t2_intraday, dt,
                    mu[u], gamma_open[u], gamma_mid[u], gamma_close[u]
                )
                # 激励部分
                exc_int = float(A[u, :].dot(r) * (1.0 - decay_factor) / decay)
                
                # 外生项：γ_spread * re_spread * dt
                # 使用上一个事件的spread值作为当前区间的spread（分段常数近似）
                exog_int = gamma_spread[u] * last_spread.get(u, 0.0) * dt
                
                Lambda_accum[u] += base_int + exc_int + exog_int
            r *= decay_factor
        
        # 记录残差
        if last_event_time[dim] > 0:
            residuals_by_dim[dim].append(float(Lambda_accum[dim]))
        
        Lambda_accum[dim] = 0.0
        last_event_time[dim] = t
        last_spread[dim] = spread  # 更新该维度的spread
        
        r[dim] += 1.0
        last_t = t
        last_t_orig = t_orig
    
    # KS 检验
    results = {}
    for d in range(4):
        res = np.array(residuals_by_dim[d], dtype=float)
        if len(res) > 10:
            ks_stat, ks_pval = kstest(res, 'expon', args=(0, 1))
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "mean": float(np.mean(res)),
                "std": float(np.std(res)),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "gof_pass": bool(ks_pval > 0.05),
            }
        else:
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "error": "insufficient_residuals",
            }
    
    gof_pass_count = sum(1 for d in range(4) if results.get(f"dim_{d}", {}).get("gof_pass", False))
    results["summary"] = {
        "gof_pass_count": gof_pass_count,
        "all_pass": gof_pass_count == 4,
    }
    
    if use_time_varying:
        results["gamma"] = {
            "gamma_open": gamma_open.tolist(),
            "gamma_mid": gamma_mid.tolist(),
            "gamma_close": gamma_close.tolist(),
        }
    
    if use_exog:
        results["gamma_spread"] = gamma_spread.tolist()
    
    return results


def run_comparison_4d_tick_exog(
    data_path: str, 
    events_4d_original: Optional[List[np.ndarray]] = None,
    re_spread_4d: Optional[List[np.ndarray]] = None
) -> dict:
    """
    运行 4D Hawkes 对比实验（带re_spread外生项）
    
    Parameters:
    -----------
    data_path : 数据文件路径
    events_4d_original : 原始事件时间（用于GOF的时段判断）
    re_spread_4d : 每个事件的re_spread值
    """
    events_4d, T = load_events_4d(data_path)
    total_events = sum(len(ev) for ev in events_4d)
    print(f"Loaded 4D events: {[len(ev) for ev in events_4d]}, total={total_events}, T={T:.2f}")
    
    if re_spread_4d is not None:
        spread_counts = [len(s) for s in re_spread_4d]
        print(f"Loaded re_spread: {spread_counts}")
    
    # Decay 网格
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
    train_events = [ev[ev < t_split] for ev in events_4d]
    
    beta_strategy = os.environ.get("BETA_STRATEGY", "grid")
    
    if beta_strategy == "grid":
        print(f"Grid search decay on training set ({n_points} points, require_stable={require_stable})...")
        train_result = grid_search_decay(train_events, decay_grid, require_stable=require_stable)
        best_decay = train_result.decay
        print(f"Best decay from grid search: {best_decay:.4f}")
    elif beta_strategy == "fixed":
        best_decay = float(os.environ.get("FIXED_BETA", "0.1"))
        print(f"Using fixed decay: {best_decay:.4f}")
    else:
        full_result = grid_search_decay(events_4d, decay_grid)
        best_decay = full_result.decay
        print(f"Baseline decay: {best_decay:.4f}")
    
    # 全量拟合
    print(f"Fitting on full data with decay={best_decay:.4f}...")
    full_result = fit_hawkes_4d_tick(events_4d, best_decay)
    train_result = fit_hawkes_4d_tick(train_events, best_decay)
    
    # 验证集
    val_events = [ev[(ev >= t_split) & (ev <= T)] for ev in events_4d]
    val_total = sum(len(ev) for ev in val_events)
    
    if val_total >= 4:
        val_result = fit_hawkes_4d_tick(val_events, best_decay)
        ll_val = val_result.loglik
    else:
        ll_val = float("nan")
    
    # 估计 gamma 参数
    gamma_open = None
    gamma_mid = None
    gamma_close = None
    gamma_spread = None
    
    if events_4d_original is not None:
        print("Estimating gamma parameters for time-varying baseline (OPEN30/MID30/CLOSE30)...")
        gamma_open, gamma_mid, gamma_close = estimate_gamma_mle(
            events_4d, events_4d_original, 
            full_result.mu, full_result.adjacency, full_result.decay
        )
        print(f"  gamma_open:  {gamma_open.round(3)} (9:30-10:00)")
        print(f"  gamma_mid:   {gamma_mid.round(3)} (13:00-13:30)")
        print(f"  gamma_close: {gamma_close.round(3)} (14:30-15:00)")
    
    if re_spread_4d is not None:
        print("Estimating gamma_spread for exogenous re_spread effect...")
        gamma_spread = estimate_gamma_spread(events_4d, re_spread_4d, full_result.mu, T)
        print(f"  gamma_spread: {gamma_spread.round(3)}")
    
    print("Computing time-rescaling residuals (GOF test) with exogenous re_spread...")
    gof_results = compute_time_rescaling_residuals_4d_exog(
        events_4d, T, full_result.mu, full_result.adjacency, full_result.decay,
        events_4d_original=events_4d_original,
        re_spread_4d=re_spread_4d,
        gamma_open=gamma_open,
        gamma_mid=gamma_mid,
        gamma_close=gamma_close,
        gamma_spread=gamma_spread
    )
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
        "exog": {
            "gamma_spread": gamma_spread.tolist() if gamma_spread is not None else None,
            "has_re_spread": re_spread_4d is not None,
        },
        "config": {
            "beta_strategy": beta_strategy,
            "decay_grid": decay_grid.tolist(),
            "total_events": int(total_events),
        },
    }
    
    # 保存结果
    os.makedirs("results_exog", exist_ok=True)
    with open("results_exog/comparison_4d_tick_exog.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    out_tag = os.environ.get("OUT_TAG", "")
    if out_tag:
        with open(f"results_exog/comparison_4d_tick_exog_{out_tag}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "events_100k.json"
    run_comparison_4d_tick_exog(data_file)
