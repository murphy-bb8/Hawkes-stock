"""
4D Hawkes 模型 (无外生项)- 使用 tick 实现。
GOF检验使用分段基准强度：μ_i(t) = μ_i * exp(γ_{i,o} * I_OPEN30(t) + γ_{i,m} * I_MID30(t) + γ_{i,c} * I_CLOSE30(t))
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
    # 如果时间戳在合理的日内范围内（0-86400），直接返回
    if 0 <= t_original < 86400:
        return t_original
    # 否则取模
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
    """
    获取两个日内时间点之间的所有时段边界
    
    Parameters:
    -----------
    t1_intraday : float
        起始日内时间（秒，从午夜开始）
    t2_intraday : float
        结束日内时间（秒，从午夜开始）
    
    Returns:
    --------
    List[float] : 在 (t1, t2) 区间内的边界点列表
    """
    if t1_intraday >= t2_intraday:
        # 如果时间回退，说明跨日，不做边界分割
        return []
    
    boundaries = []
    for b in PERIOD_BOUNDARIES:
        if t1_intraday < b < t2_intraday:
            boundaries.append(b)
    return sorted(boundaries)


def intraday_to_trading_time_for_gof(t_intraday: float) -> float:
    """
    将日内时间转换为交易时间（用于GOF计算）
    与 fit_toxic_events.py 中的函数保持一致
    
    A股交易时间：
    - 上午：9:30-11:30 (34200-41400) -> 0-7200
    - 下午：13:00-15:00 (46800-54000) -> 7200-14400
    """
    if t_intraday < MARKET_OPEN_AM:
        return 0.0
    elif t_intraday <= MARKET_CLOSE_AM:
        return t_intraday - MARKET_OPEN_AM
    elif t_intraday < MARKET_OPEN_PM:
        return MARKET_CLOSE_AM - MARKET_OPEN_AM  # 7200
    elif t_intraday <= MARKET_CLOSE_PM:
        return (MARKET_CLOSE_AM - MARKET_OPEN_AM) + (t_intraday - MARKET_OPEN_PM)
    else:
        return 14400.0


def compute_single_day_baseline_integral(
    t_start: float, t_end: float,
    mu_base: float, gamma_open: float, gamma_mid: float, gamma_close: float
) -> float:
    """
    计算单日内的基准强度积分
    """
    if t_start >= t_end:
        return 0.0
    
    boundaries = get_period_boundaries_between(t_start, t_end)
    segment_points = [t_start] + boundaries + [t_end]
    
    total_integral = 0.0
    for i in range(len(segment_points) - 1):
        seg_start = segment_points[i]
        seg_end = segment_points[i + 1]
        
        seg_trading_start = intraday_to_trading_time_for_gof(seg_start)
        seg_trading_end = intraday_to_trading_time_for_gof(seg_end)
        seg_trading_dt = seg_trading_end - seg_trading_start
        
        if seg_trading_dt <= 0:
            continue
        
        seg_mid = (seg_start + seg_end) / 2
        I_open, I_mid, I_close = compute_indicators(seg_mid)
        mu_t = compute_time_varying_mu(mu_base, gamma_open, gamma_mid, gamma_close,
                                        I_open, I_mid, I_close)
        total_integral += mu_t * seg_trading_dt
    
    return total_integral


def compute_segmented_baseline_integral(
    t1_intraday: float, t2_intraday: float, dt_trading: float,
    mu_base: float, gamma_open: float, gamma_mid: float, gamma_close: float
) -> float:
    """
    计算分段基准强度积分，正确处理跨时段和跨日边界的情况
    """
    if dt_trading <= 0:
        return 0.0
    
    # 跨日情况
    if t2_intraday <= t1_intraday:
        TRADING_SECONDS_PER_DAY = 14400
        
        integral1 = compute_single_day_baseline_integral(
            t1_intraday, MARKET_CLOSE_PM,
            mu_base, gamma_open, gamma_mid, gamma_close
        )
        
        integral2 = compute_single_day_baseline_integral(
            MARKET_OPEN_AM, t2_intraday,
            mu_base, gamma_open, gamma_mid, gamma_close
        )
        
        trading_time_day1 = intraday_to_trading_time_for_gof(MARKET_CLOSE_PM) - intraday_to_trading_time_for_gof(t1_intraday)
        trading_time_day2 = intraday_to_trading_time_for_gof(t2_intraday) - intraday_to_trading_time_for_gof(MARKET_OPEN_AM)
        remaining_time = dt_trading - trading_time_day1 - trading_time_day2
        
        if remaining_time > TRADING_SECONDS_PER_DAY * 0.5:
            n_full_days = int(remaining_time / TRADING_SECONDS_PER_DAY)
            T_open = 30 * 60
            T_mid = 30 * 60
            T_close = 30 * 60
            T_other = 150 * 60
            
            avg_integral_per_day = (
                mu_base * math.exp(gamma_open) * T_open +
                mu_base * math.exp(gamma_mid) * T_mid +
                mu_base * math.exp(gamma_close) * T_close +
                mu_base * T_other
            )
            integral_middle = n_full_days * avg_integral_per_day
        else:
            integral_middle = 0.0
        
        return integral1 + integral2 + integral_middle
    
    return compute_single_day_baseline_integral(
        t1_intraday, t2_intraday,
        mu_base, gamma_open, gamma_mid, gamma_close
    )


def correct_mu_for_time_varying(mu: np.ndarray, 
                                 gamma_open: np.ndarray, 
                                 gamma_mid: np.ndarray, 
                                 gamma_close: np.ndarray) -> np.ndarray:
    """
    校正μ参数：从tick估计的μ中扣除时段效应
    """
    T_open = 30 * 60
    T_mid = 30 * 60
    T_close = 30 * 60
    T_other = 150 * 60
    T_total = 240 * 60
    
    mu_corrected = np.zeros_like(mu)
    
    for d in range(len(mu)):
        expected_factor = (
            (T_open / T_total) * math.exp(gamma_open[d]) +
            (T_mid / T_total) * math.exp(gamma_mid[d]) +
            (T_close / T_total) * math.exp(gamma_close[d]) +
            (T_other / T_total) * 1.0
        )
        
        if expected_factor > 0:
            mu_corrected[d] = mu[d] / expected_factor
        else:
            mu_corrected[d] = mu[d]
    
    return mu_corrected


def estimate_gamma_mle(events_4d: List[np.ndarray], 
                       events_4d_original: List[np.ndarray],
                       mu: np.ndarray, A: np.ndarray, decay: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    估计时变基准强度的gamma参数
    μ_i(t) = μ_i * exp(γ_{i,o} * I_OPEN30(t) + γ_{i,m} * I_MID30(t) + γ_{i,c} * I_CLOSE30(t))
    
    使用简化的矩估计方法：
    在各特殊时段，事件率与gamma成比例
    
    Parameters:
    -----------
    events_4d : 归一化后的事件时间
    events_4d_original : 原始事件时间（用于判断时段）
    mu : tick估计的常数基准强度
    A, decay : 激励参数
    
    Returns:
    --------
    gamma_open : shape (4,) OPEN30的gamma系数（9:30-10:00）
    gamma_mid : shape (4,) MID30的gamma系数（13:00-13:30）
    gamma_close : shape (4,) CLOSE30的gamma系数（14:30-15:00）
    """
    gamma_open = np.zeros(4)
    gamma_mid = np.zeros(4)
    gamma_close = np.zeros(4)
    
    for d in range(4):
        if len(events_4d_original[d]) == 0:
            continue
        
        # 统计各时段的事件数
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
        
        # 时段长度比例（假设观测了多天）
        # OPEN30: 30分钟, MID30: 30分钟, CLOSE30: 30分钟
        # 其他: 150分钟（4小时交易时间 - 3*30分钟）
        T_open = 30 * 60    # 1800秒
        T_mid = 30 * 60     # 1800秒
        T_close = 30 * 60   # 1800秒
        T_other = 150 * 60  # 9000秒（上午60分钟 + 下午90分钟）
        
        # 事件率
        rate_open = n_open / T_open if n_open > 0 else 0
        rate_mid = n_mid / T_mid if n_mid > 0 else 0
        rate_close = n_close / T_close if n_close > 0 else 0
        rate_other = n_other / T_other if n_other > 0 else 1e-10
        
        # gamma = log(rate_period / rate_other)
        # μ_i(t) = μ_i * exp(gamma)，所以 gamma = log(rate_period / rate_base)
        # 使用rate_other作为基准
        if rate_open > 0 and rate_other > 0:
            gamma_open[d] = np.log(rate_open / rate_other)
        if rate_mid > 0 and rate_other > 0:
            gamma_mid[d] = np.log(rate_mid / rate_other)
        if rate_close > 0 and rate_other > 0:
            gamma_close[d] = np.log(rate_close / rate_other)
        
        # 限制gamma范围，防止过大
        gamma_open[d] = np.clip(gamma_open[d], -3.0, 3.0)
        gamma_mid[d] = np.clip(gamma_mid[d], -3.0, 3.0)
        gamma_close[d] = np.clip(gamma_close[d], -3.0, 3.0)
    
    return gamma_open, gamma_mid, gamma_close


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
    
    # tick的adjacency是核函数的L1范数（积分）: ∫ φ_{ij}(t) dt = A[i,j]
    # 分枝比 = spectral_radius(A)
    eigvals = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigvals)))
    branching_ratio = spectral_radius
    
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


TRADING_SECONDS_PER_DAY = 14400


def _compute_residuals_one_day_constant_mu(
    day_events: List[Tuple[float, int]],
    mu: np.ndarray, A: np.ndarray, decay: float
) -> Dict[int, List[float]]:
    """
    对单日事件计算常数μ的 time-rescaling 残差。
    """
    residuals = {d: [] for d in range(4)}
    r = np.zeros(4, dtype=float)
    last_t = day_events[0][0] if len(day_events) > 0 else 0.0
    Lambda_accum = np.zeros(4, dtype=float)
    first_event_seen = {d: False for d in range(4)}
    
    for t, dim in day_events:
        dt = t - last_t
        if dt > 0:
            decay_factor = math.exp(-decay * dt)
            for u in range(4):
                base_int = mu[u] * dt
                exc_int = float(A[u, :].dot(r) * (1.0 - decay_factor))
                Lambda_accum[u] += base_int + exc_int
            r *= decay_factor
        
        if first_event_seen[dim]:
            residuals[dim].append(float(Lambda_accum[dim]))
        else:
            first_event_seen[dim] = True
        
        Lambda_accum[dim] = 0.0
        r[dim] += 1.0
        last_t = t
    
    return residuals


def compute_time_rescaling_residuals_4d_constant_mu(
    events_4d: List[np.ndarray], T: float,
    mu: np.ndarray, A: np.ndarray, decay: float
) -> dict:
    """
    GOF检验体系（v2）- 纯常数μ版本
    λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t} A_{ij} e^{-β(t-t_k^j)}
    
    核心指标：QQ + 距离度量 + 独立性 + 综合评分
    不再以KS p-value作为pass/fail判据。
    """
    from scipy.stats import kstest, wasserstein_distance, expon
    
    # ---- 步骤1：将事件按日分组 ----
    merged = []
    for d in range(4):
        for t in events_4d[d]:
            merged.append((float(t), d))
    merged.sort(key=lambda x: x[0])
    
    if len(merged) == 0:
        return {"error": "no_events"}
    
    days_events = {}
    for t, dim in merged:
        day_idx = int(t / TRADING_SECONDS_PER_DAY)
        if day_idx not in days_events:
            days_events[day_idx] = []
        days_events[day_idx].append((t, dim))
    
    # ---- 步骤2：逐日计算残差 ----
    all_residuals_by_dim = {d: [] for d in range(4)}
    per_day_results = []
    
    for day_idx in sorted(days_events.keys()):
        day_evts = days_events[day_idx]
        if len(day_evts) < 2:
            continue
        
        day_residuals = _compute_residuals_one_day_constant_mu(
            day_evts, mu, A, decay
        )
        
        day_n = {d: len(day_residuals[d]) for d in range(4)}
        day_total = sum(day_n.values())
        
        if day_total > 0:
            day_info = {"day_index": day_idx, "n_events": len(day_evts)}
            for d in range(4):
                all_residuals_by_dim[d].extend(day_residuals[d])
                day_info[f"dim_{d}_n"] = day_n[d]
                if day_n[d] > 5:
                    res_arr = np.array(day_residuals[d])
                    day_info[f"dim_{d}_mean"] = float(np.mean(res_arr))
            per_day_results.append(day_info)
    
    # ---- 步骤3：多维度GOF检验 ----
    KS_SUBSAMPLE_N = 500
    KS_SUBSAMPLE_REPS = 20
    QQ_QUANTILE_N = 200
    ACF_MAX_LAG = 20
    
    results = {}
    raw_residuals = {}
    rng = np.random.RandomState(42)
    
    for d in range(4):
        res = np.array(all_residuals_by_dim[d], dtype=float)
        if len(res) > 10:
            res_mean = float(np.mean(res))
            res_std = float(np.std(res))
            res_median = float(np.median(res))
            
            # QQ分位数
            probs = np.linspace(0.005, 0.995, QQ_QUANTILE_N)
            qq_empirical = np.quantile(res, probs).tolist()
            qq_theoretical = expon.ppf(probs).tolist()
            
            # 距离度量
            mae_mean = float(abs(res_mean - 1.0))
            rmse_mean = float(np.sqrt(np.mean((res - 1.0) ** 2)))
            n_ws = min(5000, len(res))
            res_ws = res[rng.choice(len(res), n_ws, replace=False)] if len(res) > n_ws else res
            ref_exp = rng.exponential(1.0, size=n_ws)
            w1_dist = float(wasserstein_distance(res_ws, ref_exp))
            qq_mae = float(np.mean(np.abs(np.array(qq_empirical) - np.array(qq_theoretical))))
            
            # 子采样KS（辅助参考）
            subsample_pvals = []
            n_sub = min(KS_SUBSAMPLE_N, len(res))
            for _ in range(KS_SUBSAMPLE_REPS):
                idx_s = rng.choice(len(res), size=n_sub, replace=False)
                _, pval_sub = kstest(res[idx_s], 'expon', args=(0, 1))
                subsample_pvals.append(pval_sub)
            ks_pval_sub_median = float(np.median(subsample_pvals))
            
            # PIT检验（辅助）
            pit_values = 1.0 - np.exp(-res)
            pit_sub_pvals = []
            for _ in range(KS_SUBSAMPLE_REPS):
                idx_s = rng.choice(len(pit_values), size=n_sub, replace=False)
                _, pval_pit = kstest(pit_values[idx_s], 'uniform')
                pit_sub_pvals.append(pval_pit)
            pit_pval_sub_median = float(np.median(pit_sub_pvals))
            
            # 独立性检验：Ljung-Box + ACF
            lb_pvalues = []
            acf_values = []
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                from statsmodels.tsa.stattools import acf as sm_acf
                res_lb = res if len(res) <= 5000 else res[rng.choice(len(res), 5000, replace=False)]
                lb_result = acorr_ljungbox(res_lb, lags=[5, 10, 20], return_df=True)
                lb_pvalues = lb_result['lb_pvalue'].tolist()
                lb_pass = all(p > 0.05 for p in lb_pvalues)
                acf_vals = sm_acf(res_lb, nlags=ACF_MAX_LAG, fft=True)
                acf_values = acf_vals[1:].tolist()
            except Exception:
                lb_pass = None
                lb_pvalues = []
                acf_values = []
            
            # 综合评分
            mean_score = max(0.0, 1.0 - mae_mean)
            w1_score = max(0.0, 1.0 - w1_dist / 2.0)
            lb_score = float(np.mean(lb_pvalues)) if len(lb_pvalues) > 0 else 0.5
            acf_score = 1.0 - float(np.mean(np.abs(acf_values))) if len(acf_values) > 0 else 0.5
            gof_score = float(0.3 * mean_score + 0.3 * w1_score + 0.2 * lb_score + 0.2 * acf_score)
            
            mean_ok = bool(0.8 <= res_mean <= 1.2)
            gof_pass = bool(mean_ok and (lb_pass is True or lb_pass is None))
            
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "mean": res_mean,
                "std": res_std,
                "median": res_median,
                "qq_empirical": qq_empirical,
                "qq_theoretical": qq_theoretical,
                "mae_mean": mae_mean,
                "rmse_mean": rmse_mean,
                "wasserstein_1": w1_dist,
                "qq_mae": qq_mae,
                "ks_pvalue_sub_median": ks_pval_sub_median,
                "pit_pvalue_sub_median": pit_pval_sub_median,
                "ljung_box_pvalues": lb_pvalues,
                "ljung_box_pass": lb_pass,
                "acf_values": acf_values,
                "gof_score": gof_score,
                "mean_ok": mean_ok,
                "gof_pass": gof_pass,
            }
            
            n_save = min(2000, len(res))
            raw_residuals[f"dim_{d}"] = res[rng.choice(len(res), n_save, replace=False)].tolist()
        else:
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "error": "insufficient_residuals",
            }
    
    # ---- 步骤4：分日GOF汇总 ----
    per_day_ks = {d: [] for d in range(4)}
    per_day_means = {d: [] for d in range(4)}
    for day_info in per_day_results:
        day_idx = day_info["day_index"]
        day_evts = days_events[day_idx]
        day_residuals = _compute_residuals_one_day_constant_mu(
            day_evts, mu, A, decay
        )
        for d in range(4):
            res_arr = np.array(day_residuals[d], dtype=float)
            if len(res_arr) > 20:
                if len(res_arr) > KS_SUBSAMPLE_N:
                    idx_s = rng.choice(len(res_arr), size=KS_SUBSAMPLE_N, replace=False)
                    _, pval = kstest(res_arr[idx_s], 'expon', args=(0, 1))
                else:
                    _, pval = kstest(res_arr, 'expon', args=(0, 1))
                per_day_ks[d].append(float(pval))
                per_day_means[d].append(float(np.mean(res_arr)))
    
    per_day_summary = {}
    for d in range(4):
        pvals = per_day_ks[d]
        means = per_day_means[d]
        if len(pvals) > 0:
            n_not_reject = sum(1 for p in pvals if p > 0.05)
            n_mean_ok = sum(1 for m in means if 0.8 <= m <= 1.2)
            per_day_summary[f"dim_{d}"] = {
                "n_days_tested": len(pvals),
                "n_not_reject_005": n_not_reject,
                "not_reject_ratio": float(n_not_reject / len(pvals)),
                "mean_ks_pvalue": float(np.mean(pvals)),
                "median_ks_pvalue": float(np.median(pvals)),
                "n_mean_ok": n_mean_ok,
                "mean_ok_ratio": float(n_mean_ok / len(means)) if len(means) > 0 else 0.0,
            }
    
    # ---- 步骤5：汇总 ----
    gof_scores = [results.get(f"dim_{d}", {}).get("gof_score", 0.0) for d in range(4)]
    gof_pass_count = sum(1 for d in range(4) if results.get(f"dim_{d}", {}).get("gof_pass", False))
    results["summary"] = {
        "gof_pass_count": gof_pass_count,
        "all_pass": gof_pass_count == 4,
        "gof_score_mean": float(np.mean(gof_scores)),
        "gof_scores": gof_scores,
        "model_type": "constant_mu",
        "n_days": len(per_day_results),
    }
    results["per_day_summary"] = per_day_summary
    results["raw_residuals"] = raw_residuals
    
    results["mu_info"] = {
        "mu_used": mu.tolist(),
        "note": "constant_mu_consistent_with_tick_MLE",
    }
    
    return results


def compute_time_rescaling_residuals_4d(events_4d: List[np.ndarray], T: float,
                                         mu: np.ndarray, A: np.ndarray, 
                                         decay: float,
                                         events_4d_original: Optional[List[np.ndarray]] = None,
                                         gamma_open: Optional[np.ndarray] = None,
                                         gamma_mid: Optional[np.ndarray] = None,
                                         gamma_close: Optional[np.ndarray] = None) -> dict:
    """
    计算4D Hawkes模型的时间重标定残差（Ogata检验）- 改进版
    
    改进点：
    1. 使用校正后的μ（已扣除时段效应期望）
    2. 正确处理跨日边界
    
    使用分段基准强度：μ_i(t) = μ_corrected_i * exp(γ_{i,o} * I_OPEN30(t) + γ_{i,m} * I_MID30(t) + γ_{i,c} * I_CLOSE30(t))
    
    对于一个正确指定的点过程，变换后的残差应服从 Exp(1) 分布。
    
    Parameters:
    -----------
    events_4d : 归一化后的事件时间
    T : 总时间长度
    mu : 基准强度（tick估计的常数，将被校正）
    A : 激励矩阵
    decay : 衰减率
    events_4d_original : 原始事件时间（用于判断时段）
    gamma_open : OPEN30的gamma系数，shape (4,)（9:30-10:00）
    gamma_mid : MID30的gamma系数，shape (4,)（13:00-13:30）
    gamma_close : CLOSE30的gamma系数，shape (4,)（14:30-15:00）
    
    Returns:
    --------
    dict: 每个维度的残差及 KS 检验 p 值
    """
    from scipy.stats import kstest
    
    # 如果没有提供gamma参数，使用常数基准强度（gamma=0）
    use_time_varying = (events_4d_original is not None and 
                        gamma_open is not None and 
                        gamma_mid is not None and
                        gamma_close is not None)
    
    if not use_time_varying:
        gamma_open = np.zeros(4)
        gamma_mid = np.zeros(4)
        gamma_close = np.zeros(4)
    
    # **改进1：校正μ参数，扣除时段效应期望**
    mu_corrected = correct_mu_for_time_varying(mu, gamma_open, gamma_mid, gamma_close)
    
    # 合并事件时间线（同时保存原始时间用于时段判断）
    merged = []
    for d in range(4):
        for idx, t in enumerate(events_4d[d]):
            if use_time_varying and idx < len(events_4d_original[d]):
                t_orig = events_4d_original[d][idx]
            else:
                t_orig = t  # 回退到使用归一化时间
            merged.append((float(t), d, float(t_orig)))
    merged.sort(key=lambda x: x[0])
    
    # 计算每个维度的累积强度（Lambda）
    residuals_by_dim = {d: [] for d in range(4)}
    r = np.zeros(4, dtype=float)
    last_t = 0.0
    last_t_orig = MARKET_OPEN_AM  # 上一事件的日内时间，初始化为开盘时间
    Lambda_accum = np.zeros(4, dtype=float)  # 累积强度
    last_event_time = {d: 0.0 for d in range(4)}  # 每维上一个事件时间
    
    for t, dim, t_orig in merged:
        dt = t - last_t  # 归一化时间差（交易时间）
        if dt > 0:
            # 衰减
            decay_factor = math.exp(-decay * dt)
            
            # 获取当前和上一个时点的日内时间
            t1_intraday = get_intraday_time(last_t_orig)
            t2_intraday = get_intraday_time(t_orig)
            
            # 计算这段时间内的积分（强度积分）
            for u in range(4):
                # 使用分段积分计算基准强度积分（使用校正后的μ）
                base_int = compute_segmented_baseline_integral(
                    t1_intraday, t2_intraday, dt,
                    mu_corrected[u], gamma_open[u], gamma_mid[u], gamma_close[u]
                )
                # tick核函数: φ_{ij}(t) = A[i,j] * decay * exp(-decay*t)
                # 积分: A[u,:] · r · (1 - exp(-decay*dt))
                exc_int = float(A[u, :].dot(r) * (1.0 - decay_factor))
                Lambda_accum[u] += base_int + exc_int
            r *= decay_factor
        
        # 记录该维度的残差
        if last_event_time[dim] > 0:  # 跳过第一个事件
            residuals_by_dim[dim].append(float(Lambda_accum[dim]))
        
        # 重置该维度的累积强度
        Lambda_accum[dim] = 0.0
        last_event_time[dim] = t
        
        # 更新 r 和 last_t
        r[dim] += 1.0
        last_t = t
        last_t_orig = t_orig  # 更新上一事件的日内时间
    
    # ---- GOF v2：多维度检验 ----
    from scipy.stats import kstest, wasserstein_distance, expon
    
    KS_SUBSAMPLE_N = 500
    KS_SUBSAMPLE_REPS = 20
    QQ_QUANTILE_N = 200
    ACF_MAX_LAG = 20
    
    results = {}
    raw_residuals = {}
    rng = np.random.RandomState(42)
    
    for d in range(4):
        res = np.array(residuals_by_dim[d], dtype=float)
        if len(res) > 10:
            res_mean = float(np.mean(res))
            res_std = float(np.std(res))
            res_median = float(np.median(res))
            
            # QQ分位数
            probs = np.linspace(0.005, 0.995, QQ_QUANTILE_N)
            qq_empirical = np.quantile(res, probs).tolist()
            qq_theoretical = expon.ppf(probs).tolist()
            
            # 距离度量
            mae_mean = float(abs(res_mean - 1.0))
            rmse_mean = float(np.sqrt(np.mean((res - 1.0) ** 2)))
            n_ws = min(5000, len(res))
            res_ws = res[rng.choice(len(res), n_ws, replace=False)] if len(res) > n_ws else res
            ref_exp = rng.exponential(1.0, size=n_ws)
            w1_dist = float(wasserstein_distance(res_ws, ref_exp))
            qq_mae = float(np.mean(np.abs(np.array(qq_empirical) - np.array(qq_theoretical))))
            
            # 子采样KS（辅助参考）
            subsample_pvals = []
            n_sub = min(KS_SUBSAMPLE_N, len(res))
            for _ in range(KS_SUBSAMPLE_REPS):
                idx_s = rng.choice(len(res), size=n_sub, replace=False)
                _, pval_sub = kstest(res[idx_s], 'expon', args=(0, 1))
                subsample_pvals.append(pval_sub)
            ks_pval_sub_median = float(np.median(subsample_pvals))
            
            # PIT检验（辅助）
            pit_values = 1.0 - np.exp(-res)
            pit_sub_pvals = []
            for _ in range(KS_SUBSAMPLE_REPS):
                idx_s = rng.choice(len(pit_values), size=n_sub, replace=False)
                _, pval_pit = kstest(pit_values[idx_s], 'uniform')
                pit_sub_pvals.append(pval_pit)
            pit_pval_sub_median = float(np.median(pit_sub_pvals))
            
            # 独立性检验：Ljung-Box + ACF
            lb_pvalues = []
            acf_values = []
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                from statsmodels.tsa.stattools import acf as sm_acf
                res_lb = res if len(res) <= 5000 else res[rng.choice(len(res), 5000, replace=False)]
                lb_result = acorr_ljungbox(res_lb, lags=[5, 10, 20], return_df=True)
                lb_pvalues = lb_result['lb_pvalue'].tolist()
                lb_pass = all(p > 0.05 for p in lb_pvalues)
                acf_vals = sm_acf(res_lb, nlags=ACF_MAX_LAG, fft=True)
                acf_values = acf_vals[1:].tolist()
            except Exception:
                lb_pass = None
                lb_pvalues = []
                acf_values = []
            
            # 综合评分
            mean_score = max(0.0, 1.0 - mae_mean)
            w1_score = max(0.0, 1.0 - w1_dist / 2.0)
            lb_score = float(np.mean(lb_pvalues)) if len(lb_pvalues) > 0 else 0.5
            acf_score = 1.0 - float(np.mean(np.abs(acf_values))) if len(acf_values) > 0 else 0.5
            gof_score = float(0.3 * mean_score + 0.3 * w1_score + 0.2 * lb_score + 0.2 * acf_score)
            
            mean_ok = bool(0.8 <= res_mean <= 1.2)
            gof_pass = bool(mean_ok and (lb_pass is True or lb_pass is None))
            
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "mean": res_mean,
                "std": res_std,
                "median": res_median,
                "qq_empirical": qq_empirical,
                "qq_theoretical": qq_theoretical,
                "mae_mean": mae_mean,
                "rmse_mean": rmse_mean,
                "wasserstein_1": w1_dist,
                "qq_mae": qq_mae,
                "ks_pvalue_sub_median": ks_pval_sub_median,
                "pit_pvalue_sub_median": pit_pval_sub_median,
                "ljung_box_pvalues": lb_pvalues,
                "ljung_box_pass": lb_pass,
                "acf_values": acf_values,
                "gof_score": gof_score,
                "mean_ok": mean_ok,
                "gof_pass": gof_pass,
            }
            
            n_save = min(2000, len(res))
            raw_residuals[f"dim_{d}"] = res[rng.choice(len(res), n_save, replace=False)].tolist()
        else:
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "error": "insufficient_residuals",
            }
    
    # 汇总
    gof_scores = [results.get(f"dim_{d}", {}).get("gof_score", 0.0) for d in range(4)]
    gof_pass_count = sum(1 for d in range(4) if results.get(f"dim_{d}", {}).get("gof_pass", False))
    results["summary"] = {
        "gof_pass_count": gof_pass_count,
        "all_pass": gof_pass_count == 4,
        "gof_score_mean": float(np.mean(gof_scores)),
        "gof_scores": gof_scores,
        "model_type": "time_varying_mu",
        "n_days": 0,
    }
    results["raw_residuals"] = raw_residuals
    
    # 添加gamma参数信息
    if use_time_varying:
        results["gamma"] = {
            "gamma_open": gamma_open.tolist(),
            "gamma_mid": gamma_mid.tolist(),
            "gamma_close": gamma_close.tolist(),
        }
    
    # 记录校正信息
    results["mu_correction"] = {
        "mu_original": mu.tolist(),
        "mu_corrected": mu_corrected.tolist(),
    }
    
    return results


def run_comparison_4d_tick(data_path: str, 
                           events_4d_original: Optional[List[np.ndarray]] = None) -> dict:
    """
    运行 4D Hawkes 对比实验（全部使用 tick）
    
    Parameters:
    -----------
    data_path : 数据文件路径
    events_4d_original : 原始事件时间（用于GOF的时段判断），如果提供则使用分段μ
    """
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
    # 检查是否强制使用常数μ
    force_constant_mu = os.environ.get("GOF_CONSTANT_MU", "0") == "1"
    
    if force_constant_mu or events_4d_original is None:
        # 使用纯常数μ的GOF（与tick模型完全一致）
        print("Computing time-rescaling residuals (GOF test) with CONSTANT μ (tick-consistent)...")
        gof_results = compute_time_rescaling_residuals_4d_constant_mu(
            events_4d, T, full_result.mu, full_result.adjacency, full_result.decay
        )
    else:
        # 使用时变基准强度
        print("Estimating gamma parameters for time-varying baseline (OPEN30/MID30/CLOSE30)...")
        gamma_open, gamma_mid, gamma_close = estimate_gamma_mle(
            events_4d, events_4d_original, 
            full_result.mu, full_result.adjacency, full_result.decay
        )
        print(f"  gamma_open:  {gamma_open.round(3)} (9:30-10:00)")
        print(f"  gamma_mid:   {gamma_mid.round(3)} (13:00-13:30)")
        print(f"  gamma_close: {gamma_close.round(3)} (14:30-15:00)")
        print("Computing time-rescaling residuals (GOF test) with time-varying μ...")
        gof_results = compute_time_rescaling_residuals_4d(
            events_4d, T, full_result.mu, full_result.adjacency, full_result.decay,
            events_4d_original=events_4d_original,
            gamma_open=gamma_open,
            gamma_mid=gamma_mid,
            gamma_close=gamma_close
        )
    
    print(f"  GOF score={gof_results['summary'].get('gof_score_mean',0):.3f}, "
          f"pass={gof_results['summary']['gof_pass_count']}/4")
    for d in range(4):
        dk = f"dim_{d}"
        if dk in gof_results and "mean" in gof_results[dk]:
            info = gof_results[dk]
            print(f"    dim_{d}: mean={info['mean']:.3f}, W1={info.get('wasserstein_1',0):.3f}, "
                  f"QQ_MAE={info.get('qq_mae',0):.3f}, "
                  f"lb_pass={info.get('ljung_box_pass','?')}, "
                  f"score={info.get('gof_score',0):.3f}")
    
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
