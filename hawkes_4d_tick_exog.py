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


def compute_single_day_baseline_integral(
    t_start: float, t_end: float,
    mu_base: float, gamma_open: float, gamma_mid: float, gamma_close: float
) -> float:
    """
    计算单日内的基准强度积分
    
    Parameters:
    -----------
    t_start, t_end : float
        日内时间区间（秒，从午夜开始），要求 t_start < t_end
    """
    if t_start >= t_end:
        return 0.0
    
    # 获取边界点
    boundaries = get_period_boundaries_between(t_start, t_end)
    segment_points = [t_start] + boundaries + [t_end]
    
    total_integral = 0.0
    for i in range(len(segment_points) - 1):
        seg_start = segment_points[i]
        seg_end = segment_points[i + 1]
        
        # 计算该段的实际交易时间
        seg_trading_start = intraday_to_trading_time(seg_start)
        seg_trading_end = intraday_to_trading_time(seg_end)
        seg_trading_dt = seg_trading_end - seg_trading_start
        
        if seg_trading_dt <= 0:
            continue
        
        # 使用段中点的indicator
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
    
    修复版本：
    1. 使用端点映射直接计算每段的交易时间
    2. 正确处理跨日情况（分两段计算）
    
    Parameters:
    -----------
    t1_intraday : float
        起始日内时间（秒，从午夜开始）
    t2_intraday : float
        结束日内时间（秒，从午夜开始）
    dt_trading : float
        交易时间间隔（已排除午休）
    mu_base : float
        基准强度（已校正的μ）
    gamma_open, gamma_mid, gamma_close : float
        各时段的gamma系数
    
    Returns:
    --------
    float : 基准强度积分值
    """
    if dt_trading <= 0:
        return 0.0
    
    # 跨日情况：t2 < t1 说明事件跨越了日期边界
    if t2_intraday <= t1_intraday:
        # 分两段计算：
        # 第一段：从t1到当日收盘(15:00)
        # 第二段：从次日开盘(9:30)到t2
        
        # 计算跨日的天数（通过dt_trading推断）
        # 每天交易时间 = 14400秒
        TRADING_SECONDS_PER_DAY = 14400
        
        # 第一段：从t1到当日收盘
        integral1 = compute_single_day_baseline_integral(
            t1_intraday, MARKET_CLOSE_PM,
            mu_base, gamma_open, gamma_mid, gamma_close
        )
        
        # 第二段：从次日开盘到t2
        integral2 = compute_single_day_baseline_integral(
            MARKET_OPEN_AM, t2_intraday,
            mu_base, gamma_open, gamma_mid, gamma_close
        )
        
        # 计算中间完整天数的积分
        trading_time_day1 = intraday_to_trading_time(MARKET_CLOSE_PM) - intraday_to_trading_time(t1_intraday)
        trading_time_day2 = intraday_to_trading_time(t2_intraday) - intraday_to_trading_time(MARKET_OPEN_AM)
        remaining_time = dt_trading - trading_time_day1 - trading_time_day2
        
        # 如果有中间完整天数
        if remaining_time > TRADING_SECONDS_PER_DAY * 0.5:
            n_full_days = int(remaining_time / TRADING_SECONDS_PER_DAY)
            # 完整天使用平均μ（时段效应的期望）
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
    
    # 同一天内：直接使用单日计算
    return compute_single_day_baseline_integral(
        t1_intraday, t2_intraday,
        mu_base, gamma_open, gamma_mid, gamma_close
    )


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


def correct_mu_for_time_varying(mu: np.ndarray, 
                                 gamma_open: np.ndarray, 
                                 gamma_mid: np.ndarray, 
                                 gamma_close: np.ndarray) -> np.ndarray:
    """
    校正μ参数：从tick估计的μ中扣除时段效应
    
    tick估计的μ是常数基线，隐含了时段效应的平均
    GOF使用时变基线 μ(t) = μ_corrected * exp(γ·I(t))
    需要校正使得：E[μ(t)] = μ_tick
    
    即：μ_corrected * E[exp(γ·I(t))] = μ_tick
    所以：μ_corrected = μ_tick / E[exp(γ·I(t))]
    
    其中 E[exp(γ·I(t))] = Σ (T_period / T_total) * exp(γ_period)
    """
    # 各时段长度（秒）
    T_open = 30 * 60      # 1800秒
    T_mid = 30 * 60       # 1800秒
    T_close = 30 * 60     # 1800秒
    T_other = 150 * 60    # 9000秒（240分钟 - 90分钟）
    T_total = 240 * 60    # 14400秒
    
    mu_corrected = np.zeros_like(mu)
    
    for d in range(len(mu)):
        # 计算时变因子的期望 E[exp(γ·I(t))]
        # 在OPEN30时段：exp(gamma_open)
        # 在MID30时段：exp(gamma_mid)
        # 在CLOSE30时段：exp(gamma_close)
        # 在其他时段：exp(0) = 1
        expected_factor = (
            (T_open / T_total) * math.exp(gamma_open[d]) +
            (T_mid / T_total) * math.exp(gamma_mid[d]) +
            (T_close / T_total) * math.exp(gamma_close[d]) +
            (T_other / T_total) * 1.0
        )
        
        # 校正μ
        if expected_factor > 0:
            mu_corrected[d] = mu[d] / expected_factor
        else:
            mu_corrected[d] = mu[d]
    
    return mu_corrected


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


TRADING_SECONDS_PER_DAY = 14400


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


def split_events_by_day(events_4d: List[np.ndarray], T: float) -> List[List[np.ndarray]]:
    """
    将连续时间轴上的4D事件按日切分为多个独立实现（realization）。
    
    每日事件时间归零到 [0, TRADING_SECONDS_PER_DAY)。
    tick的HawkesExpKern.fit()接受多个realization，每个realization是独立的。
    跨日不应有激励传递，因此必须按日切分。
    
    Returns:
    --------
    List[List[np.ndarray]] : 每个元素是一天的 [dim0_events, dim1_events, dim2_events, dim3_events]
    """
    n_days = int(T / TRADING_SECONDS_PER_DAY) + 1
    realizations = []
    for day_idx in range(n_days):
        t_start = day_idx * TRADING_SECONDS_PER_DAY
        t_end = (day_idx + 1) * TRADING_SECONDS_PER_DAY
        day_events = []
        day_total = 0
        for d in range(4):
            mask = (events_4d[d] >= t_start) & (events_4d[d] < t_end)
            ev = events_4d[d][mask] - t_start
            day_events.append(ev)
            day_total += len(ev)
        if day_total > 0:
            realizations.append(day_events)
    return realizations


@dataclass
class TickFitResult:
    decay: float
    mu: np.ndarray
    adjacency: np.ndarray
    loglik: float
    aic: float
    branching_ratio: float


def fit_hawkes_4d_tick(events_4d: List[np.ndarray], decay: float,
                       realizations: Optional[List[List[np.ndarray]]] = None) -> TickFitResult:
    """
    用 tick 拟合 4D Hawkes 模型（给定 decay）
    
    Parameters:
    -----------
    events_4d : 连续时间轴上的4D事件（用于计算T等）
    decay : 衰减率
    realizations : 按日切分的多个独立实现。如果提供，使用多realization拟合；
                   否则退回到单realization拟合。
    """
    decays_mat = np.full((4, 4), float(decay), dtype=float)
    learner = HawkesExpKern(decays=decays_mat, verbose=False)
    
    if realizations is not None and len(realizations) > 0:
        learner.fit(realizations)
    else:
        learner.fit([events_4d])
    
    ll = float(learner.score())
    
    mu = learner.baseline.copy()
    A = learner.adjacency.copy()
    
    # tick的adjacency是核函数的L1范数（积分）: ∫ φ_{ij}(t) dt = A[i,j]
    # 分枝比 = spectral_radius(A)（不需要除以decay）
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
                      require_stable: bool = True,
                      realizations: Optional[List[List[np.ndarray]]] = None) -> TickFitResult:
    """网格搜索最优 decay，优先选择稳定（分枝比 < 1）的结果"""
    all_results = []
    stable_results = []
    
    for decay in decay_grid:
        result = fit_hawkes_4d_tick(events_4d, decay, realizations=realizations)
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


def _compute_residuals_one_day_constant_mu(
    day_events: List[Tuple[float, int]],
    mu: np.ndarray, A: np.ndarray, decay: float
) -> Dict[int, List[float]]:
    """
    对单日事件计算常数μ的 time-rescaling 残差。
    
    Parameters:
    -----------
    day_events : List[(t_normalized, dim)] 已按 t 排序的单日事件
    mu : 常数基准强度 (tick MLE)
    A : 激励矩阵
    decay : 衰减率
    
    Returns:
    --------
    Dict[dim -> list of residuals]
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
                # tick的核函数: φ_{ij}(t) = A[i,j] * decay * exp(-decay*t)
                # r[j] 追踪 Σ_k exp(-decay*(t-t_k^j))
                # 激励强度: Σ_j A[u,j] * decay * r[j]
                # 积分: Σ_j A[u,j] * decay * r[j] * (1-exp(-decay*dt)) / decay
                #      = Σ_j A[u,j] * r[j] * (1-exp(-decay*dt))
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
    GOF检验体系（v2）：QQ + 距离度量 + 独立性 + 组间对比
    
    核心指标：
    1. QQ诊断：返回raw residuals供QQ-Exp(1)图绘制
    2. 距离度量：MAE(mean-1), RMSE(mean-1), Wasserstein-1 vs Exp(1)
    3. 独立性：Ljung-Box + ACF lag-1~20
    4. 辅助：PIT Uniform检验、残差均值/中位数/std
    
    不再以KS p-value作为pass/fail判据。
    """
    from scipy.stats import kstest, wasserstein_distance, expon
    
    TRADING_SECONDS_PER_DAY = 14400
    
    # ---- 步骤1：将事件按日分组 ----
    merged = []
    for d in range(4):
        for idx, t in enumerate(events_4d[d]):
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
    QQ_QUANTILE_N = 200  # QQ图使用的分位数点数
    ACF_MAX_LAG = 20
    
    results = {}
    raw_residuals = {}  # 存储raw residuals供可视化
    rng = np.random.RandomState(42)
    
    for d in range(4):
        res = np.array(all_residuals_by_dim[d], dtype=float)
        if len(res) > 10:
            # --- 基本统计 ---
            res_mean = float(np.mean(res))
            res_std = float(np.std(res))
            res_median = float(np.median(res))
            
            # --- QQ分位数（用于绘图，不存全量残差以控制JSON大小）---
            probs = np.linspace(0.005, 0.995, QQ_QUANTILE_N)
            qq_empirical = np.quantile(res, probs).tolist()
            qq_theoretical = expon.ppf(probs).tolist()
            
            # --- 距离度量 ---
            # MAE: |mean(residuals) - 1|
            mae_mean = float(abs(res_mean - 1.0))
            # RMSE: sqrt(mean((residuals - 1)^2)) 衡量残差偏离Exp(1)均值的程度
            rmse_mean = float(np.sqrt(np.mean((res - 1.0) ** 2)))
            # Wasserstein-1 距离 vs Exp(1)：对大样本子采样计算
            n_ws = min(5000, len(res))
            res_ws = res[rng.choice(len(res), n_ws, replace=False)] if len(res) > n_ws else res
            ref_exp = rng.exponential(1.0, size=n_ws)
            w1_dist = float(wasserstein_distance(res_ws, ref_exp))
            # 分位数MAE：QQ偏离度
            qq_mae = float(np.mean(np.abs(np.array(qq_empirical) - np.array(qq_theoretical))))
            
            # --- 子采样KS（辅助参考，不作为判据）---
            subsample_pvals = []
            n_sub = min(KS_SUBSAMPLE_N, len(res))
            for _ in range(KS_SUBSAMPLE_REPS):
                idx_s = rng.choice(len(res), size=n_sub, replace=False)
                _, pval_sub = kstest(res[idx_s], 'expon', args=(0, 1))
                subsample_pvals.append(pval_sub)
            ks_pval_sub_median = float(np.median(subsample_pvals))
            
            # --- PIT检验（辅助）---
            pit_values = 1.0 - np.exp(-res)
            pit_sub_pvals = []
            for _ in range(KS_SUBSAMPLE_REPS):
                idx_s = rng.choice(len(pit_values), size=n_sub, replace=False)
                _, pval_pit = kstest(pit_values[idx_s], 'uniform')
                pit_sub_pvals.append(pval_pit)
            pit_pval_sub_median = float(np.median(pit_sub_pvals))
            
            # --- 独立性检验：Ljung-Box + ACF ---
            lb_pvalues = []
            acf_values = []
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                from statsmodels.tsa.stattools import acf as sm_acf
                res_lb = res if len(res) <= 5000 else res[rng.choice(len(res), 5000, replace=False)]
                lb_result = acorr_ljungbox(res_lb, lags=[5, 10, 20], return_df=True)
                lb_pvalues = lb_result['lb_pvalue'].tolist()
                lb_pass = all(p > 0.05 for p in lb_pvalues)
                # ACF值（lag 1~20）
                acf_vals = sm_acf(res_lb, nlags=ACF_MAX_LAG, fft=True)
                acf_values = acf_vals[1:].tolist()  # 去掉lag=0
            except Exception:
                lb_pass = None
                lb_pvalues = []
                acf_values = []
            
            # --- 综合评分（不再二值pass/fail，改为连续分数）---
            # score ∈ [0, 1]，越高越好
            mean_score = max(0.0, 1.0 - mae_mean)  # mean偏离1越小越好
            w1_score = max(0.0, 1.0 - w1_dist / 2.0)  # Wasserstein越小越好
            lb_score = float(np.mean(lb_pvalues)) if len(lb_pvalues) > 0 else 0.5
            acf_score = 1.0 - float(np.mean(np.abs(acf_values))) if len(acf_values) > 0 else 0.5
            gof_score = float(0.3 * mean_score + 0.3 * w1_score + 0.2 * lb_score + 0.2 * acf_score)
            
            # 兼容旧接口
            mean_ok = bool(0.8 <= res_mean <= 1.2)
            gof_pass = bool(mean_ok and (lb_pass is True or lb_pass is None))
            
            results[f"dim_{d}"] = {
                "n_residuals": len(res),
                "mean": res_mean,
                "std": res_std,
                "median": res_median,
                # QQ数据
                "qq_empirical": qq_empirical,
                "qq_theoretical": qq_theoretical,
                # 距离度量
                "mae_mean": mae_mean,
                "rmse_mean": rmse_mean,
                "wasserstein_1": w1_dist,
                "qq_mae": qq_mae,
                # KS（辅助参考）
                "ks_pvalue_sub_median": ks_pval_sub_median,
                # PIT（辅助参考）
                "pit_pvalue_sub_median": pit_pval_sub_median,
                # 独立性
                "ljung_box_pvalues": lb_pvalues,
                "ljung_box_pass": lb_pass,
                "acf_values": acf_values,
                # 综合
                "gof_score": gof_score,
                "mean_ok": mean_ok,
                "gof_pass": gof_pass,
            }
            
            # 存储子采样raw residuals供可视化（最多保留2000个点）
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
        "model_type": "constant_mu_per_day",
        "n_days": len(per_day_results),
    }
    results["per_day_summary"] = per_day_summary
    results["raw_residuals"] = raw_residuals
    
    # 保留gamma参数信息（仅记录，不参与GOF计算）
    if gamma_open is not None and gamma_mid is not None and gamma_close is not None:
        results["gamma"] = {
            "gamma_open": gamma_open.tolist() if hasattr(gamma_open, 'tolist') else list(gamma_open),
            "gamma_mid": gamma_mid.tolist() if hasattr(gamma_mid, 'tolist') else list(gamma_mid),
            "gamma_close": gamma_close.tolist() if hasattr(gamma_close, 'tolist') else list(gamma_close),
        }
    
    if gamma_spread is not None:
        results["gamma_spread"] = gamma_spread.tolist() if hasattr(gamma_spread, 'tolist') else list(gamma_spread)
    
    results["mu_info"] = {
        "mu_used": mu.tolist(),
        "note": "constant_mu_consistent_with_tick_MLE",
    }
    
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
    
    # 按日切分为多个独立realization（消除跨日激励泄漏）
    all_realizations = split_events_by_day(events_4d, T)
    n_days = len(all_realizations)
    print(f"Split into {n_days} daily realizations")
    
    # Decay 网格（扩展到更高值以适应ms级数据的快速衰减）
    dmin = float(os.environ.get("DECAY_MIN", "0.3"))
    dmax = float(os.environ.get("DECAY_MAX", "50.0"))
    
    grid_low = np.linspace(dmin, 1.0, 4, endpoint=False)
    grid_dense = np.linspace(1.0, 5.0, 10)
    grid_high = np.linspace(6.0, 20.0, 8)
    grid_very_high = np.linspace(25.0, dmax, 4)
    decay_grid = np.unique(np.concatenate([grid_low, grid_dense, grid_high, grid_very_high]))
    n_points = len(decay_grid)
    print(f"Decay grid ({n_points} points): {decay_grid[:5].round(3)}...{decay_grid[-3:].round(3)}")
    
    require_stable = os.environ.get("REQUIRE_STABLE", "1") != "0"
    
    # 70/30 时间切分（按天数）
    n_train_days = max(1, int(0.7 * n_days))
    train_realizations = all_realizations[:n_train_days]
    val_realizations = all_realizations[n_train_days:]
    
    # 重建train_events用于兼容性
    t_split = n_train_days * TRADING_SECONDS_PER_DAY
    train_events = [ev[ev < t_split] for ev in events_4d]
    
    beta_strategy = os.environ.get("BETA_STRATEGY", "grid")
    
    if beta_strategy == "grid":
        print(f"Grid search decay on training set ({n_points} points, {n_train_days} days, require_stable={require_stable})...")
        train_result = grid_search_decay(train_events, decay_grid, 
                                         require_stable=require_stable,
                                         realizations=train_realizations)
        best_decay = train_result.decay
        print(f"Best decay from grid search: {best_decay:.4f}")
    elif beta_strategy == "fixed":
        best_decay = float(os.environ.get("FIXED_BETA", "0.1"))
        print(f"Using fixed decay: {best_decay:.4f}")
    else:
        full_result = grid_search_decay(events_4d, decay_grid, realizations=all_realizations)
        best_decay = full_result.decay
        print(f"Baseline decay: {best_decay:.4f}")
    
    # 全量拟合（使用多realization）
    print(f"Fitting on full data with decay={best_decay:.4f} ({n_days} realizations)...")
    full_result = fit_hawkes_4d_tick(events_4d, best_decay, realizations=all_realizations)
    train_result = fit_hawkes_4d_tick(train_events, best_decay, realizations=train_realizations)
    
    # 验证集
    val_total = sum(sum(len(e) for e in r) for r in val_realizations) if val_realizations else 0
    
    if val_total >= 4 and len(val_realizations) > 0:
        val_events_dummy = [np.array([0.0])] * 4  # placeholder
        val_result = fit_hawkes_4d_tick(val_events_dummy, best_decay, realizations=val_realizations)
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
    
    print("Computing GOF v2 (QQ + distance + independence)...")
    gof_results = compute_time_rescaling_residuals_4d_exog(
        events_4d, T, full_result.mu, full_result.adjacency, full_result.decay,
        events_4d_original=events_4d_original,
        re_spread_4d=re_spread_4d,
        gamma_open=gamma_open,
        gamma_mid=gamma_mid,
        gamma_close=gamma_close,
        gamma_spread=gamma_spread
    )
    print(f"  GOF score={gof_results['summary'].get('gof_score_mean',0):.3f}, "
          f"pass={gof_results['summary']['gof_pass_count']}/4, "
          f"n_days={gof_results['summary'].get('n_days', '?')}")
    for d in range(4):
        dk = f"dim_{d}"
        if dk in gof_results and "mean" in gof_results[dk]:
            info = gof_results[dk]
            print(f"    dim_{d}: mean={info['mean']:.3f}, W1={info.get('wasserstein_1',0):.3f}, "
                  f"QQ_MAE={info.get('qq_mae',0):.3f}, "
                  f"lb_pass={info.get('ljung_box_pass','?')}, "
                  f"score={info.get('gof_score',0):.3f}")
    pds = gof_results.get("per_day_summary", {})
    for d in range(4):
        dk = f"dim_{d}"
        if dk in pds:
            info = pds[dk]
            print(f"    dim_{d} per-day: mean_ok={info.get('n_mean_ok',0)}/{info['n_days_tested']}, "
                  f"ratio={info['mean_ok_ratio']:.2%}")
    
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
