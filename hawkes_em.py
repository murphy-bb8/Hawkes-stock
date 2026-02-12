"""
hawkes_em.py  —  基于 EM 算法的多维 Hawkes 过程估计器
=====================================================
核心设计：
  1. 不依赖 tick 包，纯 numpy/scipy 实现
  2. β (omega/decay) 作为超参数，通过网格搜索 + 对数似然选优
  3. Model A: 常数 μ
     Model B: μ_i(t) = μ_base_i · exp(γ_open·I_OPEN + γ_mid·I_MID + γ_close·I_CLOSE)
     Model C: Model B + γ_spread · re_spread(t)
  4. 拟合与 GOF 使用同一套强度函数口径
  5. 核函数口径: φ_{ij}(Δt) = α[i,j] · ω · exp(-ω · Δt)
     积分: ∫_0^∞ φ_{ij}(s)ds = α[i,j]
  6. EM / LL / GOF 全部使用连续递推（不逐日重置 r），口径完全一致
  7. 优先使用 Cython 加速（_hawkes_cython），fallback 到纯 Python

参考：
  - stmorse/hawkes (GitHub) 的 EM 实现
  - MHP_fixed_final_play.py 中的 EM_func / MHP.EM
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.stats import kstest, wasserstein_distance, expon

# ===================== Cython 加速 (可选) =====================
_USE_CYTHON = False
try:
    from _hawkes_cython import (
        em_recursive_cython,
        loglikelihood_cython,
        gof_residuals_cython,
    )
    _USE_CYTHON = True
except ImportError:
    pass

# log-link 专用加速
_USE_CYTHON_LOGLINK = False
try:
    from hawkes_cy_loglink import (
        precompute_segments_cy,
        compute_loglink_integrals_cy,
    )
    _USE_CYTHON_LOGLINK = True
except ImportError:
    pass


# ===================== A 股交易时间常量 =====================
MARKET_OPEN_AM  = 34200   # 9:30
MARKET_CLOSE_AM = 41400   # 11:30
MARKET_OPEN_PM  = 46800   # 13:00
MARKET_CLOSE_PM = 54000   # 15:00

OPEN30_START  = 34200     # 9:30
OPEN30_END    = 36000     # 10:00
MID30_START   = 46800     # 13:00
MID30_END     = 48600     # 13:30
CLOSE30_START = 52200     # 14:30
CLOSE30_END   = 54000     # 15:00

TRADING_SECONDS_PER_DAY = 14400  # 4h

PERIOD_BOUNDARIES = [
    OPEN30_END, MARKET_CLOSE_AM, MARKET_OPEN_PM, MID30_END, CLOSE30_START
]


# ===================== 时间工具函数 =====================

def intraday_to_trading_time(t_intraday: float) -> float:
    """日内时间 → 交易时间 (0‑14400)"""
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


def get_intraday_time(t: float) -> float:
    if 0 <= t < 86400:
        return t
    return t % 86400


def compute_indicators(t_intraday: float) -> Tuple[float, float, float]:
    """返回 (I_open30, I_mid30, I_close30)"""
    I_o = 1.0 if OPEN30_START <= t_intraday < OPEN30_END else 0.0
    I_m = 1.0 if MID30_START <= t_intraday < MID30_END else 0.0
    I_c = 1.0 if CLOSE30_START <= t_intraday < CLOSE30_END else 0.0
    return I_o, I_m, I_c


def _boundaries_between(t1: float, t2: float) -> List[float]:
    if t1 >= t2:
        return []
    return sorted(b for b in PERIOD_BOUNDARIES if t1 < b < t2)


# ===================== 交易时间逆映射 =====================

def _trading_time_to_intraday(t_trading: float) -> float:
    """交易时间 (0‑14400 within a day) → 日内时间 (34200‑54000)。
    intraday_to_trading_time 的逆函数。"""
    t_trading = max(0.0, min(t_trading, 14400.0))
    if t_trading <= 7200:                       # AM: 0‑7200 → 34200‑41400
        return MARKET_OPEN_AM + t_trading
    else:                                       # PM: 7200‑14400 → 46800‑54000
        return MARKET_OPEN_PM + (t_trading - 7200)


# ===================== SpreadProcess (Kramer 外生协变量) =====================

class SpreadProcess:
    """
    独立外生 spread 协变量过程  (Kramer 2021 exogenous-factor style)。

    设计要点 (P0-1 修复):
      1. 时间点来自 **所有事件类型的并集**，避免只用某一类事件采样导致的
         内生性问题。
      2. 支持可选滞后 (lag) —— 使用 x_s(t - Δ) 保证严格外生。
      3. 默认 z-score 标准化，避免 γ_spread 因尺度极端。
      4. piecewise constant (method='previous') 或线性插值。
    """

    def __init__(self, times: np.ndarray, values: np.ndarray,
                 method: str = 'previous', lag: float = 0.0,
                 standardize: bool = True):
        assert len(times) == len(values), "times/values 长度不匹配"
        idx = np.argsort(times)
        self.raw_times = np.asarray(times, dtype=np.float64)[idx]
        self.raw_values = np.asarray(values, dtype=np.float64)[idx]
        self.method = method
        self.lag = float(lag)

        # z-score 标准化
        self.mean_ = 0.0
        self.std_ = 1.0
        if standardize and len(self.raw_values) > 1:
            self.mean_ = float(np.mean(self.raw_values))
            self.std_ = float(np.std(self.raw_values))
            if self.std_ < 1e-10:
                self.std_ = 1.0
        self.times = self.raw_times.copy()
        self.values = ((self.raw_values - self.mean_) / self.std_
                       if standardize else self.raw_values.copy())

    # --- 查询 ---
    def value_at(self, t: float) -> float:
        """piecewise-constant (或 linear) 查询 x_s(t - lag)。"""
        t_q = t - self.lag
        n = len(self.times)
        if n == 0:
            return 0.0
        if t_q <= self.times[0]:
            return float(self.values[0])
        if t_q >= self.times[-1]:
            return float(self.values[-1])
        idx = int(np.searchsorted(self.times, t_q, side='right')) - 1
        if self.method == 'linear' and idx < n - 1:
            dt = self.times[idx + 1] - self.times[idx]
            if dt > 1e-15:
                frac = (t_q - self.times[idx]) / dt
                return float(self.values[idx]
                             + frac * (self.values[idx + 1] - self.values[idx]))
        return float(self.values[idx])

    def values_at_times(self, query_times: np.ndarray) -> np.ndarray:
        """向量化查询。"""
        out = np.empty(len(query_times), dtype=np.float64)
        for i in range(len(query_times)):
            out[i] = self.value_at(float(query_times[i]))
        return out

    @property
    def n_points(self) -> int:
        return len(self.times)

    def __repr__(self):
        return (f"SpreadProcess(n={self.n_points}, method={self.method!r}, "
                f"lag={self.lag}, std={self.std_:.4f})")


# ===================== 模拟器 =====================

def simulate_hawkes_multi(mu: np.ndarray, alpha: np.ndarray,
                          omega: float, T: float,
                          seed: int = 42) -> np.ndarray:
    """
    Ogata thinning 模拟多维 Hawkes 过程。

    Parameters
    ----------
    mu    : (D,) 基线强度
    alpha : (D,D) 激励矩阵  alpha[i,j] = j→i 的激励系数
    omega : float 衰减率 (所有维度共享)
    T     : float 模拟时长
    seed  : int

    Returns
    -------
    data : (N, 2)  列 0=时间, 列 1=维度编号
    """
    rng = np.random.RandomState(seed)
    dim = mu.shape[0]
    data = []

    Istar = np.sum(mu)
    if Istar <= 0:
        return np.empty((0, 2))

    s = rng.exponential(1.0 / Istar)
    if s >= T:
        return np.empty((0, 2))

    n0 = rng.choice(dim, p=mu / Istar)
    data.append([s, int(n0)])

    lastrates = mu.copy()
    decIstar = False

    while True:
        tj = data[-1][0]
        uj = int(data[-1][1])

        if decIstar:
            Istar = np.sum(rates)
            decIstar = False
        else:
            Istar = np.sum(lastrates) + omega * np.sum(alpha[:, uj])

        if Istar <= 1e-15:
            break

        s += rng.exponential(1.0 / Istar)
        if s >= T:
            break

        rates = mu + np.exp(-omega * (s - tj)) * \
                (alpha[:, uj].flatten() * omega + lastrates - mu)
        rates = np.maximum(rates, 0.0)

        diff = Istar - np.sum(rates)
        if diff < 0:
            diff = 0.0
            rates = rates * (Istar / np.sum(rates))

        probs = np.append(rates, diff) / Istar
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()

        n0 = rng.choice(dim + 1, p=probs)
        if n0 < dim:
            data.append([s, int(n0)])
            lastrates = rates.copy()
        else:
            decIstar = True

    if len(data) == 0:
        return np.empty((0, 2))
    arr = np.array(data)
    return arr[arr[:, 0] < T]


# ===================== EM 算法 =====================

def _em_recursive_python(times: np.ndarray, types: np.ndarray,
                          dim: int, omega: float,
                          mu_hat: np.ndarray, alpha_hat: np.ndarray,
                          Tm: float, maxiter: int, tol: float,
                          verbose: bool,
                          intraday_times: Optional[np.ndarray] = None,
                          gamma_open: Optional[np.ndarray] = None,
                          gamma_mid: Optional[np.ndarray] = None,
                          gamma_close: Optional[np.ndarray] = None,
                          gamma_spread: Optional[np.ndarray] = None,
                          spread_at_events: Optional[np.ndarray] = None,
                          spread_integral_T: Optional[np.ndarray] = None,
                          baseline_integral_T: Optional[np.ndarray] = None,
                          ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    纯 Python 递推 EM（O(N·D) 每次迭代，无 N×N 矩阵）。
    联合估计 mu, alpha, gamma_spread（Model C）。

    口径：φ_{ij}(Δt) = α[i,j] · ω · exp(-ω · Δt)

    统一强度函数（与 LL / GOF 完全一致）：
      Model A: λ_d(t) = μ_d + Σ_j α_{dj}·ω·r_j(t)
      Model B: λ_d(t) = μ_base_d·exp(γ·I(t)) + Σ_j α_{dj}·ω·r_j(t)
      Model C: Model B + γ_spread_d · re_spread(t)

    γ_spread M-step 闭式解：
      γ_spread[d] = Σ_{i:u_i=d} p_exog_i / ∫_0^T max(spread(t),0) dt
    """
    N = len(times)
    use_tv = (intraday_times is not None and gamma_open is not None)
    use_spread = (spread_at_events is not None)

    # 初始化 gamma_spread
    if use_spread and gamma_spread is None:
        gamma_spread = np.full(dim, 0.01)
    if use_spread:
        gamma_spread = gamma_spread.copy()

    # 预计算 dt 和 decay
    dt_arr = np.diff(times, prepend=times[0])
    dt_arr[0] = 0.0
    decay_arr = np.exp(-omega * dt_arr)  # (N,)

    # 预计算每个事件时刻的时变基线乘子 exp(γ·I(t))
    tv_multiplier = None
    if use_tv:
        tv_multiplier = np.ones((N, dim))  # (N, dim)
        for i in range(N):
            I_o, I_m, I_c = compute_indicators(intraday_times[i])
            for d in range(dim):
                tv_multiplier[i, d] = math.exp(
                    gamma_open[d] * I_o + gamma_mid[d] * I_m + gamma_close[d] * I_c)

    if baseline_integral_T is None:
        baseline_integral_T = np.full(dim, Tm)

    # spread_integral_T[d] = ∫_0^T max(spread(t),0) dt，用于 γ_spread M-step
    if use_spread and spread_integral_T is None:
        spread_integral_T = np.full(dim, Tm * 0.01)

    old_LL = -1e15
    for it in range(maxiter):
        # 充分统计量
        sum_p_bg = np.zeros(dim)
        sum_pij_by_type = np.zeros((dim, dim))
        sum_p_exog = np.zeros(dim)     # Σ_{i:u_i=d} p_exog_i
        type_counts = np.zeros(dim)
        sum_log_rates = 0.0

        r = np.zeros(dim)
        for i in range(N):
            u_i = types[i]
            if i > 0:
                r *= decay_arr[i]

            # 基线项
            if use_tv:
                mu_val = mu_hat[u_i] * tv_multiplier[i, u_i]
            else:
                mu_val = mu_hat[u_i]

            # 激励项
            exc = alpha_hat[u_i] * omega * r  # (dim,)
            exc_sum = exc.sum()

            # 外生项
            exog_val = 0.0
            if use_spread:
                sp_i = max(spread_at_events[i], 0.0)
                exog_val = gamma_spread[u_i] * sp_i

            lam_i = mu_val + exc_sum + exog_val
            lam_i = max(lam_i, 1e-15)

            # E-step: 各成分责任
            inv_lam = 1.0 / lam_i
            sum_p_bg[u_i] += mu_val * inv_lam

            p_exc = exc * inv_lam  # (dim,)
            sum_pij_by_type[u_i] += p_exc

            if use_spread and exog_val > 0.0:
                sum_p_exog[u_i] += exog_val * inv_lam

            type_counts[u_i] += 1
            sum_log_rates += math.log(lam_i)
            r[u_i] += 1.0

        # --- M step ---
        for k in range(dim):
            mu_hat[k] = max(sum_p_bg[k] / max(baseline_integral_T[k], 1e-10), 1e-10)
        for a in range(dim):
            for b in range(dim):
                alpha_hat[a, b] = max(sum_pij_by_type[a, b] / max(type_counts[b], 1.0), 0.0)

        # γ_spread M-step: γ[d] = Σ p_exog[d] / ∫_0^T spread(t) dt
        if use_spread:
            for d in range(dim):
                gamma_spread[d] = max(sum_p_exog[d] / max(spread_integral_T[d], 1e-10), 0.0)

        # --- 收敛检查 ---
        if it % 5 == 0:
            integral_alpha = 0.0
            for d in range(dim):
                integral_alpha += np.sum(alpha_hat[d, types])
            mu_integral = sum(mu_hat[d] * baseline_integral_T[d] for d in range(dim))
            exog_integral = 0.0
            if use_spread:
                exog_integral = sum(gamma_spread[d] * spread_integral_T[d] for d in range(dim))
            LL = (sum_log_rates - mu_integral - integral_alpha - exog_integral) / N
            if verbose and it % 20 == 0:
                print(f"  EM iter {it}: LL={LL:.4f}")
            if abs(LL - old_LL) < tol:
                if verbose:
                    print(f"  EM converged at iter {it}: LL={LL:.4f}")
                break
            old_LL = LL

    return alpha_hat, mu_hat, gamma_spread if use_spread else None


def _compute_baseline_integral_T(dim: int, Tm: float, n_days: int,
                                  gamma_open: Optional[np.ndarray],
                                  gamma_mid: Optional[np.ndarray],
                                  gamma_close: Optional[np.ndarray]) -> np.ndarray:
    """
    计算时变基线的等效积分时长：∫_0^T exp(γ·I(t)) dt。
    用于 EM M-step 中 μ_base 的更新。

    对于常数基线（Model A），返回 Tm。
    对于时变基线（Model B/C），按日内时段分段计算。
    """
    if gamma_open is None:
        return np.full(dim, Tm)

    T_open = 30 * 60    # 1800s
    T_mid = 30 * 60
    T_close = 30 * 60
    T_other = 150 * 60  # 9000s
    result = np.zeros(dim)
    for d in range(dim):
        day_integral = (T_open * math.exp(gamma_open[d]) +
                        T_mid * math.exp(gamma_mid[d]) +
                        T_close * math.exp(gamma_close[d]) +
                        T_other * 1.0)
        result[d] = day_integral * max(n_days, 1)
    return result


def _compute_spread_integral_T(dim: int, spread_times: np.ndarray,
                                spread_values: np.ndarray, T: float) -> np.ndarray:
    """
    计算 ∫_0^T max(re_spread(t), 0) dt（标量，对所有维度相同）。
    用于 γ_spread 的 M-step 分母。
    使用分段常数近似（左端点插值）。
    """
    if spread_times is None or len(spread_times) == 0:
        return np.full(dim, T * 0.01)
    total = 0.0
    for i in range(len(spread_times)):
        t_start = spread_times[i]
        t_end = spread_times[i + 1] if i + 1 < len(spread_times) else T
        if t_end > T:
            t_end = T
        dt = t_end - t_start
        if dt <= 0:
            continue
        total += max(spread_values[i], 0.0) * dt
    return np.full(dim, max(total, 1e-10))


def _precompute_segments(T: float, n_days: int,
                         spread_proc: Optional['SpreadProcess'] = None,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    预计算所有分段的属性（与 gamma 无关，只依赖时间轴结构）。
    返回向量化数组 (K,) 用于快速积分计算：
      seg_dt, seg_Io, seg_Im, seg_Ic, seg_xs
    """
    # Cython 加速路径（暂时禁用，待修复spread索引bug）
    if False and _USE_CYTHON_LOGLINK:
        spread_times = spread_proc.times if spread_proc is not None else None
        seg_dt, seg_Io, seg_Im, seg_Ic, seg_xs_idx = precompute_segments_cy(
            T, n_days, spread_times)
        # 将索引转换为实际值
        if spread_proc is not None and len(seg_xs_idx) > 0:
            seg_xs = spread_proc.values[seg_xs_idx]
        else:
            seg_xs = np.zeros(len(seg_dt))
        return seg_dt, seg_Io, seg_Im, seg_Ic, seg_xs
    
    # Python fallback
    dummy_offsets_in_day = [0.0, 1800.0, 7200.0, 9000.0, 12600.0, TRADING_SECONDS_PER_DAY]
    all_bounds = set()
    all_bounds.add(0.0)
    all_bounds.add(T)
    for day in range(max(n_days, 1)):
        off = day * TRADING_SECONDS_PER_DAY
        for b in dummy_offsets_in_day:
            tb = off + b
            if 0.0 < tb < T:
                all_bounds.add(tb)
    if spread_proc is not None:
        for ts in spread_proc.times:
            if 0.0 < ts < T:
                all_bounds.add(ts)
    bounds = np.array(sorted(all_bounds))

    K = len(bounds) - 1
    if K <= 0:
        z = np.zeros(0)
        return z, z, z, z, z

    seg_dt = np.diff(bounds)
    mids = (bounds[:-1] + bounds[1:]) * 0.5

    seg_Io = np.zeros(K)
    seg_Im = np.zeros(K)
    seg_Ic = np.zeros(K)
    seg_xs = np.zeros(K)

    for k in range(K):
        day_idx = int(mids[k] / TRADING_SECONDS_PER_DAY)
        t_in_day = mids[k] - day_idx * TRADING_SECONDS_PER_DAY
        t_intraday = _trading_time_to_intraday(t_in_day)
        seg_Io[k], seg_Im[k], seg_Ic[k] = compute_indicators(t_intraday)
        if spread_proc is not None:
            seg_xs[k] = spread_proc.value_at(mids[k])

    # 过滤掉 dt<=0 的段
    mask = seg_dt > 1e-15
    return seg_dt[mask], seg_Io[mask], seg_Im[mask], seg_Ic[mask], seg_xs[mask]


def _compute_loglink_integrals(
        dim: int, T: float, n_days: int,
        gamma_open: Optional[np.ndarray],
        gamma_mid: Optional[np.ndarray],
        gamma_close: Optional[np.ndarray],
        gamma_spread: Optional[np.ndarray],
        spread_proc: Optional['SpreadProcess'],
        _seg_cache: Optional[Tuple] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    log-link 基线的所有积分量（向量化版本）。

    返回 (dim,) 数组：
      eff_T         = ∫_0^T exp(η_d(t)) dt
      eff_T_open    = ∫_0^T I_open(t) · exp(η_d(t)) dt
      eff_T_mid     = ∫_0^T I_mid(t) · exp(η_d(t)) dt
      eff_T_close   = ∫_0^T I_close(t) · exp(η_d(t)) dt
      eff_T_spread  = ∫_0^T x_s(t) · exp(η_d(t)) dt

    _seg_cache: 预计算结果 (seg_dt, seg_Io, seg_Im, seg_Ic, seg_xs)，
                L-BFGS 多次调用时避免重复构建段。
    """
    if _seg_cache is not None:
        seg_dt, seg_Io, seg_Im, seg_Ic, seg_xs = _seg_cache
    else:
        seg_dt, seg_Io, seg_Im, seg_Ic, seg_xs = _precompute_segments(
            T, n_days, spread_proc if (spread_proc is not None and gamma_spread is not None) else None)

    K = len(seg_dt)
    if K == 0:
        z = np.zeros(dim)
        return z, z, z, z, z

    use_gamma = (gamma_open is not None)
    use_spread = (gamma_spread is not None and spread_proc is not None)

    # Cython 加速路径（需要重新构建 seg_xs_idx）
    if _USE_CYTHON_LOGLINK and K > 1000:  # 大规模数据才用Cython
        # 从seg_xs反推seg_xs_idx（简化：直接传seg_xs作为spread_values）
        # 注意：Cython版本需要seg_xs_idx，但这里我们已经有seg_xs了
        # 为了简化，直接用numpy向量化（已经很快了）
        pass
    
    # Numpy 向量化路径（已经很快）
    eff_T = np.zeros(dim)
    eff_T_open = np.zeros(dim)
    eff_T_mid = np.zeros(dim)
    eff_T_close = np.zeros(dim)
    eff_T_spread = np.zeros(dim)

    for d in range(dim):
        # 构建 η (K,) — 向量化
        eta = np.zeros(K)
        if use_gamma:
            eta += gamma_open[d] * seg_Io + gamma_mid[d] * seg_Im + gamma_close[d] * seg_Ic
        if use_spread:
            eta += gamma_spread[d] * seg_xs
        exp_eta = np.exp(eta)           # (K,)
        w = exp_eta * seg_dt            # (K,)
        eff_T[d] = w.sum()
        eff_T_open[d] = (seg_Io * w).sum()
        eff_T_mid[d] = (seg_Im * w).sum()
        eff_T_close[d] = (seg_Ic * w).sum()
        if use_spread:
            eff_T_spread[d] = (seg_xs * w).sum()

    return eff_T, eff_T_open, eff_T_mid, eff_T_close, eff_T_spread


def em_estimate(seq: np.ndarray, dim: int, omega: float,
                Tm: float = -1.0,
                maxiter: int = 100, tol: float = 1e-4,
                verbose: bool = False,
                intraday_times: Optional[np.ndarray] = None,
                gamma_open: Optional[np.ndarray] = None,
                gamma_mid: Optional[np.ndarray] = None,
                gamma_close: Optional[np.ndarray] = None,
                spread_at_events: Optional[np.ndarray] = None,
                spread_times: Optional[np.ndarray] = None,
                spread_values: Optional[np.ndarray] = None,
                n_days: int = 1,
                ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    固定 omega(β) 的 EM 算法联合估计 mu, alpha, gamma_spread。

    全量数据，无子采样。优先使用 Cython 加速，fallback 到纯 Python。

    统一强度函数（与 LL / GOF 完全一致）：
      Model A: λ_d(t) = μ_d + Σ_j α_{dj}·ω·r_j(t)
      Model B: λ_d(t) = μ_base_d·exp(γ·I(t)) + Σ_j α_{dj}·ω·r_j(t)
      Model C: Model B + γ_spread_d · re_spread(t)

    Returns
    -------
    alpha_hat    : (dim, dim)
    mu_hat       : (dim,)  — 对于 Model B/C 这是 μ_base
    gamma_spread : (dim,) 或 None — Model C 联合估计的外生项系数
    """
    N = len(seq)
    if N < 2:
        return np.zeros((dim, dim)), np.ones(dim) * 0.01, None

    if Tm < 0:
        Tm = float(seq[-1, 0])
    if Tm <= 0:
        Tm = 1.0

    times = seq[:, 0].astype(np.float64)
    types = seq[:, 1].astype(np.intc)

    # 初始化
    rng = np.random.RandomState(0)
    mu_hat = rng.uniform(0.01, 0.5, size=dim).astype(np.float64)
    alpha_hat = rng.uniform(0.01, 0.3, size=(dim, dim)).astype(np.float64)

    # 计算时变基线等效积分时长
    use_tv = (intraday_times is not None and gamma_open is not None)
    baseline_integral_T = _compute_baseline_integral_T(
        dim, Tm, n_days, gamma_open, gamma_mid, gamma_close) if use_tv else None

    use_spread = (spread_at_events is not None)

    # 计算 spread 积分（γ_spread M-step 分母）
    spread_integral_T = None
    gamma_spread_init = None
    if use_spread:
        spread_integral_T = _compute_spread_integral_T(
            dim, spread_times, spread_values, Tm)
        gamma_spread_init = np.full(dim, 0.01)

    if _USE_CYTHON:
        # Cython 快速路径（支持 Model A/B/C）
        cy_kwargs = {}
        if use_tv:
            cy_kwargs['intraday_times'] = np.ascontiguousarray(intraday_times, dtype=np.float64)
            cy_kwargs['gamma_open'] = np.ascontiguousarray(gamma_open, dtype=np.float64)
            cy_kwargs['gamma_mid'] = np.ascontiguousarray(gamma_mid, dtype=np.float64)
            cy_kwargs['gamma_close'] = np.ascontiguousarray(gamma_close, dtype=np.float64)
            cy_kwargs['baseline_integral_T'] = np.ascontiguousarray(baseline_integral_T, dtype=np.float64)
        if use_spread:
            cy_kwargs['gamma_spread'] = np.ascontiguousarray(gamma_spread_init, dtype=np.float64)
            cy_kwargs['spread_at_events'] = np.ascontiguousarray(spread_at_events, dtype=np.float64)
            cy_kwargs['spread_integral_T'] = np.ascontiguousarray(spread_integral_T, dtype=np.float64)
        alpha_hat, mu_hat, gamma_spread_out = em_recursive_cython(
            np.ascontiguousarray(times),
            np.ascontiguousarray(types),
            dim, omega,
            np.ascontiguousarray(mu_hat),
            np.ascontiguousarray(alpha_hat),
            Tm, maxiter, tol, int(verbose),
            **cy_kwargs)
    else:
        # 纯 Python fallback（支持 Model A/B/C）
        alpha_hat, mu_hat, gamma_spread_out = _em_recursive_python(
            times, types.astype(int), dim, omega, mu_hat, alpha_hat,
            Tm, maxiter, tol, verbose,
            intraday_times=intraday_times,
            gamma_open=gamma_open,
            gamma_mid=gamma_mid,
            gamma_close=gamma_close,
            gamma_spread=gamma_spread_init,
            spread_at_events=spread_at_events,
            spread_integral_T=spread_integral_T,
            baseline_integral_T=baseline_integral_T)

    return alpha_hat, mu_hat, gamma_spread_out


# ===================== 对数似然 =====================

def _loglikelihood_python(times: np.ndarray, types: np.ndarray,
                          dim: int, mu: np.ndarray,
                          alpha: np.ndarray, omega: float,
                          Tm: float,
                          intraday_times: Optional[np.ndarray] = None,
                          gamma_open: Optional[np.ndarray] = None,
                          gamma_mid: Optional[np.ndarray] = None,
                          gamma_close: Optional[np.ndarray] = None,
                          gamma_spread: Optional[np.ndarray] = None,
                          spread_at_events: Optional[np.ndarray] = None,
                          baseline_integral_T: Optional[np.ndarray] = None,
                          exog_integral: float = 0.0,
                          ) -> float:
    """
    纯 Python 递推对数似然（O(N·D)）。

    统一强度函数（与 EM / GOF 完全一致）：
      Model A: λ_i(t) = μ_i + Σ_j α_{ij}·ω·r_j(t)
      Model B: λ_i(t) = μ_base_i·exp(γ·I(t)) + Σ_j α_{ij}·ω·r_j(t)
      Model C: Model B + γ_spread_i · re_spread(t)

    口径：φ_{ij}(Δt) = α[i,j] · ω · exp(-ω · Δt)
    LL = Σ_i log λ_{u_i}(t_i) - Σ_d ∫_0^T λ_d(t) dt
    积分项: ∫_0^T λ_d(t) dt = μ_d·∫exp(γ·I)dt + Σ_j α[d,u_j]·(1-exp(-ω·(T-t_j))) + exog_integral
    """
    N = len(times)
    use_tv = (intraday_times is not None and gamma_open is not None)
    use_spread = (gamma_spread is not None and spread_at_events is not None)

    r = np.zeros(dim)
    sum_log_lam = 0.0
    for i in range(N):
        u_i = types[i]
        if i > 0:
            dt_val = times[i] - times[i - 1]
            if dt_val > 0:
                r *= math.exp(-omega * dt_val)

        # 基线项
        if use_tv:
            I_o, I_m, I_c = compute_indicators(intraday_times[i])
            mu_val = mu[u_i] * math.exp(
                gamma_open[u_i] * I_o + gamma_mid[u_i] * I_m + gamma_close[u_i] * I_c)
        else:
            mu_val = mu[u_i]

        # 激励项
        lam_i = mu_val + np.dot(alpha[u_i, :] * omega, r)

        # 外生项
        if use_spread:
            lam_i += max(gamma_spread[u_i] * spread_at_events[i], 0.0)

        lam_i = max(lam_i, 1e-15)
        sum_log_lam += math.log(lam_i)
        r[u_i] += 1.0

    # 积分项
    # 基线积分：μ_d · ∫_0^T exp(γ·I(t)) dt
    if baseline_integral_T is not None:
        integral = sum(mu[d] * baseline_integral_T[d] for d in range(dim))
    else:
        integral = Tm * np.sum(mu)

    # 激励积分：Σ_j α[d, u_j] · (1 - exp(-ω·(T-t_j)))
    surv = 1.0 - np.exp(-omega * (Tm - times))  # (N,)
    for d in range(dim):
        integral += np.sum(alpha[d, types] * surv)

    # 外生项积分
    integral += exog_integral

    return sum_log_lam - integral


def loglikelihood(seq: np.ndarray, dim: int, mu: np.ndarray,
                  alpha: np.ndarray, omega: float,
                  Tm: float = -1.0,
                  intraday_times: Optional[np.ndarray] = None,
                  gamma_open: Optional[np.ndarray] = None,
                  gamma_mid: Optional[np.ndarray] = None,
                  gamma_close: Optional[np.ndarray] = None,
                  gamma_spread: Optional[np.ndarray] = None,
                  spread_at_events: Optional[np.ndarray] = None,
                  baseline_integral_T: Optional[np.ndarray] = None,
                  exog_integral: float = 0.0,
                  ) -> float:
    """
    精确对数似然（递推计算，O(N·D)），全量数据无子采样。

    统一强度函数（与 EM / GOF 完全一致）。

    LL = Σ_i log λ_{u_i}(t_i) - Σ_d ∫_0^T λ_d(t) dt
    """
    N = len(seq)
    if N < 1:
        return -1e15
    times = seq[:, 0].astype(np.float64)
    types = seq[:, 1].astype(np.intc)
    if Tm < 0:
        Tm = float(times[-1])
    if Tm <= 0:
        Tm = 1.0

    use_tv = (intraday_times is not None and gamma_open is not None)
    use_spread = (gamma_spread is not None and spread_at_events is not None)

    if _USE_CYTHON:
        # Cython 快速路径（支持 Model A/B/C）
        cy_kwargs = {}
        if use_tv:
            cy_kwargs['intraday_times'] = np.ascontiguousarray(intraday_times, dtype=np.float64)
            cy_kwargs['gamma_open'] = np.ascontiguousarray(gamma_open, dtype=np.float64)
            cy_kwargs['gamma_mid'] = np.ascontiguousarray(gamma_mid, dtype=np.float64)
            cy_kwargs['gamma_close'] = np.ascontiguousarray(gamma_close, dtype=np.float64)
        if use_spread:
            cy_kwargs['gamma_spread'] = np.ascontiguousarray(gamma_spread, dtype=np.float64)
            cy_kwargs['spread_at_events'] = np.ascontiguousarray(spread_at_events, dtype=np.float64)
        if baseline_integral_T is not None:
            cy_kwargs['baseline_integral_T'] = np.ascontiguousarray(baseline_integral_T, dtype=np.float64)
        if exog_integral != 0.0:
            cy_kwargs['exog_integral'] = exog_integral
        return loglikelihood_cython(
            np.ascontiguousarray(times),
            np.ascontiguousarray(types),
            dim,
            np.ascontiguousarray(mu.astype(np.float64)),
            np.ascontiguousarray(alpha.astype(np.float64)),
            omega, Tm,
            **cy_kwargs)
    else:
        return _loglikelihood_python(
            times, types.astype(int), dim, mu, alpha, omega, Tm,
            intraday_times=intraday_times,
            gamma_open=gamma_open, gamma_mid=gamma_mid, gamma_close=gamma_close,
            gamma_spread=gamma_spread, spread_at_events=spread_at_events,
            baseline_integral_T=baseline_integral_T,
            exog_integral=exog_integral)


# ===================== β 网格搜索 =====================

def grid_search_beta(seq: np.ndarray, dim: int,
                     beta_grid: np.ndarray,
                     Tm: float = -1.0,
                     maxiter: int = 80,
                     verbose: bool = True,
                     intraday_times: Optional[np.ndarray] = None,
                     gamma_open: Optional[np.ndarray] = None,
                     gamma_mid: Optional[np.ndarray] = None,
                     gamma_close: Optional[np.ndarray] = None,
                     spread_at_events: Optional[np.ndarray] = None,
                     spread_times: Optional[np.ndarray] = None,
                     spread_values: Optional[np.ndarray] = None,
                     n_days: int = 1,
                     ) -> Tuple[float, np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    网格搜索最优 β，每个 β 用 EM 联合估计 (mu, alpha, gamma_spread)。

    EM 和 LL 使用完全一致的强度函数（含时变基线和外生项）。

    Returns
    -------
    best_beta, best_alpha, best_mu, best_ll, best_gamma_spread
    """
    best_ll = -1e30
    best_beta = float(beta_grid[0])
    best_alpha = None
    best_mu = None
    best_gamma_spread = None

    use_tv = (intraday_times is not None and gamma_open is not None)
    use_spread = (spread_at_events is not None)

    # 预计算（与 beta 无关的量）
    bl_int_T = _compute_baseline_integral_T(
        dim, Tm, n_days, gamma_open, gamma_mid, gamma_close) if use_tv else None

    def _run_one(beta_val):
        alpha_hat, mu_hat, gs_hat = em_estimate(
            seq, dim, float(beta_val), Tm=Tm, maxiter=maxiter, verbose=False,
            intraday_times=intraday_times,
            gamma_open=gamma_open, gamma_mid=gamma_mid, gamma_close=gamma_close,
            spread_at_events=spread_at_events,
            spread_times=spread_times, spread_values=spread_values,
            n_days=n_days)
        # 计算 LL（含外生项积分）
        exog_int = 0.0
        if gs_hat is not None:
            exog_int = _compute_exog_integral(dim, gs_hat, spread_times, spread_values, Tm)
        ll = loglikelihood(
            seq, dim, mu_hat, alpha_hat, float(beta_val), Tm=Tm,
            intraday_times=intraday_times,
            gamma_open=gamma_open, gamma_mid=gamma_mid, gamma_close=gamma_close,
            gamma_spread=gs_hat, spread_at_events=spread_at_events,
            baseline_integral_T=bl_int_T, exog_integral=exog_int)
        return alpha_hat, mu_hat, gs_hat, ll

    for beta in beta_grid:
        alpha_hat, mu_hat, gs_hat, ll = _run_one(beta)

        eigvals = np.linalg.eigvals(alpha_hat)
        br = float(np.max(np.abs(eigvals)))
        if br >= 1.0:
            if verbose:
                print(f"  β={beta:.2f}: BR={br:.4f} (unstable), skip")
            continue

        if verbose:
            gs_str = f", γ_sp={gs_hat.round(3)}" if gs_hat is not None else ""
            print(f"  β={beta:.2f}: LL={ll:.2f}, BR={br:.4f}, μ={mu_hat.round(4)}{gs_str}")
        if ll > best_ll:
            best_ll = ll
            best_beta = float(beta)
            best_alpha = alpha_hat.copy()
            best_mu = mu_hat.copy()
            best_gamma_spread = gs_hat.copy() if gs_hat is not None else None

    if best_alpha is None:
        if verbose:
            print("  No stable solution, re-running to pick min BR...")
        min_br = 1e10
        for beta in beta_grid:
            alpha_hat, mu_hat, gs_hat, ll = _run_one(beta)
            eigvals = np.linalg.eigvals(alpha_hat)
            br = float(np.max(np.abs(eigvals)))
            if br < min_br:
                min_br = br
                best_beta = float(beta)
                best_alpha = alpha_hat.copy()
                best_mu = mu_hat.copy()
                best_ll = ll
                best_gamma_spread = gs_hat.copy() if gs_hat is not None else None

    return best_beta, best_alpha, best_mu, best_ll, best_gamma_spread


# ===================== 哑变量时变基线 (Model B/C) =====================

def estimate_gamma_from_events(events_4d_original: List[np.ndarray],
                               dim: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用事件率比值估计 γ_open, γ_mid, γ_close。

    γ_period = log(rate_period / rate_other)

    Parameters
    ----------
    events_4d_original : 各维度的日内时间数组 (34200‑54000)

    Returns
    -------
    gamma_open, gamma_mid, gamma_close : (dim,)
    """
    gamma_open = np.zeros(dim)
    gamma_mid = np.zeros(dim)
    gamma_close = np.zeros(dim)

    T_open = 30 * 60
    T_mid = 30 * 60
    T_close = 30 * 60
    T_other = 150 * 60

    for d in range(dim):
        if len(events_4d_original[d]) == 0:
            continue
        n_open = n_mid = n_close = n_other = 0
        for t in events_4d_original[d]:
            t_id = get_intraday_time(t)
            if OPEN30_START <= t_id < OPEN30_END:
                n_open += 1
            elif MID30_START <= t_id < MID30_END:
                n_mid += 1
            elif CLOSE30_START <= t_id < CLOSE30_END:
                n_close += 1
            else:
                n_other += 1
        rate_other = max(n_other / T_other, 1e-10)
        if n_open > 0:
            gamma_open[d] = np.clip(np.log(n_open / T_open / rate_other), -3, 3)
        if n_mid > 0:
            gamma_mid[d] = np.clip(np.log(n_mid / T_mid / rate_other), -3, 3)
        if n_close > 0:
            gamma_close[d] = np.clip(np.log(n_close / T_close / rate_other), -3, 3)

    return gamma_open, gamma_mid, gamma_close


def correct_mu_for_gamma(mu: np.ndarray,
                         gamma_open: np.ndarray,
                         gamma_mid: np.ndarray,
                         gamma_close: np.ndarray) -> np.ndarray:
    """
    校正 μ：μ_corrected = μ_tick / E[exp(γ·I)]
    使得 E[μ(t)] 在全天平均意义下等于 μ_tick。
    """
    T_open = 30 * 60
    T_mid = 30 * 60
    T_close = 30 * 60
    T_other = 150 * 60
    T_total = 240 * 60

    mu_c = np.zeros_like(mu)
    for d in range(len(mu)):
        ef = ((T_open / T_total) * math.exp(gamma_open[d]) +
              (T_mid / T_total) * math.exp(gamma_mid[d]) +
              (T_close / T_total) * math.exp(gamma_close[d]) +
              (T_other / T_total) * 1.0)
        mu_c[d] = mu[d] / max(ef, 1e-10)
    return mu_c


def mu_at_time(mu_base: float, gamma_o: float, gamma_m: float, gamma_c: float,
               I_o: float, I_m: float, I_c: float) -> float:
    """μ(t) = μ_base · exp(γ_o·I_o + γ_m·I_m + γ_c·I_c)"""
    return mu_base * math.exp(gamma_o * I_o + gamma_m * I_m + gamma_c * I_c)


# ===================== L-BFGS 基线参数优化 (Kramer log-link) =====================

def _optimize_baseline_params(
        seq: np.ndarray, dim: int,
        alpha: np.ndarray, omega: float,
        T: float, n_days: int,
        model: str,
        spread_proc: Optional['SpreadProcess'] = None,
        gamma_open_init: Optional[np.ndarray] = None,
        gamma_mid_init: Optional[np.ndarray] = None,
        gamma_close_init: Optional[np.ndarray] = None,
        mu_init: Optional[np.ndarray] = None,
        verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    L-BFGS-B 优化基线参数: (log_mu, γ_open, γ_mid, γ_close, γ_spread)。
    激励参数 (alpha, omega) 保持不变。

    对应 Kramer (2021) exogenous-factor 的 log-link 基线：
      λ_d(t) = μ_d · exp(η_d(t)) + Σ_j α_{dj}·ω·r_j(t)
      η_d(t) = γ_o[d]·I_o(t) + γ_m[d]·I_m(t) + γ_c[d]·I_c(t) + γ_s[d]·x_s(t)

    Returns: mu, gamma_open, gamma_mid, gamma_close, gamma_spread, ll
    """
    from scipy.optimize import minimize as sp_minimize

    N = len(seq)
    times = seq[:, 0].astype(np.float64)
    types = seq[:, 1].astype(int)

    # --- 预计算激励项（与基线参数无关）---
    exc_at_events = np.zeros(N)
    r_vec = np.zeros(dim)
    for i in range(N):
        u_i = types[i]
        if i > 0:
            dt_val = times[i] - times[i - 1]
            if dt_val > 0:
                r_vec *= math.exp(-omega * dt_val)
        exc_at_events[i] = np.dot(alpha[u_i, :] * omega, r_vec)
        r_vec[u_i] += 1.0

    # --- 预计算激励积分（常数项）---
    surv = 1.0 - np.exp(-omega * (T - times))
    exc_integral = 0.0
    for d in range(dim):
        exc_integral += np.sum(alpha[d, types] * surv)

    # --- 预计算 dummy indicators / spread at events ---
    I_open_ev = np.zeros(N)
    I_mid_ev = np.zeros(N)
    I_close_ev = np.zeros(N)
    for i in range(N):
        day_idx = int(times[i] / TRADING_SECONDS_PER_DAY)
        t_in_day = times[i] - day_idx * TRADING_SECONDS_PER_DAY
        t_intraday = _trading_time_to_intraday(t_in_day)
        I_open_ev[i], I_mid_ev[i], I_close_ev[i] = compute_indicators(t_intraday)

    sp_at_ev = None
    if model == "C" and spread_proc is not None:
        sp_at_ev = spread_proc.values_at_times(times)

    # --- 预计算分段缓存（与 gamma 无关，只算一次）---
    _seg_cache = _precompute_segments(
        T, n_days,
        spread_proc if model == "C" else None)

    # --- 参数 pack / unpack ---
    def _pack(log_mu, g_o, g_m, g_c, g_s):
        parts = [log_mu]
        if model in ("B", "C"):
            parts.extend([g_o, g_m, g_c])
        if model == "C":
            parts.append(g_s)
        return np.concatenate(parts)

    def _unpack(x):
        idx = 0
        lm = x[idx:idx + dim]; idx += dim
        g_o = g_m = g_c = np.zeros(dim)
        g_s = np.zeros(dim)
        if model in ("B", "C"):
            g_o = x[idx:idx + dim]; idx += dim
            g_m = x[idx:idx + dim]; idx += dim
            g_c = x[idx:idx + dim]; idx += dim
        if model == "C":
            g_s = x[idx:idx + dim]; idx += dim
        return lm, g_o, g_m, g_c, g_s

    def _neg_ll_grad(x):
        log_mu, g_o, g_m, g_c, g_s = _unpack(x)
        mu = np.exp(log_mu)

        # η at events — 向量化
        eta_ev = np.zeros(N)
        if model in ("B", "C"):
            eta_ev = (g_o[types] * I_open_ev + g_m[types] * I_mid_ev
                      + g_c[types] * I_close_ev)
        if model == "C" and sp_at_ev is not None:
            eta_ev += g_s[types] * sp_at_ev

        base_ev = mu[types] * np.exp(eta_ev)
        lam_ev = base_ev + exc_at_events
        lam_ev = np.maximum(lam_ev, 1e-15)

        # event term
        sum_log_lam = np.sum(np.log(lam_ev))

        # baseline integrals (使用预计算分段缓存)
        eff_T, eff_T_o, eff_T_m, eff_T_c, eff_T_s = _compute_loglink_integrals(
            dim, T, n_days,
            g_o if model in ("B", "C") else None,
            g_m if model in ("B", "C") else None,
            g_c if model in ("B", "C") else None,
            g_s if model == "C" else None,
            spread_proc if model == "C" else None,
            _seg_cache=_seg_cache)

        base_integral = np.sum(mu * eff_T)
        nll = -(sum_log_lam - base_integral - exc_integral)

        # --- gradient ---
        w = base_ev / lam_ev   # baseline responsibility at each event

        # ∂LL/∂log_mu_d
        grad_lm = np.zeros(dim)
        for d in range(dim):
            mask = (types == d)
            grad_lm[d] = mu[d] * (np.sum(np.exp(eta_ev[mask]) / lam_ev[mask]) - eff_T[d])

        grad_parts = [-grad_lm]

        if model in ("B", "C"):
            grad_go = np.zeros(dim)
            grad_gm = np.zeros(dim)
            grad_gc = np.zeros(dim)
            for d in range(dim):
                mask = (types == d)
                exp_eta_masked = np.exp(eta_ev[mask])
                inv_lam_masked = 1.0 / lam_ev[mask]
                grad_go[d] = mu[d] * (np.sum(exp_eta_masked * I_open_ev[mask] * inv_lam_masked) - eff_T_o[d])
                grad_gm[d] = mu[d] * (np.sum(exp_eta_masked * I_mid_ev[mask] * inv_lam_masked) - eff_T_m[d])
                grad_gc[d] = mu[d] * (np.sum(exp_eta_masked * I_close_ev[mask] * inv_lam_masked) - eff_T_c[d])
            grad_parts.extend([-grad_go, -grad_gm, -grad_gc])

        if model == "C" and sp_at_ev is not None:
            grad_gs = np.zeros(dim)
            for d in range(dim):
                mask = (types == d)
                exp_eta_masked = np.exp(eta_ev[mask])
                inv_lam_masked = 1.0 / lam_ev[mask]
                grad_gs[d] = mu[d] * (np.sum(exp_eta_masked * sp_at_ev[mask] * inv_lam_masked) - eff_T_s[d])
            grad_parts.append(-grad_gs)

        return nll, np.concatenate(grad_parts)

    # --- 初始值 ---
    if mu_init is None:
        type_counts = np.zeros(dim)
        for i in range(N):
            type_counts[types[i]] += 1
        mu_init = np.maximum(type_counts / max(T, 1.0), 1e-6)
    if gamma_open_init is None:
        gamma_open_init = np.zeros(dim)
    if gamma_mid_init is None:
        gamma_mid_init = np.zeros(dim)
    if gamma_close_init is None:
        gamma_close_init = np.zeros(dim)

    x0 = _pack(np.log(np.maximum(mu_init, 1e-8)),
               gamma_open_init, gamma_mid_init, gamma_close_init,
               np.zeros(dim))

    # --- bounds: gamma 限制在 [-5, 5] 避免 exp 溢出 ---
    bounds_list = [(None, None)] * dim  # log_mu 无约束
    if model in ("B", "C"):
        bounds_list += [(-5.0, 5.0)] * (3 * dim)  # gamma_open/mid/close
    if model == "C":
        bounds_list += [(-5.0, 5.0)] * dim  # gamma_spread

    res = sp_minimize(_neg_ll_grad, x0, method='L-BFGS-B', jac=True,
                      bounds=bounds_list,
                      options={'maxiter': 200, 'ftol': 1e-10})

    log_mu_hat, g_o, g_m, g_c, g_s = _unpack(res.x)
    mu_hat = np.exp(log_mu_hat)
    ll = -res.fun

    if verbose:
        print(f"  L-BFGS converged={res.success}, nit={res.nit}, LL={ll:.2f}")
        print(f"    μ = {mu_hat.round(4)}")
        if model in ("B", "C"):
            print(f"    γ_open  = {g_o.round(3)}")
            print(f"    γ_mid   = {g_m.round(3)}")
            print(f"    γ_close = {g_c.round(3)}")
        if model == "C":
            print(f"    γ_spread = {g_s.round(4)}")

    return mu_hat, g_o, g_m, g_c, g_s, ll


def loglikelihood_loglink(
        seq: np.ndarray, dim: int,
        mu: np.ndarray, alpha: np.ndarray, omega: float,
        T: float,
        n_days: int = 1,
        gamma_open: Optional[np.ndarray] = None,
        gamma_mid: Optional[np.ndarray] = None,
        gamma_close: Optional[np.ndarray] = None,
        gamma_spread: Optional[np.ndarray] = None,
        spread_proc: Optional['SpreadProcess'] = None,
        model: str = "A",
        _seg_cache: Optional[Tuple] = None,
) -> float:
    """
    统一 log-link 对数似然（A/B/C 通用，O(N·D)）。

    λ_d(t) = μ_d · exp(η_d(t)) + Σ_j α_{dj}·ω·r_j(t)
    LL = Σ_i log λ_{u_i}(t_i) − Σ_d [ μ_d·∫exp(η_d)dt + Σ_j α_{dj}(1−e^{−ω(T−t_j)}) ]
    """
    N = len(seq)
    if N < 1:
        return -1e15
    times = seq[:, 0].astype(np.float64)
    types = seq[:, 1].astype(int)

    use_gamma = model in ("B", "C") and gamma_open is not None
    use_spread = model == "C" and spread_proc is not None and gamma_spread is not None

    # spread at events
    sp_at_ev = None
    if use_spread:
        sp_at_ev = spread_proc.values_at_times(times)

    # event term: Σ log λ
    r = np.zeros(dim)
    sum_log_lam = 0.0
    for i in range(N):
        u_i = types[i]
        if i > 0:
            dt_val = times[i] - times[i - 1]
            if dt_val > 0:
                r *= math.exp(-omega * dt_val)

        # η
        eta = 0.0
        if use_gamma:
            day_idx = int(times[i] / TRADING_SECONDS_PER_DAY)
            t_in_day = times[i] - day_idx * TRADING_SECONDS_PER_DAY
            t_intraday = _trading_time_to_intraday(t_in_day)
            I_o, I_m, I_c = compute_indicators(t_intraday)
            eta = gamma_open[u_i] * I_o + gamma_mid[u_i] * I_m + gamma_close[u_i] * I_c
        if use_spread:
            eta += gamma_spread[u_i] * sp_at_ev[i]

        base = mu[u_i] * math.exp(eta)
        exc = np.dot(alpha[u_i, :] * omega, r)
        lam_i = max(base + exc, 1e-15)
        sum_log_lam += math.log(lam_i)
        r[u_i] += 1.0

    # integral: baseline + excitation
    eff_T, _, _, _, _ = _compute_loglink_integrals(
        dim, T, n_days,
        gamma_open if use_gamma else None,
        gamma_mid if use_gamma else None,
        gamma_close if use_gamma else None,
        gamma_spread if use_spread else None,
        spread_proc if use_spread else None,
        _seg_cache=_seg_cache)
    base_integral = np.sum(mu * eff_T)

    surv = 1.0 - np.exp(-omega * (T - times))
    exc_integral = 0.0
    for d in range(dim):
        exc_integral += np.sum(alpha[d, types] * surv)

    return sum_log_lam - base_integral - exc_integral


# ===================== 统一 GOF 检验 (log-link) =====================

def _compute_baseline_integral_segment(t1_id: float, t2_id: float,
                                       dt_trading: float,
                                       mu_base: float,
                                       g_o: float, g_m: float, g_c: float) -> float:
    """计算 [t1, t2] 区间内时变基线的积分"""
    if dt_trading <= 0:
        return 0.0

    # 跨日：简化处理
    if t2_id <= t1_id:
        # 当日剩余
        i1 = _single_day_integral(t1_id, MARKET_CLOSE_PM, mu_base, g_o, g_m, g_c)
        # 次日开始
        i2 = _single_day_integral(MARKET_OPEN_AM, t2_id, mu_base, g_o, g_m, g_c)
        tt1 = intraday_to_trading_time(MARKET_CLOSE_PM) - intraday_to_trading_time(t1_id)
        tt2 = intraday_to_trading_time(t2_id) - intraday_to_trading_time(MARKET_OPEN_AM)
        remaining = dt_trading - tt1 - tt2
        if remaining > TRADING_SECONDS_PER_DAY * 0.5:
            n_full = int(remaining / TRADING_SECONDS_PER_DAY)
            avg = (mu_base * math.exp(g_o) * 1800 +
                   mu_base * math.exp(g_m) * 1800 +
                   mu_base * math.exp(g_c) * 1800 +
                   mu_base * 9000)
            return i1 + i2 + n_full * avg
        return i1 + i2

    return _single_day_integral(t1_id, t2_id, mu_base, g_o, g_m, g_c)


def _single_day_integral(t1: float, t2: float,
                         mu_base: float,
                         g_o: float, g_m: float, g_c: float) -> float:
    if t1 >= t2:
        return 0.0
    bounds = _boundaries_between(t1, t2)
    pts = [t1] + bounds + [t2]
    total = 0.0
    for i in range(len(pts) - 1):
        s, e = pts[i], pts[i + 1]
        tt_s = intraday_to_trading_time(s)
        tt_e = intraday_to_trading_time(e)
        dt = tt_e - tt_s
        if dt <= 0:
            continue
        mid = (s + e) / 2.0
        I_o, I_m, I_c = compute_indicators(mid)
        mu_t = mu_at_time(mu_base, g_o, g_m, g_c, I_o, I_m, I_c)
        total += mu_t * dt
    return total


def compute_gof_residuals(events_4d: List[np.ndarray],
                          T: float,
                          mu: np.ndarray,
                          alpha: np.ndarray,
                          omega: float,
                          dim: int = 4,
                          events_4d_original: Optional[List[np.ndarray]] = None,
                          gamma_open: Optional[np.ndarray] = None,
                          gamma_mid: Optional[np.ndarray] = None,
                          gamma_close: Optional[np.ndarray] = None,
                          gamma_spread: Optional[np.ndarray] = None,
                          spread_times: Optional[np.ndarray] = None,
                          spread_values: Optional[np.ndarray] = None,
                          spread_proc: Optional['SpreadProcess'] = None,
                          model: str = "A",
                          ) -> Dict:
    """
    统一 GOF 检验 (log-link)：time-rescaling 残差 + 多指标评估。

    **连续递推，不逐日重置 r**，与 EM/LL/fit_4d 口径完全一致。

    log-link 强度函数：
      λ_d(t) = μ_d · exp(η_d(t)) + Σ_j α_{dj}·ω·r_j(t)
      η_d(t) = γ_o·I_o + γ_m·I_m + γ_c·I_c + γ_s·x_s(t)
    """
    use_tv = (model in ("B", "C") and
              events_4d_original is not None and
              gamma_open is not None)

    if gamma_open is None:
        gamma_open = np.zeros(dim)
    if gamma_mid is None:
        gamma_mid = np.zeros(dim)
    if gamma_close is None:
        gamma_close = np.zeros(dim)
    if gamma_spread is None:
        gamma_spread = np.zeros(dim)

    mu_corrected = mu.copy()

    use_spread = (model == "C" and spread_proc is not None)

    # 合并时间线（连续，不分日）
    merged = []
    for d in range(dim):
        for idx, t in enumerate(events_4d[d]):
            t_orig = events_4d_original[d][idx] if (use_tv and idx < len(events_4d_original[d])) else t
            merged.append((float(t), d, float(t_orig)))
    merged.sort(key=lambda x: x[0])

    if len(merged) == 0:
        return {"error": "no_events"}

    all_times_arr = np.array([m[0] for m in merged], dtype=np.float64)
    all_types_arr = np.array([m[1] for m in merged], dtype=np.intc)
    all_intra_arr = np.array([m[2] for m in merged], dtype=np.float64)

    # 纯 Python 连续递推（log-link）
    all_residuals = _gof_residuals_loglink(
        all_times_arr, all_types_arr, all_intra_arr,
        dim, mu_corrected, alpha, omega,
        gamma_open, gamma_mid, gamma_close,
        use_tv, use_spread, gamma_spread, spread_proc)

    # --- GOF 指标计算 ---
    return _compute_gof_metrics(all_residuals, dim, use_tv,
                                gamma_open, gamma_mid, gamma_close,
                                mu, mu_corrected)


def _gof_residuals_python(
        all_times: np.ndarray, all_types: np.ndarray,
        all_intraday: np.ndarray,
        dim: int, mu_corrected: np.ndarray,
        alpha: np.ndarray, omega: float,
        gamma_open: np.ndarray, gamma_mid: np.ndarray,
        gamma_close: np.ndarray,
        use_tv: bool, use_spread: bool,
        gamma_spread: Optional[np.ndarray],
        spread_times: Optional[np.ndarray],
        spread_values: Optional[np.ndarray]) -> Dict[int, List[float]]:
    """
    纯 Python GOF 残差生成：连续递推，不逐日重置 r，与 EM/LL 口径一致。
    """
    N = len(all_times)
    all_residuals = {d: [] for d in range(dim)}

    if N == 0:
        return all_residuals

    r = np.zeros(dim)
    Lambda_accum = np.zeros(dim)
    first_seen = [False] * dim

    last_t = all_times[0]
    last_t_intra = all_intraday[0]

    for i in range(N):
        t = all_times[i]
        u_i = int(all_types[i])
        t_intra = all_intraday[i]

        dt = t - last_t
        if dt > 0:
            decay_f = math.exp(-omega * dt)

            for u in range(dim):
                if use_tv:
                    base_int = _compute_baseline_integral_segment(
                        last_t_intra, t_intra, dt,
                        mu_corrected[u], gamma_open[u], gamma_mid[u], gamma_close[u])
                else:
                    base_int = mu_corrected[u] * dt

                # 激励积分: Σ_j α[u,j]·r[j]·(1-exp(-ω·dt))
                exc_int = float(np.dot(alpha[u, :], r) * (1.0 - decay_f))

                # 外生项积分 (Model C)
                exog_int = 0.0
                if use_spread and gamma_spread is not None:
                    mid_t = (last_t + t) / 2.0
                    sp_val = _interp_spread(mid_t, spread_times, spread_values)
                    exog_int = gamma_spread[u] * max(sp_val, 0.0) * dt

                Lambda_accum[u] += base_int + exc_int + exog_int

            r *= decay_f

        # 记录残差
        if first_seen[u_i]:
            res_val = float(Lambda_accum[u_i])
            if res_val > 0:
                all_residuals[u_i].append(res_val)
        else:
            first_seen[u_i] = True

        Lambda_accum[u_i] = 0.0
        r[u_i] += 1.0
        last_t = t
        last_t_intra = t_intra

    return all_residuals


def _gof_residuals_loglink(
        all_times: np.ndarray, all_types: np.ndarray,
        all_intraday: np.ndarray,
        dim: int, mu: np.ndarray,
        alpha: np.ndarray, omega: float,
        gamma_open: np.ndarray, gamma_mid: np.ndarray,
        gamma_close: np.ndarray,
        use_tv: bool, use_spread: bool,
        gamma_spread: np.ndarray,
        spread_proc: Optional['SpreadProcess']) -> Dict[int, List[float]]:
    """
    log-link GOF 残差：连续递推，与 loglikelihood_loglink 完全一致。

    λ_d(t) = μ_d · exp(η_d(t)) + Σ_j α_{dj}·ω·r_j(t)
    Λ_d = ∫_{t_{k-1}}^{t_k} λ_d(s) ds
    """
    N = len(all_times)
    all_residuals = {d: [] for d in range(dim)}
    if N == 0:
        return all_residuals

    r = np.zeros(dim)
    Lambda_accum = np.zeros(dim)
    first_seen = [False] * dim

    last_t = all_times[0]

    for i in range(N):
        t = all_times[i]
        u_i = int(all_types[i])

        dt = t - last_t
        if dt > 0:
            decay_f = math.exp(-omega * dt)

            for u in range(dim):
                # 基线积分: ∫ μ_u · exp(η_u(s)) ds
                # 用中点近似（事件间隔通常很短）
                mid_t = (last_t + t) * 0.5
                day_idx = int(mid_t / TRADING_SECONDS_PER_DAY)
                t_in_day = mid_t - day_idx * TRADING_SECONDS_PER_DAY
                t_intraday_mid = _trading_time_to_intraday(t_in_day)
                eta = 0.0
                if use_tv:
                    I_o, I_m, I_c = compute_indicators(t_intraday_mid)
                    eta = (gamma_open[u] * I_o + gamma_mid[u] * I_m
                           + gamma_close[u] * I_c)
                if use_spread and spread_proc is not None:
                    eta += gamma_spread[u] * spread_proc.value_at(mid_t)
                base_int = mu[u] * math.exp(eta) * dt

                # 激励积分: Σ_j α[u,j]·r[j]·(1-exp(-ω·dt))
                exc_int = float(np.dot(alpha[u, :], r) * (1.0 - decay_f))

                Lambda_accum[u] += base_int + exc_int

            r *= decay_f

        # 记录残差
        if first_seen[u_i]:
            res_val = float(Lambda_accum[u_i])
            if res_val > 0:
                all_residuals[u_i].append(res_val)
        else:
            first_seen[u_i] = True

        Lambda_accum[u_i] = 0.0
        r[u_i] += 1.0
        last_t = t

    return all_residuals


def _interp_spread(t: float, sp_times: np.ndarray, sp_vals: np.ndarray) -> float:
    """线性插值 spread 值（向后兼容）"""
    if len(sp_times) == 0:
        return 0.0
    idx = np.searchsorted(sp_times, t, side='right') - 1
    idx = max(0, min(idx, len(sp_vals) - 1))
    return float(sp_vals[idx])


def _compute_gof_metrics(all_residuals: Dict[int, List[float]],
                         dim: int,
                         use_tv: bool,
                         gamma_open: np.ndarray,
                         gamma_mid: np.ndarray,
                         gamma_close: np.ndarray,
                         mu_original: np.ndarray,
                         mu_corrected: np.ndarray) -> Dict:
    """从残差计算全套 GOF 指标"""
    KS_SUBSAMPLE_N = 500
    KS_SUBSAMPLE_REPS = 20
    QQ_QUANTILE_N = 200
    ACF_MAX_LAG = 20

    results = {}
    raw_residuals = {}
    rng = np.random.RandomState(42)

    for d in range(dim):
        res = np.array(all_residuals[d], dtype=float)
        if len(res) <= 10:
            results[f"dim_{d}"] = {"n_residuals": len(res), "error": "insufficient_residuals"}
            continue

        res_mean = float(np.mean(res))
        res_std = float(np.std(res))
        res_median = float(np.median(res))

        # QQ
        probs = np.linspace(0.005, 0.995, QQ_QUANTILE_N)
        qq_emp = np.quantile(res, probs).tolist()
        qq_theo = expon.ppf(probs).tolist()

        # 距离
        mae_mean = float(abs(res_mean - 1.0))
        n_ws = min(5000, len(res))
        res_ws = res[rng.choice(len(res), n_ws, replace=False)] if len(res) > n_ws else res
        ref_exp = rng.exponential(1.0, size=n_ws)
        w1 = float(wasserstein_distance(res_ws, ref_exp))
        qq_mae = float(np.mean(np.abs(np.array(qq_emp) - np.array(qq_theo))))

        # 子采样 KS
        subsample_pvals = []
        n_sub = min(KS_SUBSAMPLE_N, len(res))
        for _ in range(KS_SUBSAMPLE_REPS):
            idx_s = rng.choice(len(res), size=n_sub, replace=False)
            _, pval = kstest(res[idx_s], 'expon', args=(0, 1))
            subsample_pvals.append(pval)
        ks_pval_median = float(np.median(subsample_pvals))

        # Ljung-Box + ACF
        lb_pvalues = []
        acf_values = []
        lb_pass = None
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
            pass

        # 综合评分
        mean_score = max(0.0, 1.0 - mae_mean)
        w1_score = max(0.0, 1.0 - w1 / 2.0)
        lb_score = float(np.mean(lb_pvalues)) if lb_pvalues else 0.5
        acf_score = 1.0 - float(np.mean(np.abs(acf_values))) if acf_values else 0.5
        gof_score = 0.3 * mean_score + 0.3 * w1_score + 0.2 * lb_score + 0.2 * acf_score

        mean_ok = bool(0.8 <= res_mean <= 1.2)
        gof_pass = bool(mean_ok and (lb_pass is True or lb_pass is None))

        results[f"dim_{d}"] = {
            "n_residuals": len(res),
            "mean": res_mean, "std": res_std, "median": res_median,
            "qq_empirical": qq_emp, "qq_theoretical": qq_theo,
            "mae_mean": mae_mean, "wasserstein_1": w1, "qq_mae": qq_mae,
            "ks_pvalue_sub_median": ks_pval_median,
            "ljung_box_pvalues": lb_pvalues, "ljung_box_pass": lb_pass,
            "acf_values": acf_values,
            "gof_score": float(gof_score), "mean_ok": mean_ok, "gof_pass": gof_pass,
        }
        n_save = min(2000, len(res))
        raw_residuals[f"dim_{d}"] = res[rng.choice(len(res), n_save, replace=False)].tolist()

    # 汇总
    gof_scores = [results.get(f"dim_{d}", {}).get("gof_score", 0.0) for d in range(dim)]
    gof_pass_count = sum(1 for d in range(dim) if results.get(f"dim_{d}", {}).get("gof_pass", False))
    model_type = "time_varying_mu" if use_tv else "constant_mu"
    results["summary"] = {
        "gof_pass_count": gof_pass_count,
        "all_pass": gof_pass_count == dim,
        "gof_score_mean": float(np.mean(gof_scores)),
        "gof_scores": gof_scores,
        "model_type": model_type,
    }
    results["raw_residuals"] = raw_residuals

    if use_tv:
        results["gamma"] = {
            "gamma_open": gamma_open.tolist(),
            "gamma_mid": gamma_mid.tolist(),
            "gamma_close": gamma_close.tolist(),
        }
    results["mu_info"] = {
        "mu_original": mu_original.tolist(),
        "mu_corrected": mu_corrected.tolist(),
    }

    return results


# ===================== 1D 便捷接口 =====================

def fit_1d(times: np.ndarray, beta_grid: np.ndarray,
           T: float = -1.0, maxiter: int = 80,
           verbose: bool = True) -> Dict:
    """
    一维 Hawkes 拟合便捷接口。

    Parameters
    ----------
    times : (N,) 事件时间
    beta_grid : 候选 β 值
    T : 观测窗口
    """
    if T < 0:
        T = float(times[-1])
    seq = np.column_stack([times, np.zeros(len(times))])
    best_beta, best_alpha, best_mu, best_ll, _ = grid_search_beta(
        seq, 1, beta_grid, Tm=T, maxiter=maxiter, verbose=verbose)

    alpha_val = float(best_alpha[0, 0])
    mu_val = float(best_mu[0])
    br = alpha_val  # 1D branching ratio = alpha / 1 (since integral of kernel = alpha)

    # GOF
    events_1d = [times]
    gof = compute_gof_residuals(events_1d, T,
                                best_mu, best_alpha, best_beta, dim=1)

    return {
        "mu": mu_val,
        "alpha": alpha_val,
        "beta": best_beta,
        "branching_ratio": br,
        "loglik": best_ll,
        "gof": gof,
    }


# ===================== 4D 完整拟合流程 =====================

def fit_4d(events_4d: List[np.ndarray],
           T: float,
           beta_grid: np.ndarray,
           model: str = "A",
           events_4d_original: Optional[List[np.ndarray]] = None,
           spread_times: Optional[np.ndarray] = None,
           spread_values: Optional[np.ndarray] = None,
           spread_proc: Optional['SpreadProcess'] = None,
           n_days: int = 22,
           maxiter: int = 80,
           n_alt: int = 2,
           verbose: bool = True) -> Dict:
    """
    4D Hawkes 拟合主入口 (Kramer log-link 交替优化)。

    交替优化流程：
      Round 0: EM(mu, alpha | gamma=0) → 选最优 β
      Round 1..n_alt: L-BFGS(mu, gamma | alpha, omega fixed)
                      → EM(mu, alpha | gamma fixed) → 更新 LL
    最终 LL/AIC/BIC 均用 loglikelihood_loglink 统一计算。

    强度函数：
      λ_d(t) = μ_d · exp(η_d(t)) + Σ_j α_{dj}·ω·r_j(t)
      Model A: η=0
      Model B: η = γ_o·I_o + γ_m·I_m + γ_c·I_c
      Model C: η = γ_o·I_o + γ_m·I_m + γ_c·I_c + γ_s·x_s(t)

    Parameters
    ----------
    events_4d : 4 个 np.ndarray，归一化后的事件时间
    T : 总时长
    beta_grid : β 候选值（三个模型使用统一网格）
    model : "A" / "B" / "C"
    events_4d_original : 日内时间 (Model B/C 需要)
    spread_times, spread_values : 原始 re_spread 数据（向后兼容）
    spread_proc : SpreadProcess 实例 (Model C, 推荐使用)
    n_days : 交易日数
    n_alt : 交替优化轮数 (Model B/C)
    """
    dim = 4
    # --- 构建 seq ---
    merged = []
    merged_intra = []
    for d in range(dim):
        for idx, t in enumerate(events_4d[d]):
            merged.append([float(t), float(d)])
            if events_4d_original is not None and idx < len(events_4d_original[d]):
                merged_intra.append(float(events_4d_original[d][idx]))
            else:
                merged_intra.append(float(t))
    sort_idx = sorted(range(len(merged)), key=lambda i: merged[i][0])
    merged = [merged[i] for i in sort_idx]
    merged_intra = [merged_intra[i] for i in sort_idx]
    seq = np.array(merged)
    intraday_arr = np.array(merged_intra, dtype=np.float64)

    if len(seq) < 10:
        return {"error": "insufficient_events"}

    # 向后兼容: 如果没有 spread_proc 但给了 spread_times/values，自动构建
    if model == "C" and spread_proc is None and spread_times is not None:
        spread_proc = SpreadProcess(spread_times, spread_values,
                                    method='previous', lag=0.0, standardize=True)

    if verbose:
        print(f"  4D fit (Model {model}): {len(seq)} events, T={T:.1f}, days={n_days}")
        print(f"  β grid: {beta_grid}")
        if spread_proc is not None:
            print(f"  SpreadProcess: {spread_proc}")

    # ===== Model A: 简单 EM =====
    if model == "A":
        best_beta, best_alpha, best_mu, _, _ = grid_search_beta(
            seq, dim, beta_grid, Tm=T, maxiter=maxiter, verbose=verbose,
            n_days=n_days)
        gamma_open = gamma_mid = gamma_close = gamma_spread = np.zeros(dim)
        best_ll = loglikelihood_loglink(
            seq, dim, best_mu, best_alpha, best_beta, T,
            n_days=n_days, model="A")

    else:
        # ===== Model B / C: 交替优化 =====
        # --- Round 0: 用事件率初始化 gamma, EM 选 β ---
        gamma_open_init, gamma_mid_init, gamma_close_init = (
            estimate_gamma_from_events(events_4d_original, dim)
            if events_4d_original is not None
            else (np.zeros(dim), np.zeros(dim), np.zeros(dim)))
        if verbose:
            print(f"  [init] γ_open={gamma_open_init.round(3)}, "
                  f"γ_mid={gamma_mid_init.round(3)}, γ_close={gamma_close_init.round(3)}")

        # EM with initial gamma (baseline as multiplier, old-style compatible)
        bl_int_T = _compute_baseline_integral_T(
            dim, T, n_days, gamma_open_init, gamma_mid_init, gamma_close_init)
        best_beta, best_alpha, best_mu, _, _ = grid_search_beta(
            seq, dim, beta_grid, Tm=T, maxiter=maxiter, verbose=verbose,
            intraday_times=intraday_arr,
            gamma_open=gamma_open_init, gamma_mid=gamma_mid_init,
            gamma_close=gamma_close_init,
            n_days=n_days)
        if verbose:
            print(f"  [Round 0] β={best_beta:.2f}, μ_em={best_mu.round(4)}")

        gamma_open = gamma_open_init.copy()
        gamma_mid = gamma_mid_init.copy()
        gamma_close = gamma_close_init.copy()
        gamma_spread = np.zeros(dim)

        # --- Alternating rounds ---
        for rnd in range(1, n_alt + 1):
            if verbose:
                print(f"\n  === Alternating round {rnd}/{n_alt} ===")

            # (a) L-BFGS: optimize (mu, gamma) given (alpha, omega)
            mu_opt, gamma_open, gamma_mid, gamma_close, gamma_spread, ll_bfgs = (
                _optimize_baseline_params(
                    seq, dim, best_alpha, best_beta, T, n_days, model,
                    spread_proc=spread_proc if model == "C" else None,
                    gamma_open_init=gamma_open, gamma_mid_init=gamma_mid,
                    gamma_close_init=gamma_close,
                    mu_init=best_mu,
                    verbose=verbose))

            # (b) EM: re-estimate (mu, alpha) given gamma
            bl_int_T = _compute_loglink_integrals(
                dim, T, n_days,
                gamma_open if model in ("B", "C") else None,
                gamma_mid if model in ("B", "C") else None,
                gamma_close if model in ("B", "C") else None,
                gamma_spread if model == "C" else None,
                spread_proc if model == "C" else None)[0]  # eff_T only

            # EM with log-link baseline integral
            best_alpha, best_mu, _ = em_estimate(
                seq, dim, best_beta, Tm=T, maxiter=maxiter, verbose=False,
                intraday_times=intraday_arr,
                gamma_open=gamma_open, gamma_mid=gamma_mid,
                gamma_close=gamma_close,
                n_days=n_days)
            # Override mu with L-BFGS mu (EM mu is baseline-corrected differently)
            best_mu = mu_opt

            best_ll = loglikelihood_loglink(
                seq, dim, best_mu, best_alpha, best_beta, T,
                n_days=n_days,
                gamma_open=gamma_open, gamma_mid=gamma_mid,
                gamma_close=gamma_close,
                gamma_spread=gamma_spread if model == "C" else None,
                spread_proc=spread_proc if model == "C" else None,
                model=model)
            if verbose:
                print(f"  [Round {rnd}] LL={best_ll:.2f}")

    # === 统一 LL / AIC / BIC ===
    eigvals = np.linalg.eigvals(best_alpha)
    br = float(np.max(np.abs(eigvals)))

    N_events = len(seq)
    k_params = dim + dim * dim + 1  # mu + alpha + omega
    if model in ("B", "C"):
        k_params += 3 * dim  # gamma_open/mid/close
    if model == "C":
        k_params += dim  # gamma_spread

    aic = 2 * k_params - 2 * best_ll
    bic = k_params * math.log(max(N_events, 1)) - 2 * best_ll

    if verbose:
        print(f"\n  Best β={best_beta:.2f}, LL={best_ll:.2f}, AIC={aic:.2f}, "
              f"BIC={bic:.2f}, BR={br:.4f}")
        print(f"  μ = {best_mu.round(4)}")
        if model in ("B", "C"):
            print(f"  γ_open  = {gamma_open.round(3)}")
            print(f"  γ_mid   = {gamma_mid.round(3)}")
            print(f"  γ_close = {gamma_close.round(3)}")
        if model == "C":
            print(f"  γ_spread = {gamma_spread.round(4)}")
            print(f"  exp(γ_spread) = {np.exp(gamma_spread).round(4)}")

    # --- GOF（使用完全相同的 log-link 强度函数）---
    gof = compute_gof_residuals(
        events_4d, T, best_mu, best_alpha, best_beta, dim=dim,
        events_4d_original=events_4d_original if model in ("B", "C") else None,
        gamma_open=gamma_open if model in ("B", "C") else None,
        gamma_mid=gamma_mid if model in ("B", "C") else None,
        gamma_close=gamma_close if model in ("B", "C") else None,
        gamma_spread=gamma_spread if model == "C" else None,
        spread_proc=spread_proc if model == "C" else None,
        model=model)

    result = {
        "model": model,
        "full": {
            "decay": best_beta,
            "mu": best_mu.tolist(),
            "A": best_alpha.tolist(),
            "loglik": best_ll,
            "aic": float(aic),
            "bic": float(bic),
            "branching_ratio": br,
            "constraint_ok": bool(br < 1.0),
            "k_params": k_params,
            "n_events": N_events,
        },
        "gof": gof,
    }

    if model in ("B", "C"):
        result["gamma"] = {
            "gamma_open": gamma_open.tolist(),
            "gamma_mid": gamma_mid.tolist(),
            "gamma_close": gamma_close.tolist(),
        }
    if model == "C":
        result["gamma_spread"] = gamma_spread.tolist()
        result["gamma_spread_exp"] = np.exp(gamma_spread).tolist()
        if spread_proc is not None:
            result["spread_info"] = {
                "n_points": spread_proc.n_points,
                "mean_raw": float(spread_proc.mean_),
                "std_raw": float(spread_proc.std_),
                "lag": float(spread_proc.lag),
            }

    return result


def _compute_exog_integral(dim: int, gamma_spread: np.ndarray,
                           spread_times: np.ndarray, spread_values: np.ndarray,
                           T: float) -> float:
    """
    计算外生项的积分：Σ_d ∫_0^T γ_spread_d · max(re_spread(t), 0) dt。
    使用分段常数近似（左端点插值）。
    """
    if gamma_spread is None or spread_times is None or len(spread_times) == 0:
        return 0.0

    # 分段常数积分
    total = 0.0
    for i in range(len(spread_times)):
        t_start = spread_times[i]
        t_end = spread_times[i + 1] if i + 1 < len(spread_times) else T
        if t_end > T:
            t_end = T
        dt = t_end - t_start
        if dt <= 0:
            continue
        sp_val = max(spread_values[i], 0.0)
        for d in range(dim):
            total += gamma_spread[d] * sp_val * dt
    return total


# ===================== 入口测试 =====================

if __name__ == "__main__":
    print("=== hawkes_em.py 自测 ===")
    # 1D 仿真测试
    mu_true = np.array([0.5])
    alpha_true = np.array([[0.3]])
    omega_true = 1.0

    print("模拟 1D Hawkes: μ=0.5, α=0.3, ω=1.0, T=1000")
    data = simulate_hawkes_multi(mu_true, alpha_true, omega_true, T=1000, seed=42)
    print(f"  生成 {len(data)} 个事件")

    seq = data
    beta_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    best_beta, best_alpha, best_mu, best_ll, _ = grid_search_beta(
        seq, 1, beta_grid, verbose=True)
    print(f"\n估计结果: β={best_beta:.2f}, μ={best_mu[0]:.4f}, α={best_alpha[0,0]:.4f}")
    print(f"真实参数: β=1.00, μ=0.5000, α=0.3000")
    print("=== 自测完成 ===")
