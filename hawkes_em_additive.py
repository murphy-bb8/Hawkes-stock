"""
加性基线 Hawkes EM 实现 — 递推 O(N·d²) 版本
=============================================
参考 MHP_fixed_final_play.py 的 EM 框架，改为递推计算以支持大规模数据。

Model A: λ_i(t) = μ_i + Σ_j α_{ij}·ω·exp(-ω·Δt)
Model B: λ_i(t) = μ_{i,period(t)} + Σ_j α_{ij}·ω·exp(-ω·Δt)
         其中 period ∈ {normal, open30, mid30, close30}，分段常数基线
Model C: Model B + γ_{spread,i}·x⁺(t)

核函数: φ_{ij}(Δt) = α_{ij}·ω·exp(-ω·Δt)
积分:   ∫₀^∞ φ_{ij}(s)ds = α_{ij}

EM M步全部闭式解:
  μ_{i,p} = Σ_{n:u_n=i,per(n)=p} p_{n,bg} / T_p
  α_{ij}  = Σ_{n:u_n=i} (α_{ij}·R_j[n]/λ_i) / N_j
  γ_{spread,i} = Σ_{n:u_n=i} p_{n,spread} / X_total
"""

import numpy as np
import json
import os
import time as _time
from typing import List, Tuple, Optional, Dict, Any
from scipy.stats import kstest, wasserstein_distance

# Cython 加速模块（可选，回退到纯 Python）
try:
    from _hawkes_cy import precompute_R_cy, gof_residuals_cy, compute_ll_cy
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False

# ===================== A股交易时间常量 =====================
MARKET_OPEN_AM = 34200    # 09:30
MARKET_CLOSE_AM = 41400   # 11:30
MARKET_OPEN_PM = 46800    # 13:00
MARKET_CLOSE_PM = 54000   # 15:00

OPEN30_START = 34200;  OPEN30_END = 36000    # 09:30–10:00
MID30_START = 46800;   MID30_END = 48600     # 13:00–13:30
CLOSE30_START = 52200; CLOSE30_END = 54000   # 14:30–15:00

TRADING_SECONDS_PER_DAY = 14400  # 4h

# 交易时间内的时段边界 (相对于每天0–14400)
_OPEN_END_TT = 1800       # 开盘30分钟结束
_MID_START_TT = 7200      # 午盘开始
_MID_END_TT = 9000        # 午盘30分钟结束
_CLOSE_START_TT = 12600   # 收盘30分钟开始

PERIOD_OPEN = 0
PERIOD_MID = 1
PERIOD_CLOSE = 2
PERIOD_NORMAL = 3
N_PERIODS = 4

# 每天各时段秒数
PERIOD_SECS_PER_DAY = np.array([1800., 1800., 1800., 9000.])


# ===================== 工具函数 =====================

def get_period_tt(t_day: float) -> int:
    """从日内交易时间(0–14400)获取时段编号"""
    if t_day < _OPEN_END_TT:
        return PERIOD_OPEN
    if _MID_START_TT <= t_day < _MID_END_TT:
        return PERIOD_MID
    if t_day >= _CLOSE_START_TT:
        return PERIOD_CLOSE
    return PERIOD_NORMAL


def intraday_to_trading_time(t_intraday: float) -> float:
    """钟表时间(秒) → 日内交易时间(0–14400)"""
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


def trading_time_to_clock(t_day: float) -> float:
    """日内交易时间(0–14400) → 钟表时间(34200–54000)"""
    if t_day < 7200:
        return MARKET_OPEN_AM + t_day
    return MARKET_OPEN_PM + (t_day - 7200)


def compute_indicators(t_clock: float) -> Tuple[float, float, float]:
    """从钟表时间计算 (I_open, I_mid, I_close)"""
    I_o = 1.0 if OPEN30_START <= t_clock < OPEN30_END else 0.0
    I_m = 1.0 if MID30_START <= t_clock < MID30_END else 0.0
    I_c = 1.0 if CLOSE30_START <= t_clock < CLOSE30_END else 0.0
    return I_o, I_m, I_c


def flatten_events(events_list, intraday_list=None, spread_list=None):
    """
    将多维事件列表展平为按时间排序的数组。

    Returns: (times, types, intraday_arr, spread_arr)
    """
    dim = len(events_list)
    N_total = sum(len(ev) for ev in events_list)

    times = np.empty(N_total)
    types = np.empty(N_total, dtype=int)
    has_intra = intraday_list is not None
    has_spread = spread_list is not None
    intra = np.empty(N_total) if has_intra else None
    spread = np.empty(N_total) if has_spread else None

    idx = 0
    for d in range(dim):
        n = len(events_list[d])
        if n == 0:
            continue
        times[idx:idx + n] = events_list[d]
        types[idx:idx + n] = d
        if has_intra:
            intra[idx:idx + n] = intraday_list[d]
        if has_spread:
            spread[idx:idx + n] = spread_list[d]
        idx += n

    times = times[:idx]
    types = types[:idx]
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    types = types[sort_idx].astype(int)
    if has_intra:
        intra = intra[:idx][sort_idx]
    if has_spread:
        spread = spread[:idx][sort_idx]
    return times, types, intra, spread


_flatten_events = flatten_events  # alias for backward compat


# ===================== 递推 EM 算法 (核心) =====================

def _precompute_R(times: np.ndarray, types: np.ndarray, dim: int, omega: float) -> np.ndarray:
    """
    预计算 R_all[n, j] = Σ_{k<n, u_k=j} ω·exp(-ω·(t_n - t_k))
    只依赖 data 和 omega，与 alpha/mu 无关，只需对每个 omega 算一次。
    """
    if _USE_CYTHON:
        return precompute_R_cy(
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(types, dtype=np.int64),
            dim, float(omega))

    N = len(times)
    R_all = np.zeros((N, dim))

    dt = np.empty(N)
    dt[0] = 0.0
    dt[1:] = times[1:] - times[:-1]
    decay = np.exp(-omega * dt)

    type_add = np.zeros((N, dim))
    type_add[np.arange(N), types] = omega

    R_cur = np.zeros(dim)
    for n in range(N):
        R_cur *= decay[n]
        R_all[n] = R_cur
        R_cur += type_add[n]
    return R_all


def em_hawkes_recursive(
        times: np.ndarray, types: np.ndarray, dim: int, omega: float,
        mu_init, alpha_init: np.ndarray,
        T: float, n_days: int = 1,
        model: str = "A",
        periods: Optional[np.ndarray] = None,
        spread_shifted: Optional[np.ndarray] = None,
        X_total: float = 1.0,
        maxiter: int = 100, epsilon: float = 1e-4, verbose: bool = False,
        R_all: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    向量化递推 EM。R_all 预计算后，每次迭代全部 numpy 向量化。

    Parameters
    ----------
    mu_init : Model A → (dim,); Model B/C → (dim, 4)
    periods : (N,) int, 时段编号, Model B/C 需要
    spread_shifted : (N,) ≥0 的 spread 值, Model C 需要
    X_total : ∫₀ᵀ x⁺(t)dt 的近似
    R_all : 预计算的 (N, dim), 若 None 则自动计算
    """
    N = len(times)
    if N == 0:
        return {"mu": mu_init, "alpha": alpha_init, "loglik": -np.inf, "n_iter": 0}

    T_per = PERIOD_SECS_PER_DAY * n_days  # (4,)
    N_type = np.array([np.sum(types == d) for d in range(dim)])

    # 预计算 R_all (只做一次)
    if R_all is None:
        R_all = _precompute_R(times, types, dim, omega)

    # 补偿项
    comp = np.zeros(dim)
    for d in range(dim):
        mask_d = types == d
        comp[d] = np.sum(1.0 - np.exp(-omega * (T - times[mask_d])))

    # 预计算事件类型 mask 和索引
    type_masks = [(types == d) for d in range(dim)]
    type_idx = types  # (N,) int

    # 初始化参数
    alpha = alpha_init.copy().astype(float)
    if model == "A":
        mu = mu_init.copy().ravel().astype(float)
    else:
        mu_p = mu_init.copy().astype(float)
        if mu_p.ndim == 1:
            mu_p = np.tile(mu_p.reshape(-1, 1), (1, N_PERIODS))
    if model == "C":
        init_base = N_type / T + 0.01
        gamma_sp = init_base * 0.1
    else:
        gamma_sp = None

    # 预计算 period masks (Model B/C)
    period_masks = None
    if model in ("B", "C") and periods is not None:
        period_masks = {}
        for i in range(dim):
            for p in range(N_PERIODS):
                period_masks[(i, p)] = type_masks[i] & (periods == p)

    old_ll = -1e30

    for it in range(maxiter):
        # === E步 + M步 (全向量化) ===

        # 1. 计算基线 bases[n]
        if model == "A":
            bases = mu[type_idx]  # (N,)
        else:
            bases = mu_p[type_idx, periods]  # (N,)

        # 2. 激励 exc[n] = Σ_j α[type_n, j] * R_all[n, j]
        exc = np.sum(alpha[type_idx, :] * R_all, axis=1)  # (N,)

        # 3. Spread
        sp = np.zeros(N)
        if model == "C" and gamma_sp is not None and spread_shifted is not None:
            sp = gamma_sp[type_idx] * spread_shifted  # (N,)

        # 4. 强度
        lam = bases + exc + sp
        lam = np.maximum(lam, 1e-15)
        inv_lam = 1.0 / lam

        # 5. LL 求和
        ll_sum = np.sum(np.log(lam))

        # === M步 累积 ===
        bg_w = bases * inv_lam  # p_{n,base} (N,)

        # 更新 mu
        if model == "A":
            new_mu = np.zeros(dim)
            for i in range(dim):
                new_mu[i] = max(np.sum(bg_w[type_masks[i]]) / T, 1e-10)
            mu = new_mu
        else:
            new_mu_p = np.full((dim, N_PERIODS), 1e-10)
            for i in range(dim):
                for p in range(N_PERIODS):
                    if T_per[p] > 0:
                        s = np.sum(bg_w[period_masks[(i, p)]])
                        new_mu_p[i, p] = max(s / T_per[p], 1e-10)
            mu_p = new_mu_p

        # 更新 gamma_spread
        if model == "C" and gamma_sp is not None and X_total > 0:
            sp_w = sp * inv_lam  # (N,)
            new_gs = np.zeros(dim)
            for i in range(dim):
                new_gs[i] = max(np.sum(sp_w[type_masks[i]]) / X_total, 0.0)
            gamma_sp = new_gs

        # 更新 alpha
        # sum_alpha_num[i,j] = Σ_{n: u_n=i} α[i,j] * R_all[n,j] / λ[n]
        new_alpha = np.zeros((dim, dim))
        for i in range(dim):
            m_i = type_masks[i]
            R_i = R_all[m_i, :]  # (N_i, dim)
            w_i = inv_lam[m_i]   # (N_i,)
            for j in range(dim):
                if N_type[j] > 0:
                    new_alpha[i, j] = max(
                        alpha[i, j] * np.dot(R_i[:, j], w_i) / N_type[j], 0.0)
        alpha = new_alpha

        # === LL ===
        if model == "A":
            int_base = np.sum(mu) * T
        else:
            int_base = np.sum(mu_p * T_per[np.newaxis, :])
        int_sp = np.sum(gamma_sp) * X_total if model == "C" else 0.0
        int_exc = np.sum(alpha * comp[np.newaxis, :])
        ll = ll_sum - int_base - int_sp - int_exc

        if verbose and it % 10 == 0:
            print(f"  EM iter {it}: LL={ll:.4f}")
        if it > 0 and abs(ll - old_ll) < epsilon:
            if verbose:
                print(f"  EM converged iter {it}: LL={ll:.4f}")
            break
        old_ll = ll

    # --- 组装结果 ---
    res = {"alpha": alpha.copy(), "loglik": float(ll), "n_iter": it + 1}
    if model == "A":
        res["mu"] = mu.copy()
    else:
        mu_normal = mu_p[:, PERIOD_NORMAL].copy()
        res["mu"] = mu_normal
        res["mu_periods"] = mu_p.copy()
        res["gamma_open"] = mu_p[:, PERIOD_OPEN] - mu_normal
        res["gamma_mid"] = mu_p[:, PERIOD_MID] - mu_normal
        res["gamma_close"] = mu_p[:, PERIOD_CLOSE] - mu_normal
    if model == "C":
        res["gamma_spread"] = gamma_sp.copy()
    return res


# ===================== 对数似然 (独立验证) =====================

def compute_loglikelihood(
        times, types, dim, omega, alpha, T, n_days=1,
        model="A", mu=None, mu_periods=None,
        periods=None, gamma_spread=None, spread_shifted=None, X_total=1.0,
) -> float:
    """递推计算 LL, 用于独立验证。"""
    N = len(times)
    model_code = {"A": 0, "B": 1, "C": 2}[model]

    if _USE_CYTHON:
        _mu = np.ascontiguousarray(mu if mu is not None else np.zeros(dim), dtype=np.float64)
        _mu_p = np.ascontiguousarray(mu_periods if mu_periods is not None else np.zeros((dim, N_PERIODS)), dtype=np.float64)
        _per = np.ascontiguousarray(periods if periods is not None else np.zeros(N, dtype=np.int64), dtype=np.int64)
        _gs = np.ascontiguousarray(gamma_spread if gamma_spread is not None else np.zeros(dim), dtype=np.float64)
        _ss = np.ascontiguousarray(spread_shifted if spread_shifted is not None else np.zeros(N), dtype=np.float64)
        ll_sum = compute_ll_cy(
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(types, dtype=np.int64),
            dim, float(omega),
            np.ascontiguousarray(alpha, dtype=np.float64),
            model_code, _mu, _mu_p, _per, _gs, _ss)
    else:
        R = np.zeros(dim)
        ll_sum = 0.0
        last_t = 0.0
        for n in range(N):
            dt = times[n] - last_t
            i = types[n]
            if dt > 0:
                R *= np.exp(-omega * dt)
            if model == "A":
                lam = mu[i]
            else:
                lam = mu_periods[i, periods[n]]
            lam += np.dot(alpha[i, :], R)
            if model == "C" and gamma_spread is not None and spread_shifted is not None:
                lam += gamma_spread[i] * spread_shifted[n]
            lam = max(lam, 1e-15)
            ll_sum += np.log(lam)
            R[i] += omega
            last_t = times[n]

    T_per = PERIOD_SECS_PER_DAY * n_days
    if model == "A":
        int_base = np.sum(mu) * T
    else:
        int_base = np.sum(mu_periods * T_per[np.newaxis, :])
    int_sp = np.sum(gamma_spread) * X_total if (model == "C" and gamma_spread is not None) else 0.0
    comp = np.zeros(dim)
    for n in range(N):
        comp[types[n]] += 1.0 - np.exp(-omega * (T - times[n]))
    int_exc = np.sum(alpha * comp[np.newaxis, :])
    return float(ll_sum - int_base - int_sp - int_exc)


# ===================== GOF 残差 =====================

def gof_residuals(
        times, types, dim, omega, alpha, T, n_days=1,
        model="A", mu=None, mu_periods=None, periods=None,
        gamma_spread=None, spread_shifted=None,
) -> Dict[str, Any]:
    """
    时间重标度残差: 若模型正确, τ_k ~ Exp(1)。
    使用精确激励积分。
    """
    N = len(times)
    model_code = {"A": 0, "B": 1, "C": 2}[model]

    if _USE_CYTHON:
        _mu = np.ascontiguousarray(mu if mu is not None else np.zeros(dim), dtype=np.float64)
        _mu_p = np.ascontiguousarray(mu_periods if mu_periods is not None else np.zeros((dim, N_PERIODS)), dtype=np.float64)
        _per = np.ascontiguousarray(periods if periods is not None else np.zeros(N, dtype=np.int64), dtype=np.int64)
        _gs = np.ascontiguousarray(gamma_spread if gamma_spread is not None else np.zeros(dim), dtype=np.float64)
        _ss = np.ascontiguousarray(spread_shifted if spread_shifted is not None else np.zeros(N), dtype=np.float64)
        residuals_lists = gof_residuals_cy(
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(types, dtype=np.int64),
            dim, float(omega),
            np.ascontiguousarray(alpha, dtype=np.float64),
            T, model_code, _mu, _mu_p, _per, _gs, _ss)
        residuals = {d: residuals_lists[d] for d in range(dim)}
    else:
        residuals = {d: [] for d in range(dim)}
        R = np.zeros(dim)
        Lambda_run = np.zeros(dim)
        seen = np.zeros(dim, dtype=bool)
        last_t = 0.0

        for k in range(N):
            dt = times[k] - last_t
            i = types[k]
            if dt > 0:
                decay = np.exp(-omega * dt)
                for d in range(dim):
                    if model == "A":
                        b = mu[d]
                    else:
                        b = mu_periods[d, periods[k]]
                    base_int = b * dt
                    exc_int = np.dot(alpha[d, :], R) * (1.0 - decay) / omega if omega > 0 else 0.0
                    sp_int = 0.0
                    if model == "C" and gamma_spread is not None and spread_shifted is not None:
                        sp_int = gamma_spread[d] * spread_shifted[k] * dt
                    Lambda_run[d] += base_int + exc_int + sp_int
                R *= decay

            if seen[i] and Lambda_run[i] > 0:
                residuals[i].append(Lambda_run[i])
            seen[i] = True
            Lambda_run[i] = 0.0
            R[i] += omega
            last_t = times[k]

    results = {}
    np.random.seed(0)
    for d in range(dim):
        arr = np.array(residuals[d])
        if len(arr) > 20:
            ks_stat, ks_pval = kstest(arr, "expon", args=(0, 1))
            m = float(np.mean(arr))
            n_sub = min(len(arr), 5000)
            w1 = float(wasserstein_distance(arr[:n_sub], np.random.exponential(1.0, n_sub)))
            score_mean = max(0.0, 1.0 - abs(m - 1.0))
            score_w1 = max(0.0, 1.0 - w1)
            score_lb = 1.0 if ks_pval > 0.05 else 0.0
            gof_score = 0.4 * score_mean + 0.4 * score_w1 + 0.2 * score_lb
            results[d] = {
                "n": len(arr), "mean": m, "mean_dev": abs(m - 1.0),
                "ks_stat": float(ks_stat), "ks_pval": float(ks_pval),
                "w1": w1, "gof_score": gof_score,
                "gof_pass": bool(ks_pval > 0.05),
            }
        else:
            results[d] = {"n": len(arr), "error": "insufficient"}

    n_pass = sum(1 for d in range(dim) if results.get(d, {}).get("gof_pass", False))
    mean_score = np.mean([results[d]["gof_score"] for d in range(dim)
                          if "gof_score" in results.get(d, {})] or [0.0])
    results["summary"] = {"pass_count": n_pass, "total": dim, "all_pass": n_pass == dim,
                          "mean_gof_score": float(mean_score)}
    return results


# ===================== 拟合入口 =====================

def fit_hawkes_additive(
        events_list: List[np.ndarray],
        T: float,
        beta_grid: np.ndarray,
        model: str = "A",
        n_days: int = 1,
        intraday_list=None,
        spread_list=None,
        maxiter: int = 100,
        epsilon: float = 1e-4,
        verbose: bool = False,
        init_from: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    完整拟合流程: beta网格搜索 + EM + LL + AIC/BIC + GOF

    events_list : List[np.ndarray]  每维事件时间
    T : 总时长 (交易时间)
    beta_grid : 候选 omega
    model : "A" / "B" / "C"
    init_from : 上一个模型的拟合结果（用于热启动，确保 LL 单调性）
    """
    dim = len(events_list)
    times, types, intra_arr, spread_arr = flatten_events(
        events_list, intraday_list, spread_list)
    N = len(times)
    if N < 2:
        raise ValueError("events too few: %d" % N)

    N_type = np.array([np.sum(types == d) for d in range(dim)])

    periods = None
    if model in ("B", "C"):
        periods = np.array([get_period_tt(t % TRADING_SECONDS_PER_DAY) for t in times], dtype=int)

    # Spread 处理: z-score 归一化 + 非负平移 + max=1 归一化
    spread_shifted = None
    x_total = 1.0
    x_shift = 0.0
    spread_scale = 1.0
    if model == "C" and spread_arr is not None:
        sp_mean = float(np.mean(spread_arr))
        sp_std = float(np.std(spread_arr))
        spread_scale = sp_std if sp_std > 1e-10 else 1.0
        sp_z = (spread_arr - sp_mean) / spread_scale
        x_shift = float(np.min(sp_z))
        spread_shifted = sp_z - x_shift
        sp_max = float(np.max(spread_shifted))
        if sp_max > 1e-10:
            spread_shifted = spread_shifted / sp_max
        if N > 1:
            dt_arr = np.diff(times)
            x_total = float(np.sum(spread_shifted[:-1] * dt_arr)
                            + spread_shifted[-1] * max(T - times[-1], 0))
            x_total = max(x_total, 1.0)

    base_rate = N_type / T + 0.01

    # 如果有上一个模型的结果，优先用其 omega 作为候选
    if init_from is not None and "omega" in init_from:
        prev_omega = init_from["omega"]
        if prev_omega not in beta_grid:
            beta_grid = np.unique(np.append(beta_grid, prev_omega))

    best_ll = -np.inf
    best_res = None
    best_omega = beta_grid[0]

    for omega in beta_grid:
        R_all = _precompute_R(times, types, dim, omega)

        if init_from is not None and "alpha" in init_from:
            alpha_init = np.array(init_from["alpha"], dtype=float)
            if model == "A":
                mu_init = np.array(init_from["mu"], dtype=float)
            elif model in ("B", "C") and "mu_periods" in init_from:
                mu_init = np.array(init_from["mu_periods"], dtype=float)
            elif model in ("B", "C"):
                mu_init = np.tile(np.array(init_from["mu"], dtype=float).reshape(-1, 1), (1, N_PERIODS))
            else:
                mu_init = base_rate.copy()
        else:
            if model == "A":
                mu_init = base_rate.copy()
            else:
                mu_init = np.tile(base_rate.reshape(-1, 1), (1, N_PERIODS))
            alpha_init = 0.05 * np.ones((dim, dim))

        res = em_hawkes_recursive(
            times, types, dim, omega, mu_init, alpha_init,
            T, n_days, model, periods, spread_shifted, x_total,
            maxiter, epsilon, verbose=False, R_all=R_all)

        if verbose:
            print("  omega=%.2f: LL=%.2f, iter=%d" % (omega, res["loglik"], res["n_iter"]))
        if res["loglik"] > best_ll:
            best_ll = res["loglik"]
            best_res = res
            best_omega = omega

    # 暖启动: 用最优 omega 再跑一次 (更多迭代)
    R_all_best = _precompute_R(times, types, dim, best_omega)
    if model == "A":
        mu_warm = best_res["mu"].copy()
    else:
        mu_warm = best_res["mu_periods"].copy()
    alpha_warm = best_res["alpha"].copy()
    best_res = em_hawkes_recursive(
        times, types, dim, best_omega, mu_warm, alpha_warm,
        T, n_days, model, periods, spread_shifted, x_total,
        maxiter=maxiter * 2, epsilon=epsilon / 10, verbose=verbose,
        R_all=R_all_best)
    best_ll = best_res["loglik"]

    # 确保 LL 单调性: C >= B >= A
    if init_from is not None and "loglik" in init_from:
        if best_ll < init_from["loglik"]:
            prev_omega = init_from["omega"]
            R_all_prev = _precompute_R(times, types, dim, prev_omega)
            if "mu_periods" in init_from:
                mu_prev = np.array(init_from["mu_periods"], dtype=float)
            else:
                mu_prev = np.tile(np.array(init_from["mu"], dtype=float).reshape(-1, 1), (1, N_PERIODS)) \
                    if model in ("B", "C") else np.array(init_from["mu"], dtype=float)
            alpha_prev = np.array(init_from["alpha"], dtype=float)
            res_retry = em_hawkes_recursive(
                times, types, dim, prev_omega, mu_prev, alpha_prev,
                T, n_days, model, periods, spread_shifted, x_total,
                maxiter=maxiter * 3, epsilon=epsilon / 100, verbose=verbose,
                R_all=R_all_prev)
            if res_retry["loglik"] > best_ll:
                best_ll = res_retry["loglik"]
                best_res = res_retry
                best_omega = prev_omega

        # 兜底：若 C 模型仍 < B 模型 LL，退化 gamma_spread=0 并继承 B 参数
        if best_ll < init_from["loglik"] and model == "C":
            if verbose:
                print("  [fallback] Model C LL < Model B LL, degenerating gamma_spread=0")
            best_ll = init_from["loglik"]
            best_omega = init_from["omega"]
            best_res["loglik"] = best_ll
            best_res["gamma_spread"] = np.zeros(dim)
            if "mu_periods" in init_from:
                best_res["mu_periods"] = np.array(init_from["mu_periods"], dtype=float)
            if "alpha" in init_from:
                best_res["alpha"] = np.array(init_from["alpha"], dtype=float)
            if "mu" in init_from:
                best_res["mu"] = np.array(init_from["mu"], dtype=float)
            for k_name in ("gamma_open", "gamma_mid", "gamma_close"):
                if k_name in init_from:
                    best_res[k_name] = np.array(init_from[k_name], dtype=float)

    br = float(np.max(np.abs(np.linalg.eigvals(best_res["alpha"]))))

    if model == "A":
        k = dim + dim * dim
    elif model == "B":
        k = N_PERIODS * dim + dim * dim
    else:
        k = N_PERIODS * dim + dim + dim * dim

    aic = -2 * best_ll + 2 * k
    bic = -2 * best_ll + k * np.log(N)

    gof = gof_residuals(
        times, types, dim, best_omega, best_res["alpha"], T, n_days,
        model, best_res.get("mu"), best_res.get("mu_periods"), periods,
        best_res.get("gamma_spread"), spread_shifted)

    out = {
        "model": model, "omega": float(best_omega),
        "mu": best_res["mu"].tolist(),
        "alpha": best_res["alpha"].tolist(),
        "loglik": float(best_ll), "aic": float(aic), "bic": float(bic),
        "branching_ratio": br, "n_params": k, "n_events": N,
        "n_iter": best_res["n_iter"],
        "gof_summary": gof["summary"],
    }
    if model in ("B", "C"):
        out["gamma_open"] = best_res["gamma_open"].tolist()
        out["gamma_mid"] = best_res["gamma_mid"].tolist()
        out["gamma_close"] = best_res["gamma_close"].tolist()
        out["mu_periods"] = best_res["mu_periods"].tolist()
    if model == "C":
        out["gamma_spread"] = best_res["gamma_spread"].tolist()
        out["gamma_spread_raw"] = (best_res["gamma_spread"] / spread_scale).tolist()
        out["x_shift"] = x_shift
        out["spread_scale"] = spread_scale
    out["gof_details"] = {str(d): gof[d] for d in range(dim) if d in gof}
    return out


# ===================== Ogata thinning 模拟器 =====================

def simulate_hawkes_additive(
        dim: int, mu: np.ndarray, alpha: np.ndarray, omega: float,
        T: float, n_days: int = 1,
        gamma_open=None, gamma_mid=None, gamma_close=None,
        gamma_spread=None, spread_func=None,
        seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Ogata thinning 模拟多维 Hawkes 过程 (加性基线)。

    Returns: (events_list, intraday_list, spread_list)
    """
    rng = np.random.RandomState(seed)
    events = [[] for _ in range(dim)]
    intraday = [[] for _ in range(dim)]
    spreads = [[] for _ in range(dim)] if spread_func else None

    t = 0.0
    R = np.zeros(dim)

    mu_max = float(np.max(mu))
    gam_max = 0.0
    if gamma_open is not None:
        gam_max += float(np.max(np.abs(np.concatenate([gamma_open, gamma_mid, gamma_close]))))
    sp_max = float(np.max(np.abs(gamma_spread))) * 3.5 if gamma_spread is not None else 0.0
    lam_bar = dim * (mu_max + gam_max + sp_max) + 0.5

    while t < T:
        lam_star = lam_bar + np.sum(R) + 0.1
        dt = rng.exponential(1.0 / lam_star)
        t_new = t + dt
        if t_new >= T:
            break
        R *= np.exp(-omega * dt)

        t_day = t_new % TRADING_SECONDS_PER_DAY
        clock = trading_time_to_clock(t_day)
        I_o, I_m, I_c = compute_indicators(clock)

        lam = np.zeros(dim)
        for i in range(dim):
            lam[i] = mu[i]
            if gamma_open is not None:
                lam[i] += gamma_open[i] * I_o + gamma_mid[i] * I_m + gamma_close[i] * I_c
            if gamma_spread is not None and spread_func is not None:
                lam[i] += gamma_spread[i] * spread_func(t_new)
            lam[i] += np.dot(alpha[i, :], R)
            lam[i] = max(lam[i], 0.0)

        total = np.sum(lam)
        if rng.rand() * lam_star < total:
            probs = lam / total
            d_sel = rng.choice(dim, p=probs)
            events[d_sel].append(t_new)
            intraday[d_sel].append(clock)
            if spread_func is not None:
                spreads[d_sel].append(spread_func(t_new))
            R[d_sel] += omega
        t = t_new

    ev_out = [np.array(events[d]) for d in range(dim)]
    it_out = [np.array(intraday[d]) for d in range(dim)]
    sp_out = [np.array(spreads[d]) for d in range(dim)] if spread_func else None
    return ev_out, it_out, sp_out


# ===================== 三模型对比流程 =====================

def run_abc_comparison(
        events_list, T, beta_grid, n_days=1,
        intraday_list=None, spread_list=None,
        maxiter=100, verbose=True,
) -> Dict[str, Any]:
    """
    依次拟合 Model A / B / C 并对比。

    Returns: dict 含 "A","B","C" 三个子 dict 以及 "comparison" 汇总
    """
    results = {}
    for m in ["A", "B", "C"]:
        if verbose:
            print(f"\n{'='*50}\n  Fitting Model {m}\n{'='*50}")
        t0 = _time.time()
        r = fit_hawkes_additive(
            events_list, T, beta_grid, model=m, n_days=n_days,
            intraday_list=intraday_list if m in ("B", "C") else None,
            spread_list=spread_list if m == "C" else None,
            maxiter=maxiter, verbose=verbose)
        r["elapsed_s"] = _time.time() - t0
        results[m] = r
        if verbose:
            print(f"  Model {m}: LL={r['loglik']:.2f}  AIC={r['aic']:.2f}  "
                  f"BIC={r['bic']:.2f}  BR={r['branching_ratio']:.4f}  "
                  f"time={r['elapsed_s']:.1f}s")

    ll_a, ll_b, ll_c = results["A"]["loglik"], results["B"]["loglik"], results["C"]["loglik"]
    results["comparison"] = {
        "LL_B_minus_A": ll_b - ll_a,
        "LL_C_minus_B": ll_c - ll_b,
        "LL_monotonic": ll_c >= ll_b >= ll_a,
        "AIC_best": min(["A", "B", "C"], key=lambda m: results[m]["aic"]),
        "BIC_best": min(["A", "B", "C"], key=lambda m: results[m]["bic"]),
    }
    return results
