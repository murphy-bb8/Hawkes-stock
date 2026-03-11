"""
加性基线 Hawkes EM 实现 — 递推 O(N·d²) 版本
=============================================
参考 MHP_fixed_final_play.py 的 EM 框架，改为递推计算以支持大规模数据。

Model A: λ_i(t) = μ_i + Σ_j α_{ij}·ω·exp(-ω·Δt)
Model B: λ_i(t) = μ_{i,period(t)} + Σ_j α_{ij}·ω·exp(-ω·Δt)
         其中 period ∈ {normal, open30, mid30, close30}，分段常数基线
Model C: Model B + Σ_v γ_{v,i}·x_v⁺(t)
         支持多外生变量 (OBI, log_opp_depth 等), 通过 exog_lists dict 传入

核函数: φ_{ij}(Δt) = α_{ij}·ω·exp(-ω·Δt)
积分:   ∫₀^∞ φ_{ij}(s)ds = α_{ij}

EM M步全部闭式解:
  μ_{i,p} = Σ_{n:u_n=i,per(n)=p} p_{n,bg} / T_p
  α_{ij}  = Σ_{n:u_n=i} (α_{ij}·R_j[n]/λ_i) / N_j
  γ_{v,i} = Σ_{n:u_n=i} p_{n,exog_v} / X_v_total   (每个外生变量独立更新)
"""

import numpy as np
import json
import os
import time as _time
from typing import List, Tuple, Optional, Dict, Any
from scipy.stats import kstest, wasserstein_distance

# Cython 加速模块（可选，回退到纯 Python）
try:
    from _hawkes_cy import precompute_R_cy, gof_residuals_cy, compute_ll_cy, em_hawkes_recursive_cy
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False
    em_hawkes_recursive_cy = None

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


def flatten_events(events_list, intraday_list=None, exog_lists=None):
    """
    将多维事件列表展平为按时间排序的数组。

    Parameters
    ----------
    exog_lists : Dict[str, List[np.ndarray]] 或 None
        外生变量字典, key=变量名, value=每维事件对应值

    Returns: (times, types, intraday_arr, exog_flat)
        exog_flat : Dict[str, np.ndarray] 或 None
    """
    dim = len(events_list)
    N_total = sum(len(ev) for ev in events_list)

    times = np.empty(N_total)
    types = np.empty(N_total, dtype=int)
    has_intra = intraday_list is not None
    has_exog = exog_lists is not None and len(exog_lists) > 0
    intra = np.empty(N_total) if has_intra else None
    exog_flat = {v: np.empty(N_total) for v in exog_lists} if has_exog else None

    idx = 0
    for d in range(dim):
        n = len(events_list[d])
        if n == 0:
            continue
        times[idx:idx + n] = events_list[d]
        types[idx:idx + n] = d
        if has_intra:
            intra[idx:idx + n] = intraday_list[d]
        if has_exog:
            for v in exog_lists:
                exog_flat[v][idx:idx + n] = exog_lists[v][d]
        idx += n

    times = times[:idx]
    types = types[:idx]
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    types = types[sort_idx].astype(int)
    if has_intra:
        intra = intra[:idx][sort_idx]
    if has_exog:
        for v in exog_flat:
            exog_flat[v] = exog_flat[v][:idx][sort_idx]
    return times, types, intra, exog_flat


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
        exog_shifted: Optional[Dict[str, np.ndarray]] = None,
        x_totals: Optional[Dict[str, float]] = None,
        maxiter: int = 100, epsilon: float = 1e-4, verbose: bool = False,
        R_all: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    向量化递推 EM。R_all 预计算后，每次迭代全部 numpy 向量化。
    若 Cython 可用，优先调用 em_hawkes_recursive_cy 加速。
    """
    N = len(times)
    if N == 0:
        return {"mu": mu_init, "alpha": alpha_init, "loglik": -np.inf, "n_iter": 0}

    if exog_shifted is None:
        exog_shifted = {}
    if x_totals is None:
        x_totals = {}
    var_names = list(exog_shifted.keys())

    model_code = {"A": 0, "B": 1, "C": 2}[model]
    N_type = np.array([np.sum(types == d) for d in range(dim)], dtype=np.int64)

    if R_all is None:
        R_all = _precompute_R(times, types, dim, omega)

    comp = np.zeros(dim)
    for d in range(dim):
        mask_d = types == d
        comp[d] = np.sum(1.0 - np.exp(-omega * (T - times[mask_d])))

    if periods is None:
        periods = np.zeros(N, dtype=np.int64)

    _mu = np.asarray(mu_init, dtype=np.float64)
    if _mu.ndim == 1:
        mu_init_2d = np.ascontiguousarray(
            np.tile(_mu.reshape(-1, 1), (1, N_PERIODS)), dtype=np.float64)
    else:
        mu_init_2d = np.ascontiguousarray(_mu, dtype=np.float64)

    if _USE_CYTHON and em_hawkes_recursive_cy is not None and not verbose:
        exog_2d = None
        xtot_1d = None
        if var_names:
            exog_2d = np.ascontiguousarray(
                np.stack([exog_shifted[v] for v in var_names]), dtype=np.float64)
            xtot_1d = np.array([x_totals.get(v, 1.0) for v in var_names], dtype=np.float64)
        try:
            return em_hawkes_recursive_cy(
                np.ascontiguousarray(times, dtype=np.float64),
                np.ascontiguousarray(types, dtype=np.int64),
                dim, float(omega), mu_init_2d,
                np.ascontiguousarray(alpha_init, dtype=np.float64),
                T, n_days, model_code,
                np.ascontiguousarray(periods, dtype=np.int64),
                np.ascontiguousarray(R_all, dtype=np.float64),
                np.ascontiguousarray(comp, dtype=np.float64),
                N_type,
                exog_2d, xtot_1d, var_names,
                maxiter, epsilon)
        except Exception:
            pass

    T_per = PERIOD_SECS_PER_DAY * n_days  # (4,)
    type_masks = [(types == d) for d in range(dim)]
    type_idx = types

    alpha = alpha_init.copy().astype(float)
    if model == "A":
        mu = mu_init.copy().ravel().astype(float)
    else:
        mu_p = mu_init.copy().astype(float)
        if mu_p.ndim == 1:
            mu_p = np.tile(mu_p.reshape(-1, 1), (1, N_PERIODS))

    gamma_exog = None
    if model == "C" and var_names:
        init_base = N_type / T + 0.01
        gamma_exog = {v: init_base * 0.1 for v in var_names}

    period_masks = None
    if model in ("B", "C") and periods is not None:
        period_masks = {}
        for i in range(dim):
            for p in range(N_PERIODS):
                period_masks[(i, p)] = type_masks[i] & (periods == p)

    old_ll = -1e30

    for it in range(maxiter):
        if model == "A":
            bases = mu[type_idx]
        else:
            bases = mu_p[type_idx, periods]

        exc = np.sum(alpha[type_idx, :] * R_all, axis=1)

        sp_per_var = {}
        sp = np.zeros(N)
        if model == "C" and gamma_exog is not None:
            for v in var_names:
                sp_v = gamma_exog[v][type_idx] * exog_shifted[v]
                sp_per_var[v] = sp_v
                sp += sp_v

        lam = bases + exc + sp
        lam = np.maximum(lam, 1e-15)
        inv_lam = 1.0 / lam

        ll_sum = np.sum(np.log(lam))

        bg_w = bases * inv_lam

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

        if model == "C" and gamma_exog is not None:
            for v in var_names:
                xt = x_totals.get(v, 1.0)
                if xt > 0:
                    sp_w_v = sp_per_var[v] * inv_lam
                    new_gv = np.zeros(dim)
                    for i in range(dim):
                        new_gv[i] = max(np.sum(sp_w_v[type_masks[i]]) / xt, 0.0)
                    gamma_exog[v] = new_gv

        new_alpha = np.zeros((dim, dim))
        for i in range(dim):
            m_i = type_masks[i]
            R_i = R_all[m_i, :]
            w_i = inv_lam[m_i]
            for j in range(dim):
                if N_type[j] > 0:
                    new_alpha[i, j] = max(
                        alpha[i, j] * np.dot(R_i[:, j], w_i) / N_type[j], 0.0)
        alpha = new_alpha

        if model == "A":
            int_base = np.sum(mu) * T
        else:
            int_base = np.sum(mu_p * T_per[np.newaxis, :])
        int_sp = 0.0
        if model == "C" and gamma_exog is not None:
            int_sp = sum(np.sum(gamma_exog[v]) * x_totals.get(v, 1.0) for v in var_names)
        int_exc = np.sum(alpha * comp[np.newaxis, :])
        ll = ll_sum - int_base - int_sp - int_exc

        if verbose and it % 10 == 0:
            print(f"  EM iter {it}: LL={ll:.4f}")
        if it > 0 and abs(ll - old_ll) < epsilon:
            if verbose:
                print(f"  EM converged iter {it}: LL={ll:.4f}")
            break
        old_ll = ll

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
    if model == "C" and gamma_exog is not None:
        res["gamma_exog"] = {v: gamma_exog[v].copy() for v in var_names}
    return res


# ===================== 对数似然 (独立验证) =====================

def compute_loglikelihood(
        times, types, dim, omega, alpha, T, n_days=1,
        model="A", mu=None, mu_periods=None,
        periods=None,
        gamma_exog=None, exog_shifted=None, x_totals=None,
) -> float:
    """递推计算 LL, 用于独立验证。"""
    N = len(times)
    if gamma_exog is None:
        gamma_exog = {}
    if exog_shifted is None:
        exog_shifted = {}
    if x_totals is None:
        x_totals = {}
    var_names = list(gamma_exog.keys())

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
        if model == "C":
            for v in var_names:
                lam += gamma_exog[v][i] * exog_shifted[v][n]
        lam = max(lam, 1e-15)
        ll_sum += np.log(lam)
        R[i] += omega
        last_t = times[n]

    T_per = PERIOD_SECS_PER_DAY * n_days
    if model == "A":
        int_base = np.sum(mu) * T
    else:
        int_base = np.sum(mu_periods * T_per[np.newaxis, :])
    int_sp = 0.0
    if model == "C":
        int_sp = sum(np.sum(gamma_exog[v]) * x_totals.get(v, 1.0) for v in var_names)
    comp = np.zeros(dim)
    for n in range(N):
        comp[types[n]] += 1.0 - np.exp(-omega * (T - times[n]))
    int_exc = np.sum(alpha * comp[np.newaxis, :])
    return float(ll_sum - int_base - int_sp - int_exc)


# ===================== GOF 残差 =====================

def gof_residuals(
        times, types, dim, omega, alpha, T, n_days=1,
        model="A", mu=None, mu_periods=None, periods=None,
        gamma_exog=None, exog_shifted=None,
) -> Dict[str, Any]:
    """
    时间重标度残差: 若模型正确, τ_k ~ Exp(1)。
    使用精确激励积分。
    """
    N = len(times)
    if gamma_exog is None:
        gamma_exog = {}
    if exog_shifted is None:
        exog_shifted = {}
    var_names = list(gamma_exog.keys())

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
                if model == "C":
                    for v in var_names:
                        sp_int += gamma_exog[v][d] * exog_shifted[v][k] * dt
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
        exog_lists: Optional[Dict[str, List[np.ndarray]]] = None,
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
    exog_lists : Dict[str, List[np.ndarray]] — 外生变量 {变量名: [每维数组]}
    init_from : 上一个模型的拟合结果（用于热启动，确保 LL 单调性）
    """
    dim = len(events_list)
    times, types, intra_arr, exog_flat = flatten_events(
        events_list, intraday_list,
        exog_lists if model == "C" else None)
    N = len(times)
    if N < 2:
        raise ValueError("events too few: %d" % N)

    N_type = np.array([np.sum(types == d) for d in range(dim)])

    periods = None
    if model in ("B", "C"):
        periods = np.array([get_period_tt(t % TRADING_SECONDS_PER_DAY) for t in times], dtype=int)

    # 外生变量处理: 对每个变量做 z-score + 非负平移 + max=1 归一化
    exog_shifted = {}
    x_totals = {}
    exog_meta = {}
    if model == "C" and exog_flat is not None:
        for v, arr in exog_flat.items():
            sp_mean = float(np.mean(arr))
            sp_std = float(np.std(arr))
            scale = sp_std if sp_std > 1e-10 else 1.0
            sp_z = (arr - sp_mean) / scale
            shift = float(np.min(sp_z))
            shifted = sp_z - shift
            sp_max = float(np.max(shifted))
            if sp_max > 1e-10:
                shifted = shifted / sp_max
            exog_shifted[v] = shifted
            exog_meta[v] = {"scale": scale, "shift": shift}
            if N > 1:
                dt_arr = np.diff(times)
                xt = float(np.sum(shifted[:-1] * dt_arr)
                           + shifted[-1] * max(T - times[-1], 0))
                x_totals[v] = max(xt, 1.0)
            else:
                x_totals[v] = 1.0

    var_names = list(exog_shifted.keys())
    n_exog = len(var_names)

    base_rate = N_type / T + 0.01

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
            T, n_days, model, periods,
            exog_shifted if model == "C" else None,
            x_totals if model == "C" else None,
            maxiter, epsilon, verbose=False, R_all=R_all)

        if verbose:
            print("  omega=%.2f: LL=%.2f, iter=%d" % (omega, res["loglik"], res["n_iter"]))
        if res["loglik"] > best_ll:
            best_ll = res["loglik"]
            best_res = res
            best_omega = omega

    # 暖启动: 用最优 omega 再跑一次
    R_all_best = _precompute_R(times, types, dim, best_omega)
    if model == "A":
        mu_warm = best_res["mu"].copy()
    else:
        mu_warm = best_res["mu_periods"].copy()
    alpha_warm = best_res["alpha"].copy()
    best_res = em_hawkes_recursive(
        times, types, dim, best_omega, mu_warm, alpha_warm,
        T, n_days, model, periods,
        exog_shifted if model == "C" else None,
        x_totals if model == "C" else None,
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
                T, n_days, model, periods,
                exog_shifted if model == "C" else None,
                x_totals if model == "C" else None,
                maxiter=maxiter * 3, epsilon=epsilon / 100, verbose=verbose,
                R_all=R_all_prev)
            if res_retry["loglik"] > best_ll:
                best_ll = res_retry["loglik"]
                best_res = res_retry
                best_omega = prev_omega

        # 兜底：若 C 模型仍 < B 模型 LL，退化 gamma=0 并继承 B 参数
        if best_ll < init_from["loglik"] and model == "C":
            if verbose:
                print("  [fallback] Model C LL < Model B LL, degenerating gamma_exog=0")
            best_ll = init_from["loglik"]
            best_omega = init_from["omega"]
            best_res["loglik"] = best_ll
            best_res["gamma_exog"] = {v: np.zeros(dim) for v in var_names}
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
        k = N_PERIODS * dim + n_exog * dim + dim * dim

    aic = -2 * best_ll + 2 * k
    bic = -2 * best_ll + k * np.log(N)

    gamma_exog = best_res.get("gamma_exog", {})
    gof = gof_residuals(
        times, types, dim, best_omega, best_res["alpha"], T, n_days,
        model, best_res.get("mu"), best_res.get("mu_periods"), periods,
        gamma_exog, exog_shifted)

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
        out["exog_vars"] = var_names
        for v in var_names:
            out["gamma_%s" % v] = gamma_exog[v].tolist()
            out["gamma_%s_raw" % v] = (gamma_exog[v] / exog_meta[v]["scale"]).tolist()
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
        intraday_list=None, exog_lists=None,
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
            exog_lists=exog_lists if m == "C" else None,
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
