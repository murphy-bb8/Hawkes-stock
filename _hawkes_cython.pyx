# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""
_hawkes_cython.pyx  —  Hawkes 过程核心循环的 Cython 加速
=========================================================
包含三个核心函数：
  1. em_recursive_cython   — EM 递推主循环（支持 Model A/B/C）
  2. loglikelihood_cython   — 对数似然递推（支持 Model A/B/C）
  3. gof_residuals_cython   — GOF 残差生成主循环（支持 Model A/B/C）

所有函数使用与 hawkes_em.py 完全一致的口径：
  核函数: φ_{ij}(Δt) = α[i,j] · ω · exp(-ω · Δt)
  积分:   ∫_0^∞ φ_{ij}(s)ds = α[i,j]

统一强度函数：
  Model A: λ_d(t) = μ_d + Σ_j α_{dj}·ω·r_j(t)
  Model B: λ_d(t) = μ_base_d·exp(γ·I(t)) + Σ_j α_{dj}·ω·r_j(t)
  Model C: Model B + γ_spread_d · re_spread(t)
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, fabs

np.import_array()

# 时段边界常量
cdef double _OPEN30_S = 34200.0
cdef double _OPEN30_E = 36000.0
cdef double _MID30_S  = 46800.0
cdef double _MID30_E  = 48600.0
cdef double _CLOSE30_S = 52200.0
cdef double _CLOSE30_E = 54000.0


cdef double _sum_vec(double[::1] v, int n) noexcept:
    cdef double s = 0.0
    cdef int i
    for i in range(n):
        s += v[i]
    return s


cdef inline void _compute_indicators(double t_intra, double *I_o, double *I_m, double *I_c) noexcept:
    """计算日内时段指示变量"""
    I_o[0] = 1.0 if (_OPEN30_S <= t_intra < _OPEN30_E) else 0.0
    I_m[0] = 1.0 if (_MID30_S <= t_intra < _MID30_E) else 0.0
    I_c[0] = 1.0 if (_CLOSE30_S <= t_intra < _CLOSE30_E) else 0.0


# ===================== EM 递推主循环 =====================

def em_recursive_cython(
    double[::1] times,
    int[::1] types,
    int dim,
    double omega,
    double[::1] mu_init,
    double[:,::1] alpha_init,
    double Tm,
    int maxiter,
    double tol,
    int verbose,
    double[::1] intraday_times = None,
    double[::1] gamma_open = None,
    double[::1] gamma_mid = None,
    double[::1] gamma_close = None,
    double[::1] gamma_spread = None,
    double[::1] spread_at_events = None,
    double[::1] baseline_integral_T = None,
    double[::1] spread_integral_T = None,
):
    """
    Cython 加速的 EM 递推算法（支持 Model A/B/C）。
    联合估计 mu, alpha, gamma_spread。

    Returns
    -------
    alpha_hat    : (dim, dim)
    mu_hat       : (dim,)
    gamma_spread : (dim,) 或 None
    """
    cdef int N = times.shape[0]
    cdef int i, it, a_idx, b_idx, k, u_i, d
    cdef int use_tv = (intraday_times is not None and gamma_open is not None)
    cdef int use_spread = (spread_at_events is not None)

    # 工作数组
    cdef double[::1] mu_hat = np.array(mu_init, dtype=np.float64, copy=True)
    cdef double[:,::1] alpha_hat = np.array(alpha_init, dtype=np.float64, copy=True)

    # gamma_spread 工作拷贝
    cdef double[::1] gs_hat
    if use_spread:
        if gamma_spread is not None:
            gs_hat = np.array(gamma_spread, dtype=np.float64, copy=True)
        else:
            gs_hat = np.full(dim, 0.01, dtype=np.float64)

    # 预计算 dt 和 decay
    cdef double[::1] decay_arr = np.empty(N, dtype=np.float64)
    cdef double dt_val
    decay_arr[0] = 1.0
    for i in range(1, N):
        dt_val = times[i] - times[i - 1]
        if dt_val < 0.0:
            dt_val = 0.0
        decay_arr[i] = exp(-omega * dt_val)

    # 预计算时变基线乘子 tv_mult[i, d] = exp(γ·I(t_i))
    cdef double[:,::1] tv_mult
    cdef double I_o, I_m, I_c
    if use_tv:
        tv_mult = np.ones((N, dim), dtype=np.float64)
        for i in range(N):
            _compute_indicators(intraday_times[i], &I_o, &I_m, &I_c)
            for d in range(dim):
                tv_mult[i, d] = exp(gamma_open[d] * I_o + gamma_mid[d] * I_m + gamma_close[d] * I_c)

    # baseline_integral_T 退化为 Tm
    cdef double[::1] bl_int_T
    if baseline_integral_T is not None:
        bl_int_T = np.array(baseline_integral_T, dtype=np.float64, copy=True)
    else:
        bl_int_T = np.full(dim, Tm, dtype=np.float64)

    # spread_integral_T
    cdef double[::1] sp_int_T
    if use_spread:
        if spread_integral_T is not None:
            sp_int_T = np.array(spread_integral_T, dtype=np.float64, copy=True)
        else:
            sp_int_T = np.full(dim, Tm * 0.01, dtype=np.float64)

    # 充分统计量
    cdef double[::1] sum_p_bg = np.zeros(dim, dtype=np.float64)
    cdef double[:,::1] sum_pij = np.zeros((dim, dim), dtype=np.float64)
    cdef double[::1] sum_p_exog = np.zeros(dim, dtype=np.float64)
    cdef double[::1] type_counts = np.zeros(dim, dtype=np.float64)
    cdef double[::1] r = np.zeros(dim, dtype=np.float64)
    cdef double[::1] exc = np.zeros(dim, dtype=np.float64)

    cdef double sum_log_rates, lam_i, exc_sum, inv_lam, mu_val, exog_val, sp_i
    cdef double old_LL = -1e15
    cdef double LL, integral_alpha, mu_integral, exog_integral

    for it in range(maxiter):
        # 重置统计量
        for k in range(dim):
            sum_p_bg[k] = 0.0
            sum_p_exog[k] = 0.0
            type_counts[k] = 0.0
            r[k] = 0.0
            for d in range(dim):
                sum_pij[k, d] = 0.0

        sum_log_rates = 0.0

        for i in range(N):
            u_i = types[i]

            # 衰减递推量
            if i > 0:
                for d in range(dim):
                    r[d] *= decay_arr[i]

            # 基线项
            if use_tv:
                mu_val = mu_hat[u_i] * tv_mult[i, u_i]
            else:
                mu_val = mu_hat[u_i]

            # 激励项
            exc_sum = 0.0
            for d in range(dim):
                exc[d] = alpha_hat[u_i, d] * omega * r[d]
                exc_sum += exc[d]

            # 外生项
            exog_val = 0.0
            if use_spread:
                sp_i = spread_at_events[i]
                if sp_i < 0.0:
                    sp_i = 0.0
                exog_val = gs_hat[u_i] * sp_i

            lam_i = mu_val + exc_sum + exog_val
            if lam_i < 1e-15:
                lam_i = 1e-15

            # E-step: 各成分责任
            inv_lam = 1.0 / lam_i
            sum_p_bg[u_i] += mu_val * inv_lam

            for d in range(dim):
                sum_pij[u_i, d] += exc[d] * inv_lam

            if use_spread and exog_val > 0.0:
                sum_p_exog[u_i] += exog_val * inv_lam

            type_counts[u_i] += 1.0
            sum_log_rates += log(lam_i)

            # 更新递推量
            r[u_i] += 1.0

        # --- M step ---
        for k in range(dim):
            mu_hat[k] = sum_p_bg[k] / bl_int_T[k]
            if mu_hat[k] < 1e-10:
                mu_hat[k] = 1e-10

        for a_idx in range(dim):
            for b_idx in range(dim):
                if type_counts[b_idx] > 0.0:
                    alpha_hat[a_idx, b_idx] = sum_pij[a_idx, b_idx] / type_counts[b_idx]
                else:
                    alpha_hat[a_idx, b_idx] = 0.0
                if alpha_hat[a_idx, b_idx] < 0.0:
                    alpha_hat[a_idx, b_idx] = 0.0

        # γ_spread M-step
        if use_spread:
            for d in range(dim):
                gs_hat[d] = sum_p_exog[d] / sp_int_T[d]
                if gs_hat[d] < 0.0:
                    gs_hat[d] = 0.0

        # --- 收敛检查 (每5次) ---
        if it % 5 == 0:
            integral_alpha = 0.0
            for d in range(dim):
                for i in range(N):
                    integral_alpha += alpha_hat[d, types[i]]

            mu_integral = 0.0
            for d in range(dim):
                mu_integral += mu_hat[d] * bl_int_T[d]

            exog_integral = 0.0
            if use_spread:
                for d in range(dim):
                    exog_integral += gs_hat[d] * sp_int_T[d]

            LL = (sum_log_rates - mu_integral - integral_alpha - exog_integral) / N

            if verbose and it % 20 == 0:
                print(f"  EM iter {it}: LL={LL:.4f}")

            if fabs(LL - old_LL) < tol:
                if verbose:
                    print(f"  EM converged at iter {it}: LL={LL:.4f}")
                break
            old_LL = LL

    if use_spread:
        return np.asarray(alpha_hat), np.asarray(mu_hat), np.asarray(gs_hat)
    return np.asarray(alpha_hat), np.asarray(mu_hat), None


# ===================== 对数似然递推 =====================

def loglikelihood_cython(
    double[::1] times,
    int[::1] types,
    int dim,
    double[::1] mu,
    double[:,::1] alpha,
    double omega,
    double Tm,
    double[::1] intraday_times = None,
    double[::1] gamma_open = None,
    double[::1] gamma_mid = None,
    double[::1] gamma_close = None,
    double[::1] gamma_spread = None,
    double[::1] spread_at_events = None,
    double[::1] baseline_integral_T = None,
    double exog_integral = 0.0,
):
    """
    精确对数似然（递推计算，O(N·D)），支持 Model A/B/C。

    LL = Σ_i log λ_{u_i}(t_i) - Σ_d ∫_0^T λ_d(t) dt

    口径与 EM 完全一致。
    """
    cdef int N = times.shape[0]
    if N < 1:
        return -1e15

    cdef int i, d, u_i
    cdef int use_tv = (intraday_times is not None and gamma_open is not None)
    cdef int use_spread = (gamma_spread is not None and spread_at_events is not None)
    cdef double dt_val, decay_f, lam_i, mu_val, exog_val
    cdef double sum_log_lam = 0.0
    cdef double I_o, I_m, I_c

    # 递推量
    cdef double[::1] r = np.zeros(dim, dtype=np.float64)

    for i in range(N):
        u_i = types[i]

        # 衰减
        if i > 0:
            dt_val = times[i] - times[i - 1]
            if dt_val < 0.0:
                dt_val = 0.0
            decay_f = exp(-omega * dt_val)
            for d in range(dim):
                r[d] *= decay_f

        # 基线项
        if use_tv:
            _compute_indicators(intraday_times[i], &I_o, &I_m, &I_c)
            mu_val = mu[u_i] * exp(gamma_open[u_i] * I_o + gamma_mid[u_i] * I_m + gamma_close[u_i] * I_c)
        else:
            mu_val = mu[u_i]

        # λ_i = mu_val + Σ_d α_{u_i,d} · ω · r[d]
        lam_i = mu_val
        for d in range(dim):
            lam_i += alpha[u_i, d] * omega * r[d]

        # 外生项
        if use_spread:
            exog_val = gamma_spread[u_i] * spread_at_events[i]
            if exog_val > 0.0:
                lam_i += exog_val

        if lam_i < 1e-15:
            lam_i = 1e-15

        sum_log_lam += log(lam_i)
        r[u_i] += 1.0

    # 积分项
    cdef double integral = 0.0

    # 基线积分
    if baseline_integral_T is not None:
        for d in range(dim):
            integral += mu[d] * baseline_integral_T[d]
    else:
        for d in range(dim):
            integral += mu[d] * Tm

    # 激励积分
    cdef double surv
    for i in range(N):
        surv = 1.0 - exp(-omega * (Tm - times[i]))
        for d in range(dim):
            integral += alpha[d, types[i]] * surv

    # 外生项积分
    integral += exog_integral

    return sum_log_lam - integral


# ===================== GOF 残差生成主循环 =====================

def gof_residuals_cython(
    double[::1] all_times,
    int[::1] all_types,
    double[::1] all_intraday,
    int dim,
    double[::1] mu_corrected,
    double[:,::1] alpha,
    double omega,
    double[::1] gamma_open,
    double[::1] gamma_mid,
    double[::1] gamma_close,
    int use_tv,
    double trading_seconds_per_day,
    int use_spread = 0,
    double[::1] gamma_spread = None,
    double[::1] spread_times = None,
    double[::1] spread_values = None,
):
    """
    GOF 残差生成：连续递推（不逐日重置），与 EM/LL 口径一致。
    支持 Model A/B/C（含外生项）。

    对每个事件 i（维度 d），计算从上一个同维度事件到当前事件的
    累积强度 Λ_d = ∫_{t_{prev_d}}^{t_i} λ_d(s) ds。

    如果模型正确，Λ_d ~ Exp(1)。
    """
    cdef int N = all_times.shape[0]
    cdef int i, d, u_i, j
    cdef double t, dt, decay_f, base_int, exc_int, exog_int, t_intra
    cdef double mid_t, sp_val

    # 递推量 r[d] — 连续递推，不逐日重置
    cdef double[::1] r = np.zeros(dim, dtype=np.float64)

    # 累积强度 Λ_d（从上一个同维度事件开始累积）
    cdef double[::1] Lambda_accum = np.zeros(dim, dtype=np.float64)

    # 上一个事件的时间和日内时间
    cdef double last_t = 0.0
    cdef double last_t_intra = 0.0

    # 每个维度是否已见过第一个事件
    cdef int[::1] first_seen = np.zeros(dim, dtype=np.intc)

    # spread 数组长度
    cdef int n_spread = 0
    if use_spread and spread_times is not None:
        n_spread = spread_times.shape[0]

    # 输出：用 Python list 收集
    residuals = [[] for _ in range(dim)]

    if N == 0:
        return residuals

    last_t = all_times[0]
    last_t_intra = all_intraday[0]

    for i in range(N):
        t = all_times[i]
        u_i = all_types[i]
        t_intra = all_intraday[i]

        dt = t - last_t
        if dt > 0.0:
            decay_f = exp(-omega * dt)

            for d in range(dim):
                # 基线积分
                if use_tv:
                    base_int = _baseline_integral_fast(
                        last_t_intra, t_intra, dt,
                        mu_corrected[d],
                        gamma_open[d], gamma_mid[d], gamma_close[d],
                        _OPEN30_S, _OPEN30_E, _MID30_S, _MID30_E,
                        _CLOSE30_S, _CLOSE30_E, trading_seconds_per_day)
                else:
                    base_int = mu_corrected[d] * dt

                # 激励积分: Σ_j α[d,j] · r[j] · (1 - exp(-ω·dt))
                exc_int = 0.0
                for j in range(dim):
                    exc_int += alpha[d, j] * r[j]
                exc_int *= (1.0 - decay_f)

                # 外生项积分
                exog_int = 0.0
                if use_spread and n_spread > 0:
                    mid_t = (last_t + t) / 2.0
                    sp_val = _interp_spread_fast(mid_t, spread_times, spread_values, n_spread)
                    if sp_val < 0.0:
                        sp_val = 0.0
                    exog_int = gamma_spread[d] * sp_val * dt

                Lambda_accum[d] += base_int + exc_int + exog_int

            # 衰减递推量
            for d in range(dim):
                r[d] *= decay_f

        # 记录残差
        if first_seen[u_i]:
            if Lambda_accum[u_i] > 0.0:
                residuals[u_i].append(Lambda_accum[u_i])
        else:
            first_seen[u_i] = 1

        # 重置该维度的累积强度
        Lambda_accum[u_i] = 0.0

        # 更新递推量
        r[u_i] += 1.0

        last_t = t
        last_t_intra = t_intra

    return residuals


cdef double _interp_spread_fast(double t, double[::1] sp_times, double[::1] sp_values, int n) noexcept:
    """快速线性插值 spread 值"""
    cdef int lo = 0
    cdef int hi = n - 1
    cdef int mid_idx
    if n == 0:
        return 0.0
    if t <= sp_times[0]:
        return sp_values[0]
    if t >= sp_times[hi]:
        return sp_values[hi]
    # 二分查找
    while lo < hi - 1:
        mid_idx = (lo + hi) // 2
        if sp_times[mid_idx] <= t:
            lo = mid_idx
        else:
            hi = mid_idx
    # 线性插值
    cdef double dt_sp = sp_times[hi] - sp_times[lo]
    if dt_sp < 1e-15:
        return sp_values[lo]
    cdef double frac = (t - sp_times[lo]) / dt_sp
    return sp_values[lo] + frac * (sp_values[hi] - sp_values[lo])


cdef double _baseline_integral_fast(
    double t1_intra, double t2_intra, double dt_trading,
    double mu_base,
    double g_o, double g_m, double g_c,
    double OPEN30_S, double OPEN30_E,
    double MID30_S, double MID30_E,
    double CLOSE30_S, double CLOSE30_E,
    double tspd
) noexcept:
    """
    快速计算时变基线积分。

    对于跨日情况（dt > tspd），使用全天平均强度近似中间完整天。
    对于日内情况，使用分段常数近似（中点法）。
    """
    # 所有 cdef 声明必须在函数顶部
    cdef double avg_day, n_days, avg_rate
    cdef double mid_intra
    cdef double total = 0.0
    cdef double seg_start, seg_end, seg_dt, seg_mid
    cdef double I_o, I_m, I_c, mu_t
    cdef double[6] bounds
    cdef int n_bounds = 1
    cdef double[5] all_bounds
    cdef int bi

    if dt_trading <= 0.0:
        return 0.0

    # 跨日检测：如果 dt 超过一个交易日
    if dt_trading > tspd * 1.5:
        # 多日：用全天平均强度
        avg_day = (
            mu_base * exp(g_o) * 1800.0 +
            mu_base * exp(g_m) * 1800.0 +
            mu_base * exp(g_c) * 1800.0 +
            mu_base * 9000.0
        )
        n_days = dt_trading / tspd
        return avg_day * n_days

    # 日内：分段计算
    mid_intra = (t1_intra + t2_intra) / 2.0

    # 处理跨午休的情况
    if t1_intra > t2_intra:
        # 跨日：用全天平均
        avg_rate = (
            mu_base * exp(g_o) * 1800.0 +
            mu_base * exp(g_m) * 1800.0 +
            mu_base * exp(g_c) * 1800.0 +
            mu_base * 9000.0
        ) / tspd
        return avg_rate * dt_trading

    # 分段计算
    bounds[0] = t1_intra
    all_bounds[0] = OPEN30_E   # 36000
    all_bounds[1] = 41400.0    # MARKET_CLOSE_AM
    all_bounds[2] = 46800.0    # MARKET_OPEN_PM
    all_bounds[3] = MID30_E    # 48600
    all_bounds[4] = CLOSE30_S  # 52200

    for bi in range(5):
        if t1_intra < all_bounds[bi] < t2_intra:
            bounds[n_bounds] = all_bounds[bi]
            n_bounds += 1
    bounds[n_bounds] = t2_intra
    n_bounds += 1

    # 对每段用中点法
    for bi in range(n_bounds - 1):
        seg_start = bounds[bi]
        seg_end = bounds[bi + 1]
        if seg_end <= seg_start:
            continue

        seg_mid = (seg_start + seg_end) / 2.0

        # 判断时段
        I_o = 1.0 if (OPEN30_S <= seg_mid < OPEN30_E) else 0.0
        I_m = 1.0 if (MID30_S <= seg_mid < MID30_E) else 0.0
        I_c = 1.0 if (CLOSE30_S <= seg_mid < CLOSE30_E) else 0.0

        mu_t = mu_base * exp(g_o * I_o + g_m * I_m + g_c * I_c)

        # 将日内时间差转为交易时间差
        seg_dt = _intraday_to_trading_dt(seg_start, seg_end)
        total += mu_t * seg_dt

    return total


cdef double _intraday_to_trading_dt(double t1, double t2) noexcept:
    """计算两个日内时间点之间的交易时间差"""
    cdef double tt1 = _intraday_to_tt(t1)
    cdef double tt2 = _intraday_to_tt(t2)
    cdef double dt = tt2 - tt1
    if dt < 0.0:
        dt = 0.0
    return dt


cdef double _intraday_to_tt(double t) noexcept:
    """日内时间 → 交易时间 (0-14400)"""
    if t < 34200.0:
        return 0.0
    elif t <= 41400.0:
        return t - 34200.0
    elif t < 46800.0:
        return 7200.0
    elif t <= 54000.0:
        return 7200.0 + (t - 46800.0)
    else:
        return 14400.0
