# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython 加速核心: precompute_R, em_hawkes_recursive_cy, gof_residuals_cy, compute_ll_cy
适配 hawkes_em_additive.py 的加性基线 EM 算法
"""
import numpy as np
cimport numpy as np
from libc.math cimport exp, log, fabs, fmax

np.import_array()

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.int32_t I32_t

cdef int N_PERIODS = 4
cdef int PERIOD_OPEN = 0
cdef int PERIOD_MID = 1
cdef int PERIOD_CLOSE = 2
cdef int PERIOD_NORMAL = 3


def precompute_R_cy(np.ndarray[DTYPE_t, ndim=1] times,
                    np.ndarray[ITYPE_t, ndim=1] types,
                    int dim, double omega):
    """
    预计算 R_all[n, j] = sum_{k<n, u_k=j} omega * exp(-omega*(t_n - t_k))
    O(N*dim) 递推
    """
    cdef int N = times.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] R_all = np.zeros((N, dim), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] R_cur = np.zeros(dim, dtype=np.float64)
    cdef double prev_t = 0.0
    cdef double dt, decay_val
    cdef int n, j, typ

    for n in range(N):
        dt = times[n] - prev_t
        if dt > 0:
            decay_val = exp(-omega * dt)
            for j in range(dim):
                R_cur[j] *= decay_val
        for j in range(dim):
            R_all[n, j] = R_cur[j]
        typ = <int>types[n]
        R_cur[typ] += omega
        prev_t = times[n]
    return R_all


def em_hawkes_recursive_cy(
    np.ndarray[DTYPE_t, ndim=1] times,
    np.ndarray[ITYPE_t, ndim=1] types,
    int dim,
    double omega,
    np.ndarray[DTYPE_t, ndim=2] mu_init,
    np.ndarray[DTYPE_t, ndim=2] alpha_init,
    double T,
    int n_days,
    int model_code,
    np.ndarray[ITYPE_t, ndim=1] periods,
    np.ndarray[DTYPE_t, ndim=2] R_all,
    np.ndarray[DTYPE_t, ndim=1] comp,
    np.ndarray[ITYPE_t, ndim=1] N_type,
    object exog_shifted_2d,  # (n_vars, N) or None
    object x_totals_1d,      # (n_vars,) or None
    object var_names,        # list of str for gamma_exog keys
    int maxiter,
    double epsilon
):
    """
    Cython 加速的 EM 主循环。与 em_hawkes_recursive 数学完全一致。
    model_code: 0=A, 1=B, 2=C
    mu_init: Model A 时 (dim,1) 会被展平; B/C 时 (dim, 4)
    """
    cdef int N = times.shape[0]
    if N == 0:
        return {"mu": mu_init[:, 0].copy(), "alpha": alpha_init.copy(),
                "loglik": -1e30, "n_iter": 0}

    cdef int n_vars = 0
    cdef np.ndarray[DTYPE_t, ndim=2] exog_2d
    cdef np.ndarray[DTYPE_t, ndim=1] xtot_1d
    if exog_shifted_2d is not None and x_totals_1d is not None:
        exog_2d = np.ascontiguousarray(exog_shifted_2d, dtype=np.float64)
        xtot_1d = np.ascontiguousarray(x_totals_1d, dtype=np.float64)
        n_vars = exog_2d.shape[0]

    cdef double T_per_0 = 1800.0 * n_days
    cdef double T_per_1 = 1800.0 * n_days
    cdef double T_per_2 = 1800.0 * n_days
    cdef double T_per_3 = 9000.0 * n_days

    cdef np.ndarray[DTYPE_t, ndim=1] mu = np.zeros(dim, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] mu_p = np.zeros((dim, N_PERIODS), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] alpha = np.array(alpha_init, dtype=np.float64, copy=True)
    cdef np.ndarray[DTYPE_t, ndim=2] gamma_exog_2d
    if model_code == 0:
        for i in range(dim):
            mu[i] = mu_init[i, 0] if mu_init.shape[1] > 0 else 0.01
    else:
        for i in range(dim):
            for p in range(N_PERIODS):
                mu_p[i, p] = mu_init[i, p]
    if model_code == 2 and n_vars > 0:
        gamma_exog_2d = np.zeros((n_vars, dim), dtype=np.float64)
        for v in range(n_vars):
            for i in range(dim):
                gamma_exog_2d[v, i] = (N_type[i] / T + 0.01) * 0.1

    cdef np.ndarray[DTYPE_t, ndim=1] bases = np.empty(N, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] exc = np.empty(N, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] sp = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] lam = np.empty(N, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] inv_lam = np.empty(N, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] bg_w = np.empty(N, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] sp_per_var
    if model_code == 2 and n_vars > 0:
        sp_per_var = np.zeros((n_vars, N), dtype=np.float64)

    cdef double ll_sum, int_base, int_sp, int_exc, ll_val
    cdef double old_ll = -1e30
    cdef int it, n, ti, pi
    cdef double s, xt, dot_val
    cdef int converged = 0

    for it in range(maxiter):
        if model_code == 0:
            for n in range(N):
                ti = <int>types[n]
                bases[n] = mu[ti]
        else:
            for n in range(N):
                ti = <int>types[n]
                pi = <int>periods[n]
                bases[n] = mu_p[ti, pi]

        for n in range(N):
            exc[n] = 0.0
            ti = <int>types[n]
            for j in range(dim):
                exc[n] += alpha[ti, j] * R_all[n, j]

        for n in range(N):
            sp[n] = 0.0
        if model_code == 2 and n_vars > 0:
            for v in range(n_vars):
                for n in range(N):
                    ti = <int>types[n]
                    s = gamma_exog_2d[v, ti] * exog_2d[v, n]
                    sp_per_var[v, n] = s
                    sp[n] += s

        ll_sum = 0.0
        for n in range(N):
            lam[n] = bases[n] + exc[n] + sp[n]
            if lam[n] < 1e-15:
                lam[n] = 1e-15
            inv_lam[n] = 1.0 / lam[n]
            ll_sum += log(lam[n])

        for n in range(N):
            bg_w[n] = bases[n] * inv_lam[n]

        if model_code == 0:
            for i in range(dim):
                s = 0.0
                for n in range(N):
                    if types[n] == i:
                        s += bg_w[n]
                mu[i] = fmax(s / T, 1e-10)
        else:
            for i in range(dim):
                for p in range(N_PERIODS):
                    s = 0.0
                    for n in range(N):
                        if types[n] == i and periods[n] == p:
                            s += bg_w[n]
                    if p == 0:
                        xt = T_per_0
                    elif p == 1:
                        xt = T_per_1
                    elif p == 2:
                        xt = T_per_2
                    else:
                        xt = T_per_3
                    if xt > 0:
                        mu_p[i, p] = fmax(s / xt, 1e-10)
                    else:
                        mu_p[i, p] = 1e-10

        if model_code == 2 and n_vars > 0:
            for v in range(n_vars):
                xt = xtot_1d[v] if xtot_1d[v] > 0 else 1.0
                for i in range(dim):
                    s = 0.0
                    for n in range(N):
                        if types[n] == i:
                            s += sp_per_var[v, n] * inv_lam[n]
                    gamma_exog_2d[v, i] = fmax(s / xt, 0.0)

        for i in range(dim):
            for j in range(dim):
                if N_type[j] > 0:
                    dot_val = 0.0
                    for n in range(N):
                        if types[n] == i:
                            dot_val += R_all[n, j] * inv_lam[n]
                    alpha[i, j] = fmax(alpha[i, j] * dot_val / N_type[j], 0.0)
                else:
                    alpha[i, j] = 0.0

        if model_code == 0:
            int_base = 0.0
            for i in range(dim):
                int_base += mu[i]
            int_base *= T
        else:
            int_base = 0.0
            for i in range(dim):
                int_base += (mu_p[i, 0] + mu_p[i, 1] + mu_p[i, 2]) * 1800.0 * n_days + mu_p[i, 3] * 9000.0 * n_days

        int_sp = 0.0
        if model_code == 2 and n_vars > 0:
            for v in range(n_vars):
                s = 0.0
                for i in range(dim):
                    s += gamma_exog_2d[v, i]
                int_sp += s * xtot_1d[v]

        int_exc = 0.0
        for i in range(dim):
            for j in range(dim):
                int_exc += alpha[i, j] * comp[j]

        ll_val = ll_sum - int_base - int_sp - int_exc

        if it > 0 and fabs(ll_val - old_ll) < epsilon:
            converged = 1
            break
        old_ll = ll_val

    cdef dict res = {}
    res["alpha"] = alpha.copy()
    res["loglik"] = float(ll_val)
    res["n_iter"] = it + 1

    if model_code == 0:
        res["mu"] = mu.copy()
    else:
        res["mu"] = mu_p[:, PERIOD_NORMAL].copy()
        res["mu_periods"] = mu_p.copy()
        res["gamma_open"] = (mu_p[:, PERIOD_OPEN] - mu_p[:, PERIOD_NORMAL]).copy()
        res["gamma_mid"] = (mu_p[:, PERIOD_MID] - mu_p[:, PERIOD_NORMAL]).copy()
        res["gamma_close"] = (mu_p[:, PERIOD_CLOSE] - mu_p[:, PERIOD_NORMAL]).copy()

    if model_code == 2 and n_vars > 0 and var_names is not None:
        gamma_exog = {}
        for v in range(n_vars):
            gamma_exog[var_names[v]] = gamma_exog_2d[v, :].copy()
        res["gamma_exog"] = gamma_exog

    return res


def gof_residuals_cy(np.ndarray[DTYPE_t, ndim=1] times,
                     np.ndarray[ITYPE_t, ndim=1] types,
                     int dim, double omega,
                     np.ndarray[DTYPE_t, ndim=2] alpha,
                     double T,
                     int model_code,
                     np.ndarray[DTYPE_t, ndim=1] mu,
                     np.ndarray[DTYPE_t, ndim=2] mu_periods,
                     np.ndarray[ITYPE_t, ndim=1] periods,
                     np.ndarray[DTYPE_t, ndim=1] gamma_spread,
                     np.ndarray[DTYPE_t, ndim=1] spread_shifted):
    """
    时间重标度残差计算 (GOF)
    model_code: 0=A, 1=B, 2=C
    返回 list of arrays (每个维度的残差)
    """
    cdef int N = times.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] R = np.zeros(dim, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] Lambda_run = np.zeros(dim, dtype=np.float64)
    cdef np.ndarray[np.uint8_t, ndim=1] seen = np.zeros(dim, dtype=np.uint8)
    cdef double last_t = 0.0
    cdef double dt, decay_val, b, base_int, exc_int, sp_int, dot_val
    cdef int k, i, d, j

    residuals = [[] for _ in range(dim)]

    for k in range(N):
        dt = times[k] - last_t
        i = <int>types[k]
        if dt > 0:
            decay_val = exp(-omega * dt)
            for d in range(dim):
                if model_code == 0:
                    b = mu[d]
                else:
                    b = mu_periods[d, <int>periods[k]]
                base_int = b * dt
                dot_val = 0.0
                for j in range(dim):
                    dot_val += alpha[d, j] * R[j]
                if omega > 0:
                    exc_int = dot_val * (1.0 - decay_val) / omega
                else:
                    exc_int = 0.0
                sp_int = 0.0
                if model_code == 2:
                    sp_int = gamma_spread[d] * spread_shifted[k] * dt
                Lambda_run[d] += base_int + exc_int + sp_int
            for j in range(dim):
                R[j] *= decay_val

        if seen[i] and Lambda_run[i] > 0:
            residuals[i].append(Lambda_run[i])
        seen[i] = 1
        Lambda_run[i] = 0.0
        R[i] += omega
        last_t = times[k]

    return residuals


def compute_ll_cy(np.ndarray[DTYPE_t, ndim=1] times,
                  np.ndarray[ITYPE_t, ndim=1] types,
                  int dim, double omega,
                  np.ndarray[DTYPE_t, ndim=2] alpha,
                  int model_code,
                  np.ndarray[DTYPE_t, ndim=1] mu,
                  np.ndarray[DTYPE_t, ndim=2] mu_periods,
                  np.ndarray[ITYPE_t, ndim=1] periods,
                  np.ndarray[DTYPE_t, ndim=1] gamma_spread,
                  np.ndarray[DTYPE_t, ndim=1] spread_shifted):
    """
    递推计算 log-likelihood 求和项
    返回 ll_sum (不含积分补偿部分)
    """
    cdef int N = times.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] R = np.zeros(dim, dtype=np.float64)
    cdef double ll_sum = 0.0
    cdef double last_t = 0.0
    cdef double dt, decay_val, lam_val, dot_val
    cdef int n, i, j

    for n in range(N):
        dt = times[n] - last_t
        i = <int>types[n]
        if dt > 0:
            decay_val = exp(-omega * dt)
            for j in range(dim):
                R[j] *= decay_val
        if model_code == 0:
            lam_val = mu[i]
        else:
            lam_val = mu_periods[i, <int>periods[n]]
        dot_val = 0.0
        for j in range(dim):
            dot_val += alpha[i, j] * R[j]
        lam_val += dot_val
        if model_code == 2:
            lam_val += gamma_spread[i] * spread_shifted[n]
        if lam_val < 1e-15:
            lam_val = 1e-15
        ll_sum += log(lam_val)
        R[i] += omega
        last_t = times[n]
    return ll_sum
