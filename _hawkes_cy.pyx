# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython 加速核心: precompute_R, gof_residuals_loop, compute_ll_loop
适配 hawkes_em_additive.py 的加性基线 EM 算法
"""
import numpy as np
cimport numpy as np
from libc.math cimport exp, log, fabs

np.import_array()

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.int32_t I32_t


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
