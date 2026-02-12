# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
hawkes_cy_loglink.pyx — Cython加速 log-link 基线积分计算
==========================================================
加速热点：
  1. _precompute_segments_cy — 预计算所有分段属性
  2. _compute_loglink_integrals_cy — 向量化积分计算
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log
cimport cython

cnp.import_array()

# A股交易时间常量
cdef double MARKET_OPEN_AM = 34200.0
cdef double MARKET_CLOSE_AM = 41400.0
cdef double MARKET_OPEN_PM = 46800.0
cdef double MARKET_CLOSE_PM = 54000.0
cdef double OPEN30_START = 34200.0
cdef double OPEN30_END = 36000.0
cdef double MID30_START = 41400.0
cdef double MID30_END = 43200.0
cdef double CLOSE30_START = 52200.0
cdef double CLOSE30_END = 54000.0
cdef double TRADING_SECONDS_PER_DAY = 14400.0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _trading_time_to_intraday(double t_trading) nogil:
    """连续交易时间 → 日内时间戳（秒）"""
    if t_trading < 7200.0:
        return MARKET_OPEN_AM + t_trading
    else:
        return MARKET_OPEN_PM + (t_trading - 7200.0)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _compute_indicators(double t_intraday, double* I_o, double* I_m, double* I_c) nogil:
    """计算时段哑变量"""
    I_o[0] = 1.0 if (OPEN30_START <= t_intraday < OPEN30_END) else 0.0
    I_m[0] = 1.0 if (MID30_START <= t_intraday < MID30_END) else 0.0
    I_c[0] = 1.0 if (CLOSE30_START <= t_intraday < CLOSE30_END) else 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def precompute_segments_cy(double T, int n_days,
                           cnp.ndarray[cnp.float64_t, ndim=1] spread_times=None):
    """
    预计算所有分段的属性（Cython加速版本）
    
    Returns:
      seg_dt, seg_Io, seg_Im, seg_Ic, seg_xs_idx (all 1D arrays)
    """
    cdef list all_bounds_list = [0.0, T]
    cdef double[:] dummy_offsets = np.array([0.0, 1800.0, 7200.0, 9000.0, 12600.0, TRADING_SECONDS_PER_DAY])
    cdef int day, i, k
    cdef double off, tb, t_mid, day_idx_f, t_in_day, t_intraday
    cdef double I_o, I_m, I_c
    cdef set bounds_set = {0.0, T}
    
    # 添加每日dummy边界
    for day in range(max(n_days, 1)):
        off = day * TRADING_SECONDS_PER_DAY
        for i in range(6):
            tb = off + dummy_offsets[i]
            if 0.0 < tb < T:
                bounds_set.add(tb)
    
    # 添加spread观测时间
    cdef int n_spread = 0
    if spread_times is not None:
        n_spread = len(spread_times)
        for i in range(n_spread):
            if 0.0 < spread_times[i] < T:
                bounds_set.add(spread_times[i])
    
    # 排序
    cdef list bounds_list = sorted(bounds_set)
    cdef int K = len(bounds_list) - 1
    
    if K <= 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int32))
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1] seg_dt = np.zeros(K)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] seg_Io = np.zeros(K)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] seg_Im = np.zeros(K)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] seg_Ic = np.zeros(K)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] seg_xs_idx = np.zeros(K, dtype=np.int32)
    
    cdef double t1, t2, dt
    cdef int day_idx, spread_idx, j
    
    for k in range(K):
        t1 = bounds_list[k]
        t2 = bounds_list[k + 1]
        dt = t2 - t1
        
        if dt <= 1e-15:
            seg_dt[k] = 0.0
            continue
        
        seg_dt[k] = dt
        
        # 中点判断dummy
        t_mid = (t1 + t2) * 0.5
        day_idx = <int>(t_mid / TRADING_SECONDS_PER_DAY)
        t_in_day = t_mid - day_idx * TRADING_SECONDS_PER_DAY
        t_intraday = _trading_time_to_intraday(t_in_day)
        _compute_indicators(t_intraday, &I_o, &I_m, &I_c)
        seg_Io[k] = I_o
        seg_Im[k] = I_m
        seg_Ic[k] = I_c
        
        # 记录spread索引（piecewise constant: 找到 <= t_mid 的最大索引）
        if spread_times is not None:
            spread_idx = 0
            for j in range(n_spread):
                if spread_times[j] <= t_mid:
                    spread_idx = j
                else:
                    break
            seg_xs_idx[k] = spread_idx
        else:
            seg_xs_idx[k] = -1
    
    # 过滤 dt <= 0 的段
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] mask = seg_dt > 1e-15
    return (seg_dt[mask], seg_Io[mask], seg_Im[mask], seg_Ic[mask], seg_xs_idx[mask])


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_loglink_integrals_cy(
        int dim,
        cnp.ndarray[cnp.float64_t, ndim=1] seg_dt,
        cnp.ndarray[cnp.float64_t, ndim=1] seg_Io,
        cnp.ndarray[cnp.float64_t, ndim=1] seg_Im,
        cnp.ndarray[cnp.float64_t, ndim=1] seg_Ic,
        cnp.ndarray[cnp.int32_t, ndim=1] seg_xs_idx,
        cnp.ndarray[cnp.float64_t, ndim=1] spread_values=None,
        cnp.ndarray[cnp.float64_t, ndim=1] gamma_open=None,
        cnp.ndarray[cnp.float64_t, ndim=1] gamma_mid=None,
        cnp.ndarray[cnp.float64_t, ndim=1] gamma_close=None,
        cnp.ndarray[cnp.float64_t, ndim=1] gamma_spread=None):
    """
    log-link 基线积分计算（Cython加速版本）
    
    Returns:
      eff_T, eff_T_open, eff_T_mid, eff_T_close, eff_T_spread (all shape (dim,))
    """
    cdef int K = len(seg_dt)
    if K == 0:
        return (np.zeros(dim), np.zeros(dim), np.zeros(dim), np.zeros(dim), np.zeros(dim))
    
    cdef int use_gamma = (gamma_open is not None)
    cdef int use_spread = (gamma_spread is not None and spread_values is not None)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1] eff_T = np.zeros(dim)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] eff_T_open = np.zeros(dim)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] eff_T_mid = np.zeros(dim)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] eff_T_close = np.zeros(dim)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] eff_T_spread = np.zeros(dim)
    
    cdef int d, k, xs_idx
    cdef double eta, exp_eta, w, xs_val, dt
    
    for d in range(dim):
        for k in range(K):
            dt = seg_dt[k]
            if dt <= 1e-15:
                continue
            
            # 构建 η
            eta = 0.0
            if use_gamma:
                eta += gamma_open[d] * seg_Io[k]
                eta += gamma_mid[d] * seg_Im[k]
                eta += gamma_close[d] * seg_Ic[k]
            
            xs_val = 0.0
            if use_spread:
                xs_idx = seg_xs_idx[k]
                if xs_idx >= 0 and xs_idx < len(spread_values):
                    xs_val = spread_values[xs_idx]
                eta += gamma_spread[d] * xs_val
            
            exp_eta = exp(eta)
            w = exp_eta * dt
            
            eff_T[d] += w
            eff_T_open[d] += seg_Io[k] * w
            eff_T_mid[d] += seg_Im[k] * w
            eff_T_close[d] += seg_Ic[k] * w
            if use_spread:
                eff_T_spread[d] += xs_val * w
    
    return eff_T, eff_T_open, eff_T_mid, eff_T_close, eff_T_spread
