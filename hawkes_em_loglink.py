"""
log-link 基线版本的 Hawkes EM 实现
直接复用当前 hawkes_em.py 的 fit_4d、loglikelihood_loglink、_gof_residuals_loglink
确保与加性版本严格对应，仅改变基线链接方式。
"""

import numpy as np
from typing import List, Optional, Dict, Any

# 直接导入当前 log-link 实现
from hawkes_em import (
    fit_4d, loglikelihood_loglink, _gof_residuals_loglink,
    SpreadProcess,
)

def fit_4d_loglink(events_4d: List[np.ndarray],
                  T: float,
                  beta_grid: np.ndarray,
                  model: str = "A",
                  events_4d_original: Optional[List[np.ndarray]] = None,
                  maxiter: int = 10,
                  tol: float = 1e-5,
                  verbose: bool = False,
                  n_days: int = 22) -> Dict[str, Any]:
    """
    log-link 基线版本的 4D Hawkes 拟合（直接调用当前实现）。
    """
    result = fit_4d(
        events_4d, T, beta_grid,
        model=model,
        events_4d_original=events_4d_original if model in ("B", "C") else None,
        maxiter=maxiter,
        tol=tol,
        verbose=verbose,
        n_days=n_days,
    )
    
    # 添加标识字段
    result["baseline_type"] = "log-link"
    return result


def loglikelihood_loglink_wrapper(*args, **kwargs):
    """包装当前 loglikelihood_loglink"""
    return loglikelihood_loglink(*args, **kwargs)


def gof_residuals_loglink_wrapper(*args, **kwargs):
    """包装当前 _gof_residuals_loglink"""
    return _gof_residuals_loglink(*args, **kwargs)
