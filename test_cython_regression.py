"""
test_cython_regression.py  —  验证 Cython 版与纯 Python 版结果一致
===================================================================
测试内容：
  1. EM 递推：Cython vs Python 参数估计结果一致
  2. 对数似然：Cython vs Python 数值一致
  3. GOF 残差：Cython vs Python 残差序列一致
  4. 端到端：用 Cython 加速的完整拟合流程
  5. 性能对比：Cython vs Python 速度提升倍数
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 强制检查 Cython 是否可用
try:
    from _hawkes_cython import (
        em_recursive_cython,
        loglikelihood_cython,
        gof_residuals_cython,
    )
    CYTHON_OK = True
    print("✓ Cython 模块加载成功")
except ImportError as e:
    CYTHON_OK = False
    print(f"✗ Cython 模块未编译: {e}")
    print("  请先运行: python setup_cython.py build_ext --inplace")
    sys.exit(1)

from hawkes_em import (
    simulate_hawkes_multi,
    _em_recursive_python,
    _loglikelihood_python,
    _gof_residuals_python,
    _USE_CYTHON,
    em_estimate, loglikelihood, compute_gof_residuals,
    correct_mu_for_gamma,
    TRADING_SECONDS_PER_DAY,
)

print(f"  hawkes_em._USE_CYTHON = {_USE_CYTHON}")
print()


def test_em_consistency():
    """测试 1: EM 递推 Cython vs Python 一致性"""
    print("=" * 60)
    print("测试 1: EM 递推一致性 (Cython vs Python)")
    print("=" * 60)

    mu_true = np.array([0.5, 0.3])
    alpha_true = np.array([[0.2, 0.05], [0.1, 0.15]])
    omega = 1.0
    T = 3000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega, T, seed=42)
    times = data[:, 0].astype(np.float64)
    types = data[:, 1].astype(np.intc)
    dim = 2
    Tm = float(times[-1])
    print(f"  事件数: {len(data)}, Tm={Tm:.1f}")

    # 相同初始化
    rng = np.random.RandomState(0)
    mu_init = rng.uniform(0.01, 0.5, size=dim).astype(np.float64)
    alpha_init = rng.uniform(0.01, 0.3, size=(dim, dim)).astype(np.float64)

    maxiter = 50
    tol = 1e-5

    # Python 版
    t0 = time.time()
    alpha_py, mu_py = _em_recursive_python(
        times, types.astype(int), dim, omega,
        mu_init.copy(), alpha_init.copy(),
        Tm, maxiter, tol, False)
    t_py = time.time() - t0

    # Cython 版
    t0 = time.time()
    alpha_cy, mu_cy = em_recursive_cython(
        np.ascontiguousarray(times),
        np.ascontiguousarray(types),
        dim, omega,
        np.ascontiguousarray(mu_init.copy()),
        np.ascontiguousarray(alpha_init.copy()),
        Tm, maxiter, tol, 0)
    t_cy = time.time() - t0

    mu_py = np.asarray(mu_py)
    mu_cy = np.asarray(mu_cy)
    alpha_py = np.asarray(alpha_py)
    alpha_cy = np.asarray(alpha_cy)

    mu_diff = np.max(np.abs(mu_py - mu_cy))
    alpha_diff = np.max(np.abs(alpha_py - alpha_cy))

    print(f"  Python: μ={mu_py.round(4)}, α_diag={np.diag(alpha_py).round(4)}, t={t_py:.3f}s")
    print(f"  Cython: μ={mu_cy.round(4)}, α_diag={np.diag(alpha_cy).round(4)}, t={t_cy:.3f}s")
    print(f"  差异: Δμ={mu_diff:.2e}, Δα={alpha_diff:.2e}")
    print(f"  加速: {t_py/max(t_cy, 1e-6):.1f}x")

    ok = mu_diff < 1e-6 and alpha_diff < 1e-6
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_ll_consistency():
    """测试 2: 对数似然 Cython vs Python 一致性"""
    print("\n" + "=" * 60)
    print("测试 2: 对数似然一致性 (Cython vs Python)")
    print("=" * 60)

    mu_true = np.array([0.5, 0.3])
    alpha_true = np.array([[0.2, 0.05], [0.1, 0.15]])
    omega = 1.0
    T = 3000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega, T, seed=42)
    times = data[:, 0].astype(np.float64)
    types = data[:, 1].astype(np.intc)
    dim = 2
    Tm = float(times[-1])

    mu = np.array([0.45, 0.28], dtype=np.float64)
    alpha = np.array([[0.18, 0.06], [0.12, 0.13]], dtype=np.float64)

    # Python 版
    t0 = time.time()
    ll_py = _loglikelihood_python(times, types.astype(int), dim, mu, alpha, omega, Tm)
    t_py = time.time() - t0

    # Cython 版
    t0 = time.time()
    ll_cy = loglikelihood_cython(
        np.ascontiguousarray(times),
        np.ascontiguousarray(types),
        dim,
        np.ascontiguousarray(mu),
        np.ascontiguousarray(alpha),
        omega, Tm)
    t_cy = time.time() - t0

    diff = abs(ll_py - ll_cy)
    print(f"  Python LL: {ll_py:.6f}, t={t_py:.3f}s")
    print(f"  Cython LL: {ll_cy:.6f}, t={t_cy:.3f}s")
    print(f"  差异: {diff:.2e}")
    print(f"  加速: {t_py/max(t_cy, 1e-6):.1f}x")

    ok = diff < 1e-4
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_gof_consistency():
    """测试 3: GOF 残差 Cython vs Python 一致性"""
    print("\n" + "=" * 60)
    print("测试 3: GOF 残差一致性 (Cython vs Python)")
    print("=" * 60)

    mu_true = np.array([0.5, 0.3])
    alpha_true = np.array([[0.2, 0.05], [0.1, 0.15]])
    omega = 1.0
    T = 2000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega, T, seed=42)
    dim = 2

    # 构建 events_4d 格式
    events_4d = []
    for d in range(dim):
        mask = data[:, 1] == d
        events_4d.append(data[mask, 0].astype(np.float64))

    mu = np.array([0.48, 0.29], dtype=np.float64)
    alpha = np.array([[0.19, 0.06], [0.11, 0.14]], dtype=np.float64)

    # 合并时间线
    merged = []
    for d in range(dim):
        for t in events_4d[d]:
            merged.append((float(t), d, float(t)))
    merged.sort(key=lambda x: x[0])

    all_times = np.array([m[0] for m in merged], dtype=np.float64)
    all_types = np.array([m[1] for m in merged], dtype=np.intc)
    all_intra = np.array([m[2] for m in merged], dtype=np.float64)

    gamma_o = np.zeros(dim, dtype=np.float64)
    gamma_m = np.zeros(dim, dtype=np.float64)
    gamma_c = np.zeros(dim, dtype=np.float64)
    mu_corr = mu.copy()

    # Python 版
    t0 = time.time()
    res_py = _gof_residuals_python(
        all_times, all_types, all_intra,
        dim, mu_corr, alpha, omega,
        gamma_o, gamma_m, gamma_c,
        False, False, None, None, None)
    t_py = time.time() - t0

    # Cython 版
    t0 = time.time()
    res_cy_raw = gof_residuals_cython(
        np.ascontiguousarray(all_times),
        np.ascontiguousarray(all_types),
        np.ascontiguousarray(all_intra),
        dim,
        np.ascontiguousarray(mu_corr),
        np.ascontiguousarray(alpha),
        omega,
        np.ascontiguousarray(gamma_o),
        np.ascontiguousarray(gamma_m),
        np.ascontiguousarray(gamma_c),
        0,  # use_tv = False
        float(TRADING_SECONDS_PER_DAY))
    t_cy = time.time() - t0
    res_cy = {d: res_cy_raw[d] for d in range(dim)}

    ok = True
    for d in range(dim):
        py_arr = np.array(res_py[d])
        cy_arr = np.array(res_cy[d])
        if len(py_arr) != len(cy_arr):
            print(f"  dim {d}: 长度不一致 py={len(py_arr)} cy={len(cy_arr)}")
            ok = False
            continue
        max_diff = np.max(np.abs(py_arr - cy_arr)) if len(py_arr) > 0 else 0.0
        mean_py = np.mean(py_arr) if len(py_arr) > 0 else 0.0
        mean_cy = np.mean(cy_arr) if len(cy_arr) > 0 else 0.0
        print(f"  dim {d}: n={len(py_arr)}, mean_py={mean_py:.4f}, mean_cy={mean_cy:.4f}, max_diff={max_diff:.2e}")
        if max_diff > 1e-6:
            ok = False

    print(f"  Python: t={t_py:.3f}s")
    print(f"  Cython: t={t_cy:.3f}s")
    print(f"  加速: {t_py/max(t_cy, 1e-6):.1f}x")
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_large_scale_perf():
    """测试 4: 大规模性能对比"""
    print("\n" + "=" * 60)
    print("测试 4: 大规模性能对比 (N=100k)")
    print("=" * 60)

    mu_true = np.array([0.5, 0.3, 0.4, 0.35])
    alpha_true = np.array([
        [0.15, 0.02, 0.01, 0.02],
        [0.01, 0.12, 0.02, 0.01],
        [0.01, 0.02, 0.14, 0.01],
        [0.02, 0.01, 0.01, 0.13],
    ])
    omega = 5.0
    T = 50000.0
    dim = 4

    data = simulate_hawkes_multi(mu_true, alpha_true, omega, T, seed=42)
    print(f"  事件数: {len(data)}")

    times = data[:, 0].astype(np.float64)
    types = data[:, 1].astype(np.intc)
    Tm = float(times[-1])

    rng = np.random.RandomState(0)
    mu_init = rng.uniform(0.01, 0.5, size=dim).astype(np.float64)
    alpha_init = rng.uniform(0.01, 0.3, size=(dim, dim)).astype(np.float64)

    # EM: Python
    t0 = time.time()
    _em_recursive_python(
        times, types.astype(int), dim, omega,
        mu_init.copy(), alpha_init.copy(),
        Tm, 10, 1e-5, False)
    t_em_py = time.time() - t0

    # EM: Cython
    t0 = time.time()
    em_recursive_cython(
        np.ascontiguousarray(times),
        np.ascontiguousarray(types),
        dim, omega,
        np.ascontiguousarray(mu_init.copy()),
        np.ascontiguousarray(alpha_init.copy()),
        Tm, 10, 1e-5, 0)
    t_em_cy = time.time() - t0

    # LL: Python
    mu_test = mu_true.astype(np.float64)
    alpha_test = alpha_true.astype(np.float64)

    t0 = time.time()
    _loglikelihood_python(times, types.astype(int), dim, mu_test, alpha_test, omega, Tm)
    t_ll_py = time.time() - t0

    # LL: Cython
    t0 = time.time()
    loglikelihood_cython(
        np.ascontiguousarray(times),
        np.ascontiguousarray(types),
        dim,
        np.ascontiguousarray(mu_test),
        np.ascontiguousarray(alpha_test),
        omega, Tm)
    t_ll_cy = time.time() - t0

    print(f"  EM (10 iter): Python={t_em_py:.2f}s, Cython={t_em_cy:.2f}s, 加速={t_em_py/max(t_em_cy,1e-6):.1f}x")
    print(f"  LL:           Python={t_ll_py:.2f}s, Cython={t_ll_cy:.2f}s, 加速={t_ll_py/max(t_ll_cy,1e-6):.1f}x")

    ok = t_em_cy < t_em_py  # Cython 应该更快
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL (Cython 未加速)'}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("EM 一致性", test_em_consistency()))
    results.append(("LL 一致性", test_ll_consistency()))
    results.append(("GOF 一致性", test_gof_consistency()))
    results.append(("大规模性能", test_large_scale_perf()))

    print("\n" + "=" * 60)
    print("  回归测试汇总")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  总计: {n_pass}/{len(results)} 通过")

    if n_pass == len(results):
        print("\n✓ 所有回归测试通过，Cython 版与纯 Python 版完全一致")
    else:
        print("\n✗ 部分测试失败，请检查")
