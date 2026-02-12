"""
test_1d_simulation.py  —  一维 Hawkes 仿真 → EM 拟合 → GOF 验证
================================================================
目的：在小样本上确认 hawkes_em.py 的代码正确性，包括：
  1. 模拟器正确性（事件数 ≈ 理论值）
  2. EM 参数回收（μ, α 接近真实值）
  3. GOF 残差 ≈ Exp(1)
  4. 不同参数组合的鲁棒性

运行：conda activate py385 && python test_1d_simulation.py
"""

import numpy as np
import sys
import os

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawkes_em import (simulate_hawkes_multi, em_estimate, loglikelihood,
                       grid_search_beta, fit_1d, compute_gof_residuals)


def test_simulation_basic():
    """测试 1: 模拟器基本正确性"""
    print("=" * 60)
    print("测试 1: 模拟器基本正确性")
    print("=" * 60)

    mu = np.array([0.5])
    alpha = np.array([[0.3]])
    omega = 1.0
    T = 2000.0

    data = simulate_hawkes_multi(mu, alpha, omega, T, seed=42)
    N = len(data)

    # 理论期望事件数: E[N] = μ / (1 - α) * T = 0.5 / 0.7 * 2000 ≈ 1428.6
    expected_N = float(mu[0] / (1.0 - alpha[0, 0]) * T)
    ratio = N / expected_N

    print(f"  参数: μ={mu[0]}, α={alpha[0,0]}, ω={omega}, T={T}")
    print(f"  生成事件数: {N}")
    print(f"  理论期望:   {expected_N:.1f}")
    print(f"  比值:       {ratio:.3f}")

    ok = 0.5 < ratio < 2.0
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'} (比值在 0.5~2.0 之间)")
    return ok


def test_em_parameter_recovery():
    """测试 2: EM 参数回收"""
    print("\n" + "=" * 60)
    print("测试 2: EM 参数回收 (固定 β)")
    print("=" * 60)

    mu_true = np.array([0.5])
    alpha_true = np.array([[0.3]])
    omega_true = 1.0
    T = 3000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega_true, T, seed=123)
    N = len(data)
    print(f"  真实参数: μ=0.5, α=0.3, ω=1.0")
    print(f"  模拟事件数: {N}, T={T}")

    # 用真实 β 做 EM
    alpha_hat, mu_hat = em_estimate(data, dim=1, omega=omega_true,
                                    Tm=T, maxiter=200, verbose=True)

    mu_err = abs(mu_hat[0] - mu_true[0])
    alpha_err = abs(alpha_hat[0, 0] - alpha_true[0, 0])

    print(f"\n  估计结果: μ̂={mu_hat[0]:.4f}, α̂={alpha_hat[0,0]:.4f}")
    print(f"  真实参数: μ=0.5000, α=0.3000")
    print(f"  误差:     Δμ={mu_err:.4f}, Δα={alpha_err:.4f}")

    ok = mu_err < 0.15 and alpha_err < 0.15
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'} (误差 < 0.15)")
    return ok


def test_beta_grid_search():
    """测试 3: β 网格搜索"""
    print("\n" + "=" * 60)
    print("测试 3: β 网格搜索")
    print("=" * 60)

    mu_true = np.array([0.5])
    alpha_true = np.array([[0.3]])
    omega_true = 2.0
    T = 3000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega_true, T, seed=456)
    N = len(data)
    print(f"  真实参数: μ=0.5, α=0.3, ω=2.0")
    print(f"  模拟事件数: {N}")

    beta_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0])
    best_beta, best_alpha, best_mu, best_ll = grid_search_beta(
        data, 1, beta_grid, Tm=T, verbose=True)

    print(f"\n  选中 β={best_beta:.2f} (真实=2.0)")
    print(f"  μ̂={best_mu[0]:.4f}, α̂={best_alpha[0,0]:.4f}")

    # β 应该在 1.0~3.0 范围内
    ok = 1.0 <= best_beta <= 3.0
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'} (β 在 1.0~3.0 范围)")
    return ok


def test_loglikelihood():
    """测试 4: 对数似然计算"""
    print("\n" + "=" * 60)
    print("测试 4: 对数似然一致性")
    print("=" * 60)

    mu = np.array([0.5])
    alpha = np.array([[0.3]])
    omega = 1.0
    T = 1000.0

    data = simulate_hawkes_multi(mu, alpha, omega, T, seed=789)
    N = len(data)

    ll_true = loglikelihood(data, 1, mu, alpha, omega, Tm=T)

    # 用错误参数计算 LL，应该更低
    mu_wrong = np.array([2.0])
    alpha_wrong = np.array([[0.01]])
    ll_wrong = loglikelihood(data, 1, mu_wrong, alpha_wrong, omega, Tm=T)

    print(f"  真实参数 LL: {ll_true:.2f}")
    print(f"  错误参数 LL: {ll_wrong:.2f}")

    ok = ll_true > ll_wrong
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'} (真实参数 LL > 错误参数 LL)")
    return ok


def test_gof_residuals():
    """测试 5: GOF 残差检验"""
    print("\n" + "=" * 60)
    print("测试 5: GOF 残差 (应接近 Exp(1))")
    print("=" * 60)

    mu_true = np.array([0.5])
    alpha_true = np.array([[0.3]])
    omega_true = 1.0
    T = 5000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega_true, T, seed=321)
    N = len(data)
    print(f"  模拟事件数: {N}, T={T}")

    # 用真实参数计算 GOF
    events_1d = [data[:, 0]]
    gof = compute_gof_residuals(events_1d, T, mu_true, alpha_true, omega_true, dim=1)

    d0 = gof.get("dim_0", {})
    if "error" in d0:
        print(f"  GOF 错误: {d0['error']}")
        return False

    res_mean = d0["mean"]
    w1 = d0["wasserstein_1"]
    gof_score = d0["gof_score"]
    gof_pass = d0["gof_pass"]

    print(f"  残差均值: {res_mean:.4f} (理想=1.0)")
    print(f"  W1 距离:  {w1:.4f}")
    print(f"  GOF 评分: {gof_score:.4f}")
    print(f"  GOF 通过: {gof_pass}")

    ok = 0.7 < res_mean < 1.3 and gof_score > 0.5
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'} (均值在 0.7~1.3, 评分 > 0.5)")
    return ok


def test_fit_1d_convenience():
    """测试 6: fit_1d 便捷接口"""
    print("\n" + "=" * 60)
    print("测试 6: fit_1d 便捷接口 (端到端)")
    print("=" * 60)

    mu_true = np.array([0.8])
    alpha_true = np.array([[0.4]])
    omega_true = 1.5
    T = 3000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega_true, T, seed=654)
    times = data[:, 0]
    N = len(times)
    print(f"  真实参数: μ=0.8, α=0.4, ω=1.5")
    print(f"  模拟事件数: {N}")

    beta_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    result = fit_1d(times, beta_grid, T=T, verbose=True)

    print(f"\n  估计结果:")
    print(f"    μ̂ = {result['mu']:.4f} (真实 0.8)")
    print(f"    α̂ = {result['alpha']:.4f} (真实 0.4)")
    print(f"    β̂ = {result['beta']:.2f} (真实 1.5)")
    print(f"    BR = {result['branching_ratio']:.4f} (真实 0.4)")
    print(f"    GOF score = {result['gof']['summary']['gof_score_mean']:.4f}")

    mu_err = abs(result['mu'] - 0.8)
    alpha_err = abs(result['alpha'] - 0.4)
    ok = mu_err < 0.3 and alpha_err < 0.2
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'} (Δμ<0.3, Δα<0.2)")
    return ok


def test_multidim_simulation_and_em():
    """测试 7: 2D 模拟 + EM"""
    print("\n" + "=" * 60)
    print("测试 7: 2D 模拟 + EM 参数回收")
    print("=" * 60)

    mu_true = np.array([0.3, 0.2])
    alpha_true = np.array([[0.2, 0.05],
                           [0.1, 0.15]])
    omega_true = 1.0
    T = 5000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega_true, T, seed=999)
    N = len(data)
    print(f"  真实参数: μ={mu_true}, ω={omega_true}")
    print(f"  α = {alpha_true.tolist()}")
    print(f"  模拟事件数: {N}")

    alpha_hat, mu_hat = em_estimate(data, dim=2, omega=omega_true,
                                    Tm=T, maxiter=200, verbose=True)

    print(f"\n  估计结果:")
    print(f"    μ̂ = {mu_hat.round(4)}")
    print(f"    α̂ = {alpha_hat.round(4).tolist()}")

    # 检查对角线元素
    diag_err = abs(alpha_hat[0, 0] - 0.2) + abs(alpha_hat[1, 1] - 0.15)
    mu_err = np.sum(np.abs(mu_hat - mu_true))
    print(f"    对角线误差: {diag_err:.4f}")
    print(f"    μ 误差:     {mu_err:.4f}")

    ok = diag_err < 0.3 and mu_err < 0.3
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_gof_with_estimated_params():
    """测试 8: 用估计参数做 GOF（最严格的端到端测试）"""
    print("\n" + "=" * 60)
    print("测试 8: 估计参数 → GOF (端到端)")
    print("=" * 60)

    mu_true = np.array([0.5])
    alpha_true = np.array([[0.3]])
    omega_true = 1.0
    T = 5000.0

    data = simulate_hawkes_multi(mu_true, alpha_true, omega_true, T, seed=111)
    times = data[:, 0]
    N = len(times)
    print(f"  模拟事件数: {N}")

    beta_grid = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    result = fit_1d(times, beta_grid, T=T, verbose=False)

    gof_score = result['gof']['summary']['gof_score_mean']
    res_mean = result['gof'].get('dim_0', {}).get('mean', 0)

    print(f"  估计: μ̂={result['mu']:.4f}, α̂={result['alpha']:.4f}, β̂={result['beta']:.2f}")
    print(f"  GOF score: {gof_score:.4f}")
    print(f"  残差均值:  {res_mean:.4f}")

    ok = gof_score > 0.5 and 0.5 < res_mean < 1.5
    print(f"  结果: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def main():
    print("=" * 60)
    print("  一维 Hawkes 仿真验证套件")
    print("  hawkes_em.py 代码正确性检查")
    print("=" * 60)

    tests = [
        ("模拟器基本正确性", test_simulation_basic),
        ("EM 参数回收", test_em_parameter_recovery),
        ("β 网格搜索", test_beta_grid_search),
        ("对数似然一致性", test_loglikelihood),
        ("GOF 残差检验", test_gof_residuals),
        ("fit_1d 端到端", test_fit_1d_convenience),
        ("2D 模拟+EM", test_multidim_simulation_and_em),
        ("估计参数→GOF", test_gof_with_estimated_params),
    ]

    results = []
    for name, func in tests:
        try:
            ok = func()
        except Exception as e:
            print(f"  ✗ 异常: {e}")
            import traceback
            traceback.print_exc()
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    n_pass = 0
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if ok:
            n_pass += 1

    print(f"\n  总计: {n_pass}/{len(results)} 通过")
    print("=" * 60)

    if n_pass < len(results):
        print("\n⚠ 存在失败的测试，请检查 hawkes_em.py")
        sys.exit(1)
    else:
        print("\n✓ 所有测试通过，可以进入四维实盘建模")
        sys.exit(0)


if __name__ == "__main__":
    main()
