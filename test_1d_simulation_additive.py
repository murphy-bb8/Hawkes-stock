"""
一维仿真实验 - 验证加性基线三模型 A < B < C

实验设计：
1. 模拟真实的 Model C 数据（包含常数基线、时变基线、spread外生项）
2. 分别用 Model A、B、C 拟合
3. 比较对数似然：应该观察到 LL_C > LL_B > LL_A
4. 评估参数恢复准确度
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple

# 导入我们的加性基线实现
from hawkes_em_additive import (
    em_additive_baseline,
    loglikelihood_additive,
    gof_residuals_additive,
    fit_4d_additive,
    _flatten_events,
)
from hawkes_em import (
    compute_indicators,
    OPEN30_START, OPEN30_END, MID30_START, MID30_END,
    CLOSE30_START, CLOSE30_END,
    TRADING_SECONDS_PER_DAY,
)

np.random.seed(42)

# ===================== 1D Hawkes 模拟器（Ogata thinning） =====================

def simulate_1d_hawkes_additive(mu: float, alpha: float, omega: float,
                                T: float,
                                gamma_open: float = 0.0,
                                gamma_mid: float = 0.0,
                                gamma_close: float = 0.0,
                                gamma_spread: float = 0.0,
                                spread_func=None,
                                n_days: int = 1,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟一维 Hawkes 过程（加性基线）。
    
    强度函数：
      λ(t) = μ + γ_open·I_open(t) + γ_mid·I_mid(t) + γ_close·I_close(t) 
             + γ_spread·spread(t) + Σ_{t_k < t} α·ω·exp(-ω·(t - t_k))
    
    Returns
    -------
    times : np.ndarray
        事件时间序列（归一化连续时间）
    intraday_times : np.ndarray
        事件对应的日内时间（用于时段判断）
    """
    np.random.seed(seed)
    
    events = []
    intraday = []
    t = 0.0
    r = 0.0  # 激励项递推
    
    # 计算强度上界（用于 thinning）
    lambda_max = mu + max(gamma_open, gamma_mid, gamma_close, 0) * 1.5 + \
                 abs(gamma_spread) * 3.0  # spread 假设在 [-3, 3] 范围
    
    while t < T:
        # 采样候选时间
        lambda_star = lambda_max + r
        if lambda_star <= 0:
            lambda_star = 0.1
        
        u = np.random.exponential(1.0 / lambda_star)
        t_candidate = t + u
        
        if t_candidate > T:
            break
        
        # 计算真实强度
        t_intraday = (t_candidate % TRADING_SECONDS_PER_DAY) + OPEN30_START
        
        # 基线
        lambda_t = mu
        
        # 时变基线
        I_o, I_m, I_c = compute_indicators(t_intraday)
        lambda_t += gamma_open * I_o + gamma_mid * I_m + gamma_close * I_c
        
        # spread 外生项
        if spread_func is not None:
            lambda_t += gamma_spread * spread_func(t_candidate)
        
        # 激励项（递推更新）
        dt = t_candidate - t
        r = r * np.exp(-omega * dt)
        lambda_t += r
        
        # 接受/拒绝
        accept_prob = max(lambda_t, 0) / lambda_star
        if np.random.rand() < accept_prob:
            events.append(t_candidate)
            intraday.append(t_intraday)
            r += alpha * omega  # 事件发生后增加激励
        
        t = t_candidate
    
    return np.array(events), np.array(intraday)


# ===================== 主实验 =====================

def run_1d_simulation():
    """运行一维仿真实验"""
    print("=" * 60)
    print("一维仿真实验 - 验证加性基线三模型 A < B < C")
    print("=" * 60)
    
    # 真实参数（Model C）
    true_mu = 0.15
    true_alpha = 0.55
    true_omega = 3.0
    true_gamma_open = 0.3   # 开盘强度增加
    true_gamma_mid = 0.05   # 午盘效应弱
    true_gamma_close = -0.1  # 收盘略降
    true_gamma_spread = 0.2  # spread 正相关
    
    # 模拟参数
    n_days = 5
    T = TRADING_SECONDS_PER_DAY * n_days
    
    # 构造 spread 函数（正弦波 + 噪声）
    def spread_func(t):
        # 日内周期性 + 随机波动
        period = TRADING_SECONDS_PER_DAY
        phase = (t % period) / period * 2 * np.pi
        return 0.5 * np.sin(phase) + np.random.normal(0, 0.2)
    
    print(f"\n【真实参数】")
    print(f"  μ = {true_mu:.3f}")
    print(f"  α = {true_alpha:.3f}")
    print(f"  ω = {true_omega:.3f}")
    print(f"  γ_open = {true_gamma_open:.3f}")
    print(f"  γ_mid = {true_gamma_mid:.3f}")
    print(f"  γ_close = {true_gamma_close:.3f}")
    print(f"  γ_spread = {true_gamma_spread:.3f}")
    print(f"  分枝比 = {true_alpha:.3f}")
    
    # 模拟数据
    print(f"\n【模拟数据】")
    print(f"  模拟 {n_days} 天数据...")
    events, intraday = simulate_1d_hawkes_additive(
        true_mu, true_alpha, true_omega, T,
        true_gamma_open, true_gamma_mid, true_gamma_close,
        true_gamma_spread, spread_func, n_days, seed=42
    )
    
    N = len(events)
    print(f"  生成事件数: {N}")
    print(f"  平均每天事件数: {N / n_days:.1f}")
    print(f"  时间范围: [{events[0]:.2f}, {events[-1]:.2f}]")
    
    # 构造 spread 值序列
    spread_values = np.array([spread_func(t) for t in events])
    
    # 转换为 4D 格式（单维度）
    events_4d = [events]  # List of 1 array
    intraday_4d = [intraday]
    spread_4d = [spread_values]
    
    # β 网格
    beta_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    # ==================== 拟合三种模型 ====================
    results = {}
    
    for model_name in ["A", "B", "C"]:
        print(f"\n{'='*60}")
        print(f"拟合 Model {model_name}")
        print('='*60)
        
        start_time = time.time()
        
        if model_name == "A":
            result = fit_4d_additive(
                events_4d, T, beta_grid,
                model="A", maxiter=100, verbose=True, n_days=n_days
            )
        elif model_name == "B":
            result = fit_4d_additive(
                events_4d, T, beta_grid,
                model="B", events_4d_intraday=intraday_4d,
                maxiter=100, verbose=True, n_days=n_days
            )
        else:  # Model C
            result = fit_4d_additive(
                events_4d, T, beta_grid,
                model="C", events_4d_intraday=intraday_4d,
                spread_values=spread_4d,
                maxiter=100, verbose=True, n_days=n_days
            )
        
        elapsed = time.time() - start_time
        result["time_s"] = elapsed
        
        results[model_name] = result
        
        print(f"\n【Model {model_name} 拟合结果】")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  最优 β: {result['beta']:.1f}")
        print(f"  μ: {result['mu'][0]:.4f} (真实: {true_mu:.4f})")
        print(f"  α: {result['alpha'][0][0]:.4f} (真实: {true_alpha:.4f})")
        print(f"  分枝比: {result['branching_ratio']:.4f}")
        print(f"  对数似然: {result['loglik']:.2f}")
        print(f"  AIC: {result['aic']:.2f}")
        print(f"  BIC: {result['bic']:.2f}")
        
        if model_name in ["B", "C"]:
            print(f"  γ_open: {result['gamma_open'][0]:.4f} (真实: {true_gamma_open:.4f})")
            print(f"  γ_mid: {result['gamma_mid'][0]:.4f} (真实: {true_gamma_mid:.4f})")
            print(f"  γ_close: {result['gamma_close'][0]:.4f} (真实: {true_gamma_close:.4f})")
        
        if model_name == "C":
            print(f"  γ_spread: {result['gamma_spread'][0]:.4f} (真实: {true_gamma_spread:.4f})")
    
    # ==================== 结果对比 ====================
    print(f"\n{'='*60}")
    print("模型对比总结")
    print('='*60)
    
    print("\n【对数似然对比】（越大越好）")
    for model_name in ["A", "B", "C"]:
        ll = results[model_name]["loglik"]
        print(f"  Model {model_name}: {ll:.2f}")
    
    ll_diff_BA = results["B"]["loglik"] - results["A"]["loglik"]
    ll_diff_CB = results["C"]["loglik"] - results["B"]["loglik"]
    print(f"\n  LL(B) - LL(A) = {ll_diff_BA:.2f}")
    print(f"  LL(C) - LL(B) = {ll_diff_CB:.2f}")
    
    if ll_diff_BA > 0 and ll_diff_CB > 0:
        print("  ✓ 验证通过：LL(C) > LL(B) > LL(A)")
    else:
        print("  ✗ 验证失败：不满足 LL(C) > LL(B) > LL(A)")
    
    print("\n【AIC 对比】（越小越好）")
    for model_name in ["A", "B", "C"]:
        aic = results[model_name]["aic"]
        print(f"  Model {model_name}: {aic:.2f}")
    
    best_aic = min(results[m]["aic"] for m in ["A", "B", "C"])
    best_model_aic = [m for m in ["A", "B", "C"] if results[m]["aic"] == best_aic][0]
    print(f"  AIC 最优模型: Model {best_model_aic}")
    
    print("\n【BIC 对比】（越小越好）")
    for model_name in ["A", "B", "C"]:
        bic = results[model_name]["bic"]
        print(f"  Model {model_name}: {bic:.2f}")
    
    best_bic = min(results[m]["bic"] for m in ["A", "B", "C"])
    best_model_bic = [m for m in ["A", "B", "C"] if results[m]["bic"] == best_bic][0]
    print(f"  BIC 最优模型: Model {best_model_bic}")
    
    # 保存结果
    output_file = "simulation_1d_additive_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "true_parameters": {
                "mu": true_mu,
                "alpha": true_alpha,
                "omega": true_omega,
                "gamma_open": true_gamma_open,
                "gamma_mid": true_gamma_mid,
                "gamma_close": true_gamma_close,
                "gamma_spread": true_gamma_spread,
            },
            "simulation": {
                "n_days": n_days,
                "n_events": N,
                "T": T,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # ==================== 可视化 ====================
    plt.figure(figsize=(14, 8))
    
    # 子图1: 事件时间序列
    plt.subplot(3, 2, 1)
    plt.plot(events, np.zeros_like(events), 'b|', markersize=10, alpha=0.5)
    plt.xlabel('Time (s)')
    plt.title(f'Event Timeline (N={N})')
    plt.ylim([-0.5, 0.5])
    
    # 子图2: 对数似然对比
    plt.subplot(3, 2, 2)
    models = ["A", "B", "C"]
    lls = [results[m]["loglik"] for m in models]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(models, lls, color=colors, alpha=0.7)
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Comparison')
    plt.grid(axis='y', alpha=0.3)
    for i, (m, ll) in enumerate(zip(models, lls)):
        plt.text(i, ll + max(lls) * 0.01, f'{ll:.1f}', ha='center', va='bottom')
    
    # 子图3: AIC 对比
    plt.subplot(3, 2, 3)
    aics = [results[m]["aic"] for m in models]
    plt.bar(models, aics, color=colors, alpha=0.7)
    plt.ylabel('AIC')
    plt.title('AIC Comparison (lower is better)')
    plt.grid(axis='y', alpha=0.3)
    for i, (m, aic) in enumerate(zip(models, aics)):
        plt.text(i, aic + max(aics) * 0.01, f'{aic:.1f}', ha='center', va='bottom')
    
    # 子图4: 参数恢复（μ 和 α）
    plt.subplot(3, 2, 4)
    mu_est = [results[m]["mu"][0] for m in models]
    alpha_est = [results[m]["alpha"][0][0] for m in models]
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, mu_est, width, label='μ (估计)', alpha=0.7)
    plt.bar(x + width/2, alpha_est, width, label='α (估计)', alpha=0.7)
    plt.axhline(true_mu, color='blue', linestyle='--', linewidth=2, label=f'μ (真实={true_mu:.3f})')
    plt.axhline(true_alpha, color='orange', linestyle='--', linewidth=2, label=f'α (真实={true_alpha:.3f})')
    plt.xticks(x, models)
    plt.ylabel('Parameter Value')
    plt.title('Parameter Recovery')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 子图5: γ 参数恢复
    plt.subplot(3, 2, 5)
    gamma_types = ['open', 'mid', 'close']
    true_gammas = [true_gamma_open, true_gamma_mid, true_gamma_close]
    
    x = np.arange(len(gamma_types))
    width = 0.25
    
    # Model B
    gamma_b = [results["B"]["gamma_open"][0], 
               results["B"]["gamma_mid"][0],
               results["B"]["gamma_close"][0]]
    plt.bar(x - width, gamma_b, width, label='Model B', alpha=0.7)
    
    # Model C
    gamma_c = [results["C"]["gamma_open"][0],
               results["C"]["gamma_mid"][0],
               results["C"]["gamma_close"][0]]
    plt.bar(x, gamma_c, width, label='Model C', alpha=0.7)
    
    # 真实值
    plt.bar(x + width, true_gammas, width, label='True', alpha=0.7)
    
    plt.xticks(x, gamma_types)
    plt.ylabel('γ Value')
    plt.title('Time-varying Baseline Parameters')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 子图6: spread 参数（Model C）
    plt.subplot(3, 2, 6)
    gamma_spread_c = results["C"]["gamma_spread"][0]
    plt.bar(['Model C', 'True'], [gamma_spread_c, true_gamma_spread],
            color=['green', 'gray'], alpha=0.7)
    plt.ylabel('γ_spread')
    plt.title('Spread Exogenous Parameter')
    plt.ylim([0, max(gamma_spread_c, true_gamma_spread) * 1.2])
    for i, val in enumerate([gamma_spread_c, true_gamma_spread]):
        plt.text(i, val + max(gamma_spread_c, true_gamma_spread) * 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('simulation_1d_additive_comparison.png', dpi=150, bbox_inches='tight')
    print(f"可视化已保存到: simulation_1d_additive_comparison.png")
    
    return results


if __name__ == "__main__":
    results = run_1d_simulation()
