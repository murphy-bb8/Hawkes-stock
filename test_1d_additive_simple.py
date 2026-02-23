"""
1D / 4D 仿真实验 — 验证加性基线三模型 LL: C > B > A
并输出结果到 results_additive/ 目录
"""
import sys, os, json, time
import numpy as np

# 确保当前目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawkes_em_additive import (
    simulate_hawkes_additive, run_abc_comparison,
    TRADING_SECONDS_PER_DAY, flatten_events,
)

np.random.seed(42)

# ===================== 可视化 =====================

def plot_comparison(results, true_params, save_path, title_prefix="1D"):
    """绘制三模型对比图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skip plot")
        return

    models = ["A", "B", "C"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{title_prefix} Hawkes Additive Baseline: Model Comparison", fontsize=14)

    # 1. LL
    ax = axes[0, 0]
    lls = [results[m]["loglik"] for m in models]
    bars = ax.bar(models, lls, color=colors, alpha=0.75)
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Log-Likelihood (higher=better)")
    for b, v in zip(bars, lls):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 2. AIC
    ax = axes[0, 1]
    aics = [results[m]["aic"] for m in models]
    bars = ax.bar(models, aics, color=colors, alpha=0.75)
    ax.set_ylabel("AIC")
    ax.set_title("AIC (lower=better)")
    for b, v in zip(bars, aics):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 3. BIC
    ax = axes[0, 2]
    bics = [results[m]["bic"] for m in models]
    bars = ax.bar(models, bics, color=colors, alpha=0.75)
    ax.set_ylabel("BIC")
    ax.set_title("BIC (lower=better)")
    for b, v in zip(bars, bics):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 4. mu & alpha recovery
    dim = len(results["A"]["mu"])
    ax = axes[1, 0]
    mu_est = {m: results[m]["mu"] for m in models}
    x = np.arange(dim)
    w = 0.2
    for idx, m in enumerate(models):
        ax.bar(x + idx * w, mu_est[m], w, label=f"Model {m}", alpha=0.75, color=colors[idx])
    if "mu" in true_params:
        for i in range(dim):
            ax.axhline(true_params["mu"][i], color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"d{i}" for i in range(dim)])
    ax.set_ylabel("mu")
    ax.set_title("Baseline mu")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 5. gamma_open recovery
    ax = axes[1, 1]
    for idx, m in enumerate(["B", "C"]):
        if "gamma_open" in results[m]:
            go = results[m]["gamma_open"]
            ax.bar(x + idx * w, go, w, label=f"Model {m}", alpha=0.75, color=colors[idx + 1])
    if "gamma_open" in true_params:
        for i in range(dim):
            ax.axhline(true_params["gamma_open"][i], color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([f"d{i}" for i in range(dim)])
    ax.set_ylabel("gamma")
    ax.set_title("gamma_open (B/C)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 6. GOF score
    ax = axes[1, 2]
    gof_scores = []
    for m in models:
        gs = results[m].get("gof_summary", {}).get("mean_gof_score", 0)
        gof_scores.append(gs)
    bars = ax.bar(models, gof_scores, color=colors, alpha=0.75)
    ax.set_ylabel("GOF Score")
    ax.set_title("Mean GOF Score (higher=better)")
    ax.set_ylim([0, 1.05])
    for b, v in zip(bars, gof_scores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {save_path}")


def print_table(results, true_params=None):
    """打印表格式对比"""
    models = ["A", "B", "C"]
    print("\n" + "=" * 80)
    print(f"{'Metric':<22} {'Model A':>16} {'Model B':>16} {'Model C':>16}")
    print("-" * 80)

    rows = [
        ("Log-Likelihood", "loglik", ".2f"),
        ("AIC", "aic", ".2f"),
        ("BIC", "bic", ".2f"),
        ("Branching Ratio", "branching_ratio", ".4f"),
        ("N params", "n_params", "d"),
        ("N events", "n_events", "d"),
        ("EM iterations", "n_iter", "d"),
        ("omega (best)", "omega", ".2f"),
        ("Elapsed (s)", "elapsed_s", ".1f"),
    ]
    for label, key, fmt in rows:
        vals = []
        for m in models:
            v = results[m].get(key, "N/A")
            if isinstance(v, (int, float)):
                vals.append(f"{v:{fmt}}")
            else:
                vals.append(str(v))
        print(f"{label:<22} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

    # GOF
    for m in models:
        gs = results[m].get("gof_summary", {})
        results[m]["_gof_score"] = gs.get("mean_gof_score", 0)
        results[m]["_gof_pass"] = f"{gs.get('pass_count', 0)}/{gs.get('total', 0)}"
    print(f"{'GOF Score':<22} {results['A']['_gof_score']:>16.3f} "
          f"{results['B']['_gof_score']:>16.3f} {results['C']['_gof_score']:>16.3f}")
    print(f"{'GOF Pass':<22} {results['A']['_gof_pass']:>16} "
          f"{results['B']['_gof_pass']:>16} {results['C']['_gof_pass']:>16}")

    # mu
    dim = len(results["A"]["mu"])
    for i in range(dim):
        label = f"mu[{i}]"
        true_str = f" (true={true_params['mu'][i]:.3f})" if true_params and "mu" in true_params else ""
        vals = [f"{results[m]['mu'][i]:.4f}" for m in models]
        print(f"{label + true_str:<22} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

    # gamma
    for key_name in ["gamma_open", "gamma_mid", "gamma_close", "gamma_spread"]:
        if key_name not in results["C"]:
            continue
        for i in range(dim):
            label = f"{key_name}[{i}]"
            true_str = ""
            if true_params and key_name in true_params:
                true_str = f" (t={true_params[key_name][i]:.3f})"
            vals = []
            for m in models:
                v = results[m].get(key_name)
                vals.append(f"{v[i]:.4f}" if v is not None else "---")
            print(f"{label + true_str:<22} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

    print("-" * 80)
    comp = results.get("comparison", {})
    print(f"LL(B)-LL(A) = {comp.get('LL_B_minus_A', 0):.4f}")
    print(f"LL(C)-LL(B) = {comp.get('LL_C_minus_B', 0):.4f}")
    print(f"LL monotonic (C>=B>=A): {comp.get('LL_monotonic', False)}")
    print(f"AIC best: Model {comp.get('AIC_best', '?')}")
    print(f"BIC best: Model {comp.get('BIC_best', '?')}")
    print("=" * 80)


# ===================== 1D 仿真 =====================

def run_1d_simulation(out_dir):
    print("\n" + "#" * 60)
    print("  1D Simulation: Verify LL(C) > LL(B) > LL(A)")
    print("#" * 60)

    # 真实参数 (Model C)
    true_mu = np.array([0.15])
    true_alpha = np.array([[0.55]])
    true_omega = 3.0
    true_gamma_open = np.array([0.30])
    true_gamma_mid = np.array([0.05])
    true_gamma_close = np.array([0.10])   # 正值: 收盘加速
    true_gamma_spread = np.array([0.15])

    n_days = 5
    T = TRADING_SECONDS_PER_DAY * n_days

    def spread_func(t):
        period = TRADING_SECONDS_PER_DAY
        phase = (t % period) / period * 2 * np.pi
        return 0.5 * np.sin(phase) + 0.3 * np.cos(2 * phase)

    print(f"  True: mu={true_mu[0]:.3f}, alpha={true_alpha[0,0]:.3f}, omega={true_omega}")
    print(f"  gamma_open={true_gamma_open[0]:.3f}, gamma_mid={true_gamma_mid[0]:.3f}, "
          f"gamma_close={true_gamma_close[0]:.3f}, gamma_spread={true_gamma_spread[0]:.3f}")
    print(f"  Simulating {n_days} days (T={T:.0f}s)...")

    events_list, intraday_list, spread_list = simulate_hawkes_additive(
        dim=1, mu=true_mu, alpha=true_alpha, omega=true_omega,
        T=T, n_days=n_days,
        gamma_open=true_gamma_open, gamma_mid=true_gamma_mid, gamma_close=true_gamma_close,
        gamma_spread=true_gamma_spread, spread_func=spread_func, seed=42)

    N = sum(len(ev) for ev in events_list)
    print(f"  Generated {N} events")

    beta_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    results = run_abc_comparison(
        events_list, T, beta_grid, n_days=n_days,
        intraday_list=intraday_list, spread_list=spread_list,
        maxiter=150, verbose=True)

    true_params = {
        "mu": true_mu.tolist(), "alpha": true_alpha.tolist(),
        "gamma_open": true_gamma_open.tolist(), "gamma_mid": true_gamma_mid.tolist(),
        "gamma_close": true_gamma_close.tolist(), "gamma_spread": true_gamma_spread.tolist(),
    }
    results["true_params"] = true_params

    print_table(results, true_params)

    # Save
    with open(os.path.join(out_dir, "sim_1d_results.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if k != "comparison" or True},
                  f, indent=2, ensure_ascii=False, default=str)

    plot_comparison(results, true_params, os.path.join(out_dir, "sim_1d_comparison.png"), "1D")
    return results


# ===================== 4D 仿真 =====================

def run_4d_simulation(out_dir):
    print("\n" + "#" * 60)
    print("  4D Simulation: Toxic/Non-Toxic Order Flow")
    print("#" * 60)

    dim = 4
    labels = ["BT", "BN", "ST", "SN"]

    # 真实参数 (仿照 README 中 High 组结构)
    true_mu = np.array([0.08, 0.12, 0.07, 0.11])
    true_alpha = np.array([
        [0.45, 0.00, 0.00, 0.02],
        [0.00, 0.35, 0.02, 0.00],
        [0.00, 0.02, 0.44, 0.00],
        [0.02, 0.00, 0.00, 0.36],
    ])
    true_omega = 3.0
    true_gamma_open = np.array([0.25, 0.15, 0.22, 0.12])
    true_gamma_mid = np.array([0.02, 0.01, 0.02, 0.01])
    true_gamma_close = np.array([0.05, 0.08, 0.04, 0.07])
    true_gamma_spread = np.array([0.10, 0.02, 0.12, 0.03])

    n_days = 3
    T = TRADING_SECONDS_PER_DAY * n_days

    def spread_func(t):
        period = TRADING_SECONDS_PER_DAY
        phase = (t % period) / period * 2 * np.pi
        return 0.4 * np.sin(phase) + 0.2 * np.cos(3 * phase)

    print(f"  dim=4: {labels}")
    print(f"  Simulating {n_days} days (T={T:.0f}s)...")

    events_list, intraday_list, spread_list = simulate_hawkes_additive(
        dim=dim, mu=true_mu, alpha=true_alpha, omega=true_omega,
        T=T, n_days=n_days,
        gamma_open=true_gamma_open, gamma_mid=true_gamma_mid, gamma_close=true_gamma_close,
        gamma_spread=true_gamma_spread, spread_func=spread_func, seed=123)

    for d in range(dim):
        print(f"    {labels[d]}: {len(events_list[d])} events")

    beta_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    results = run_abc_comparison(
        events_list, T, beta_grid, n_days=n_days,
        intraday_list=intraday_list, spread_list=spread_list,
        maxiter=150, verbose=True)

    true_params = {
        "mu": true_mu.tolist(), "alpha": true_alpha.tolist(),
        "gamma_open": true_gamma_open.tolist(), "gamma_mid": true_gamma_mid.tolist(),
        "gamma_close": true_gamma_close.tolist(), "gamma_spread": true_gamma_spread.tolist(),
    }
    results["true_params"] = true_params

    print_table(results, true_params)

    with open(os.path.join(out_dir, "sim_4d_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    plot_comparison(results, true_params, os.path.join(out_dir, "sim_4d_comparison.png"), "4D")
    return results


# ===================== main =====================

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_additive")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    r1 = run_1d_simulation(out_dir)
    r4 = run_4d_simulation(out_dir)

    print("\n\nDONE. All results saved to results_additive/")
