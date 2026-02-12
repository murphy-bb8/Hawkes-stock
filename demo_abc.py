"""
demo_abc.py — 最小可运行 demo: 单只股票 A/B/C 三模型拟合
============================================================
验证:
  1. LL_C >= LL_B >= LL_A (嵌套模型单调性)
  2. AIC/BIC 统一口径比较
  3. GOF 使用同一 log-link 强度函数
  4. SpreadProcess 从所有事件类型收集 (P0-1)

运行:
  conda activate py385 && python demo_abc.py
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_4d_models import load_stock_data, build_4d_events
from hawkes_em import fit_4d, SpreadProcess


def main():
    # --- 配置 ---
    data_dir = os.path.join(os.path.dirname(__file__), "data", "high_price_events")
    # 找一个 per-stock 数据文件 (events_XXXXXX_*.json)
    candidates = sorted(f for f in os.listdir(data_dir)
                        if f.startswith("events_") and f.endswith(".json"))
    if not candidates:
        print("ERROR: 未找到 per-stock 数据文件")
        return
    data_file = os.path.join(data_dir, candidates[0])
    print(f"数据文件: {data_file}")

    stock_data = load_stock_data(data_file)
    built = build_4d_events(stock_data)

    events_4d = built["events"]
    events_orig = built["events_orig"]
    T = built["T"]
    n_days = built["n_days"]
    spread_proc = built.get("spread_proc")
    counts = built["counts"]
    total = sum(counts)

    print(f"事件数: {counts}, 总计={total}, T={T:.1f}s, 天数={n_days}")
    if spread_proc is not None:
        print(f"SpreadProcess: {spread_proc}")
    else:
        print("WARNING: 无 spread 数据，Model C 将退化为 Model B")

    if total < 20:
        print("ERROR: 事件不足, 退出")
        return

    beta_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    maxiter = 80

    results = {}

    for model in ["A", "B", "C"]:
        print(f"\n{'='*60}")
        print(f"  Model {model}")
        print(f"{'='*60}")
        t0 = time.time()
        res = fit_4d(
            events_4d, T, beta_grid,
            model=model,
            events_4d_original=events_orig if model in ("B", "C") else None,
            spread_times=built["spread_times"] if model == "C" else None,
            spread_values=built["spread_vals"] if model == "C" else None,
            spread_proc=spread_proc if model == "C" else None,
            n_days=n_days,
            maxiter=maxiter,
            n_alt=2,
            verbose=True,
        )
        elapsed = time.time() - t0
        res["elapsed_s"] = round(elapsed, 2)
        results[model] = res
        print(f"\n  耗时: {elapsed:.1f}s")

    # === 汇总比较 ===
    print(f"\n\n{'='*60}")
    print("  模型比较 (统一 loglikelihood_loglink 口径)")
    print(f"{'='*60}")
    print(f"{'Model':<8} {'LL':>12} {'AIC':>12} {'BIC':>12} {'k':>5} {'BR':>8} {'GOF_pass':>10} {'Time':>6}")
    print("-" * 75)
    for m in ["A", "B", "C"]:
        r = results[m]
        if "error" in r:
            print(f"{m:<8} ERROR: {r['error']}")
            continue
        f = r["full"]
        gof = r.get("gof", {}).get("summary", {})
        gof_pass = gof.get("gof_pass_count", "?")
        gof_total = 4
        print(f"{m:<8} {f['loglik']:>12.2f} {f['aic']:>12.2f} {f.get('bic', 0):>12.2f} "
              f"{f.get('k_params', '?'):>5} {f['branching_ratio']:>8.4f} "
              f"{gof_pass}/{gof_total:>6} {r.get('elapsed_s', 0):>6.1f}s")

    # === 验证 LL 单调性 ===
    lls = {}
    for m in ["A", "B", "C"]:
        r = results[m]
        if "error" not in r:
            lls[m] = r["full"]["loglik"]

    print(f"\n  LL_A = {lls.get('A', 'N/A')}")
    print(f"  LL_B = {lls.get('B', 'N/A')}")
    print(f"  LL_C = {lls.get('C', 'N/A')}")

    if "A" in lls and "B" in lls:
        ok_ba = lls["B"] >= lls["A"] - 1.0  # 允许数值误差
        print(f"  LL_B >= LL_A ? {'YES' if ok_ba else 'NO *** VIOLATION ***'} "
              f"(diff = {lls['B'] - lls['A']:.2f})")
    if "B" in lls and "C" in lls:
        ok_cb = lls["C"] >= lls["B"] - 1.0
        print(f"  LL_C >= LL_B ? {'YES' if ok_cb else 'NO *** VIOLATION ***'} "
              f"(diff = {lls['C'] - lls['B']:.2f})")

    # === γ 参数解释 ===
    if "B" in results and "gamma" in results["B"]:
        g = results["B"]["gamma"]
        print(f"\n  Model B γ 参数:")
        for name in ["gamma_open", "gamma_mid", "gamma_close"]:
            vals = np.array(g[name])
            print(f"    {name}: {vals.round(3)}, exp(γ)={np.exp(vals).round(3)}")

    if "C" in results and "gamma_spread" in results["C"]:
        gs = np.array(results["C"]["gamma_spread"])
        print(f"\n  Model C γ_spread: {gs.round(4)}")
        print(f"  exp(γ_spread): {np.exp(gs).round(4)}")
        print(f"  解释: spread 每增加 1 std, 基线强度乘以 exp(γ_spread)")
        if "spread_info" in results["C"]:
            si = results["C"]["spread_info"]
            print(f"  spread 原始 mean={si['mean_raw']:.4f}, std={si['std_raw']:.4f}, "
                  f"n_points={si['n_points']}, lag={si['lag']}")

    # === 保存结果 ===
    out_file = os.path.join(os.path.dirname(__file__), "demo_abc_results.json")
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(_convert(results), f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_file}")


if __name__ == "__main__":
    main()
