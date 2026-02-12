"""
实盘数据批量实验：三组 × 三模型，统一 β 网格，收集汇总结果。
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hawkes_em import fit_4d
from run_4d_models import build_4d_events, load_stock_data

DATA_BASE = os.path.join(os.path.dirname(__file__), "data")
GROUPS = {
    "high": os.path.join(DATA_BASE, "high_price_events"),
    "mid":  os.path.join(DATA_BASE, "mid_price_events"),
    "low":  os.path.join(DATA_BASE, "low_price_events"),
}
BETA_GRID = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
MODELS = ["A", "B", "C"]

all_results = {}

t_global = time.time()

for gn, data_dir in GROUPS.items():
    if not os.path.isdir(data_dir):
        print(f"  跳过 {gn}：目录不存在")
        continue

    stock_files = sorted([f for f in os.listdir(data_dir)
                          if f.startswith("events_") and f.endswith(".json")
                          and "all_events" not in f])
    print(f"\n{'='*70}")
    print(f"  {gn.upper()} 组：{len(stock_files)} 只股票")
    print(f"{'='*70}")

    for model in MODELS:
        print(f"\n--- Model {model} ---")
        model_results = []
        t_model = time.time()

        for sf in stock_files:
            code = sf.replace("events_", "").replace("_201912.json", "")
            data = load_stock_data(os.path.join(data_dir, sf))
            built = build_4d_events(data)
            events_4d = built["events"]
            events_orig = built["events_orig"]
            T = built["T"]
            n_days = built.get("n_days", 22)
            sp_proc = built.get("spread_proc")
            total = sum(len(e) for e in events_4d)

            if total < 100:
                print(f"  {code}: 事件太少({total})，跳过")
                continue

            t0 = time.time()
            try:
                result = fit_4d(
                    events_4d, T, BETA_GRID,
                    model=model,
                    events_4d_original=events_orig if model in ("B", "C") else None,
                    spread_proc=sp_proc if model == "C" else None,
                    n_days=n_days,
                    maxiter=80,
                    n_alt=2,
                    verbose=False,
                )
            except Exception as e:
                print(f"  {code}: 拟合失败 ({e})")
                continue
            elapsed = time.time() - t0

            if "error" in result:
                print(f"  {code}: {result['error']}")
                continue

            full = result["full"]
            gof = result.get("gof", {}).get("summary", {})
            gs = result.get("gamma_spread", None)

            rec = {
                "code": code, "model": model, "group": gn,
                "n_events": total, "n_days": n_days,
                "beta": full["decay"],
                "mu": full["mu"],
                "branching_ratio": full["branching_ratio"],
                "loglik": full["loglik"],
                "aic": full["aic"],
                "bic": full.get("bic", float("nan")),
                "gof_score": gof.get("gof_score_mean", 0.0),
                "gof_pass": gof.get("gof_pass_count", 0),
                "gamma_spread": gs,
                "gamma_spread_exp": result.get("gamma_spread_exp"),
                "time_s": round(elapsed, 1),
            }
            if "gamma" in result:
                rec["gamma_open"] = result["gamma"]["gamma_open"]
            model_results.append(rec)

            gs_str = ""
            if gs is not None:
                gs_str = f", γ_sp={np.array(gs).round(3)}"
            print(f"  {code}: β={full['decay']:.1f}, BR={full['branching_ratio']:.4f}, "
                  f"LL={full['loglik']:.0f}, GOF={gof.get('gof_score_mean',0):.3f} "
                  f"({gof.get('gof_pass_count',0)}/4){gs_str} [{elapsed:.1f}s]")

        elapsed_model = time.time() - t_model
        key = f"{gn}_{model}"
        all_results[key] = model_results
        print(f"  Model {model} 完成：{len(model_results)} 只股票，耗时 {elapsed_model:.1f}s")

elapsed_total = time.time() - t_global
print(f"\n{'='*70}")
print(f"  全部完成，总耗时 {elapsed_total:.1f}s")
print(f"{'='*70}")

# 保存原始结果
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("结果已保存到 experiment_results.json")

# ========== 汇总统计 ==========
print(f"\n\n{'='*70}")
print("                    汇 总 统 计")
print(f"{'='*70}")

for gn in GROUPS:
    print(f"\n【{gn.upper()} 组】")
    print(f"{'Model':<8} {'N':>4} {'BR_mean':>8} {'BR_std':>7} {'LL_mean':>12} "
          f"{'AIC_mean':>12} {'GOF_mean':>9} {'GOF_4/4':>8} {'Time':>6}")
    print("-" * 85)
    for model in MODELS:
        key = f"{gn}_{model}"
        recs = all_results.get(key, [])
        if not recs:
            continue
        n = len(recs)
        brs = [r["branching_ratio"] for r in recs]
        lls = [r["loglik"] for r in recs]
        aics = [r["aic"] for r in recs]
        gofs = [r["gof_score"] for r in recs]
        gof4 = sum(1 for r in recs if r["gof_pass"] == 4)
        ts = sum(r["time_s"] for r in recs)
        print(f"  {model:<6} {n:>4} {np.mean(brs):>8.4f} {np.std(brs):>7.4f} "
              f"{np.mean(lls):>12.0f} {np.mean(aics):>12.0f} "
              f"{np.mean(gofs):>9.3f} {gof4:>5}/{n:<2} {ts:>6.0f}s")

# Model C gamma_spread 汇总
print(f"\n\n{'='*70}")
print("           Model C γ_spread 汇总")
print(f"{'='*70}")
print(f"{'Group':<8} {'Code':<10} {'γ_buy_tox':>10} {'γ_buy_not':>10} {'γ_sell_tox':>10} {'γ_sell_not':>10}")
print("-" * 65)
for gn in GROUPS:
    key = f"{gn}_C"
    recs = all_results.get(key, [])
    for r in recs:
        gs = r.get("gamma_spread")
        if gs is not None:
            print(f"  {gn:<6} {r['code']:<10} {gs[0]:>10.4f} {gs[1]:>10.4f} {gs[2]:>10.4f} {gs[3]:>10.4f}")

# 跨模型 LL/AIC 比较
print(f"\n\n{'='*70}")
print("           跨模型 LL / AIC 比较（逐股票）")
print(f"{'='*70}")
for gn in GROUPS:
    print(f"\n【{gn.upper()} 组】")
    print(f"{'Code':<10} {'LL_A':>10} {'LL_B':>10} {'LL_C':>10} {'AIC_A':>12} {'AIC_B':>12} {'AIC_C':>12} {'Best':>5}")
    print("-" * 85)
    recs_a = {r["code"]: r for r in all_results.get(f"{gn}_A", [])}
    recs_b = {r["code"]: r for r in all_results.get(f"{gn}_B", [])}
    recs_c = {r["code"]: r for r in all_results.get(f"{gn}_C", [])}
    all_codes = sorted(set(list(recs_a.keys()) + list(recs_b.keys()) + list(recs_c.keys())))
    for code in all_codes:
        ra = recs_a.get(code)
        rb = recs_b.get(code)
        rc = recs_c.get(code)
        ll_a = ra["loglik"] if ra else float("nan")
        ll_b = rb["loglik"] if rb else float("nan")
        ll_c = rc["loglik"] if rc else float("nan")
        aic_a = ra["aic"] if ra else float("nan")
        aic_b = rb["aic"] if rb else float("nan")
        aic_c = rc["aic"] if rc else float("nan")
        aics = {"A": aic_a, "B": aic_b, "C": aic_c}
        best = min(aics, key=lambda k: aics[k])
        print(f"  {code:<8} {ll_a:>10.0f} {ll_b:>10.0f} {ll_c:>10.0f} "
              f"{aic_a:>12.0f} {aic_b:>12.0f} {aic_c:>12.0f} {best:>5}")
