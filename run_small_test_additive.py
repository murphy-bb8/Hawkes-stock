"""
小样本快速测试：每组取1只股票，验证 LL(C) >= LL(B) >= LL(A)
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawkes_em_additive import (
    fit_hawkes_additive, TRADING_SECONDS_PER_DAY,
    intraday_to_trading_time, _USE_CYTHON,
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "results_additive")
BETA_GRID = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0])
GROUP_NAMES = ["high", "mid", "low"]

print("=" * 70)
print("  Small-sample additive EM test")
print("  Cython: %s" % ("ENABLED" if _USE_CYTHON else "DISABLED"))
print("=" * 70)


def load_and_build(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    et_keys = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    events_list, intra_list, spread_list = [], [], []
    first_et = raw.get("events", {}).get(et_keys[0], {})
    dates_set = set()
    if isinstance(first_et, dict) and "days" in first_et:
        dates_set = set(d.get("date", "") for d in first_et["days"] if isinstance(d, dict))
    n_days = max(len(dates_set), 1)
    for et in et_keys:
        evts = raw.get("events", {}).get(et, {})
        t_off, t_intra, sp_vals = [], [], []
        if isinstance(evts, dict) and "days" in evts:
            days = evts["days"]
            dates = sorted(set(d.get("date", "") for d in days if isinstance(d, dict)))
            d_idx = {d: i for i, d in enumerate(dates)}
            for dd in days:
                if not isinstance(dd, dict) or "t" not in dd:
                    continue
                offset = d_idx.get(dd.get("date", ""), 0) * TRADING_SECONDS_PER_DAY
                ts = dd["t"]
                rs = dd.get("re_spread", [None] * len(ts))
                for ti, ri in zip(ts, rs):
                    if not isinstance(ti, (int, float)):
                        continue
                    t_id = float(ti)
                    t_off.append(offset + intraday_to_trading_time(t_id))
                    t_intra.append(t_id)
                    sp_val = float(ri) if isinstance(ri, (int, float)) else 0.0
                    sp_vals.append(sp_val)
        events_list.append(np.asarray(t_off, dtype=float))
        intra_list.append(np.asarray(t_intra, dtype=float))
        spread_list.append(np.asarray(sp_vals, dtype=float))
    all_t = np.concatenate([e for e in events_list if len(e) > 0])
    if len(all_t) == 0:
        return None
    t0 = float(np.min(all_t))
    events_list = [e - t0 if len(e) > 0 else e for e in events_list]
    T = float(np.max(all_t) - t0)
    for i in range(4):
        if len(events_list[i]) > 0:
            idx = np.argsort(events_list[i])
            events_list[i] = events_list[i][idx]
            intra_list[i] = intra_list[i][idx]
            spread_list[i] = spread_list[i][idx]
    return {
        "events": events_list, "intraday": intra_list, "spread": spread_list,
        "T": T, "n_days": n_days, "code": raw.get("code", ""),
    }


all_results = []
t_start = time.time()

for g in GROUP_NAMES:
    data_dir = os.path.join(DATA_DIR, "%s_price_events" % g)
    if not os.path.isdir(data_dir):
        print("[WARN] missing %s" % data_dir)
        continue
    stock_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith("events_") and f.endswith(".json") and not f.startswith("all_")])
    # 取前3只股票做测试
    test_files = stock_files[:3]
    print("\nGROUP: %s  (testing %d / %d stocks)" % (g.upper(), len(test_files), len(stock_files)))

    for sf in test_files:
        code = sf.replace("events_", "").replace("_201912.json", "").replace(".json", "")
        built = load_and_build(os.path.join(data_dir, sf))
        if built is None:
            print("  [%s] SKIP empty" % code)
            continue
        counts = [len(e) for e in built["events"]]
        total = sum(counts)
        print("  [%s] events=%s total=%d days=%d T=%.0f" % (
            code, counts, total, built["n_days"], built["T"]))

        prev_res = None
        stock_results = {"code": code, "group": g}
        for m in ["A", "B", "C"]:
            t0 = time.time()
            try:
                r = fit_hawkes_additive(
                    built["events"], built["T"], BETA_GRID, model=m,
                    n_days=built["n_days"],
                    intraday_list=built["intraday"] if m in ("B", "C") else None,
                    spread_list=built["spread"] if m == "C" else None,
                    maxiter=200, epsilon=1e-5, verbose=False,
                    init_from=prev_res)
                elapsed = time.time() - t0
                prev_res = r
                stock_results[m] = {
                    "LL": r["loglik"], "AIC": r["aic"], "BIC": r["bic"],
                    "BR": r["branching_ratio"],
                    "GOF": r["gof_summary"]["mean_gof_score"],
                    "omega": r["omega"], "time": elapsed,
                }
                print("    Model %s: LL=%12.1f  AIC=%12.1f  BIC=%12.1f  BR=%.4f  GOF=%.3f  omega=%.0f  %.1fs" % (
                    m, r["loglik"], r["aic"], r["bic"],
                    r["branching_ratio"], r["gof_summary"]["mean_gof_score"],
                    r["omega"], elapsed))
            except Exception as e:
                elapsed = time.time() - t0
                print("    Model %s: ERROR %s (%.1fs)" % (m, e, elapsed))
                stock_results[m] = {"error": str(e)}

        # LL 单调性检查
        if all(m in stock_results and "LL" in stock_results[m] for m in "ABC"):
            ll_a = stock_results["A"]["LL"]
            ll_b = stock_results["B"]["LL"]
            ll_c = stock_results["C"]["LL"]
            mono_ba = ll_b >= ll_a
            mono_cb = ll_c >= ll_b
            print("    LL check: B-A=%+.1f (%s)  C-B=%+.1f (%s)  Monotonic=%s" % (
                ll_b - ll_a, "OK" if mono_ba else "FAIL",
                ll_c - ll_b, "OK" if mono_cb else "FAIL",
                "YES" if (mono_ba and mono_cb) else "NO"))
            # AIC/BIC 最优
            aic_best = min("ABC", key=lambda x: stock_results[x]["AIC"])
            bic_best = min("ABC", key=lambda x: stock_results[x]["BIC"])
            print("    AIC best: %s  BIC best: %s" % (aic_best, bic_best))
        all_results.append(stock_results)

elapsed_total = time.time() - t_start
print("\n" + "=" * 70)
print("  Total elapsed: %.1fs" % elapsed_total)
print("=" * 70)

# 汇总
print("\nSummary:")
print("%-6s %-6s %12s %12s %12s %8s %6s %6s" % (
    "Group", "Code", "LL(A)", "LL(B)", "LL(C)", "BR(C)", "GOF(C)", "Mono"))
print("-" * 78)
for sr in all_results:
    if all(m in sr and "LL" in sr[m] for m in "ABC"):
        mono = "YES" if sr["C"]["LL"] >= sr["B"]["LL"] >= sr["A"]["LL"] else "NO"
        print("%-6s %-6s %12.1f %12.1f %12.1f %8.4f %6.3f %6s" % (
            sr["group"].upper(), sr["code"],
            sr["A"]["LL"], sr["B"]["LL"], sr["C"]["LL"],
            sr["C"]["BR"], sr["C"]["GOF"], mono))

# 保存结果
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "small_test_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
print("\nResults saved to %s" % os.path.join(OUT_DIR, "small_test_results.json"))
