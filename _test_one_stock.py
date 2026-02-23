"""Quick test: fit one stock with Model A/B/C (chained init)"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawkes_em_additive import (
    fit_hawkes_additive, intraday_to_trading_time,
    TRADING_SECONDS_PER_DAY,
)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "high_price_events")
files = sorted([f for f in os.listdir(data_dir) if f.startswith("events_") and not f.startswith("all_")])
fp = os.path.join(data_dir, files[0])
print("Loading:", fp)
with open(fp, "r", encoding="utf-8") as f:
    raw = json.load(f)

et_keys = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
events_list, intra_list, spread_list = [], [], []

first_et = raw["events"][et_keys[0]]
dates_set = set(d.get("date", "") for d in first_et["days"] if isinstance(d, dict))
n_days = len(dates_set)

for et in et_keys:
    evts = raw["events"][et]
    t_off, t_intra, sp_vals = [], [], []
    days = evts["days"]
    dates = sorted(set(d.get("date", "") for d in days if isinstance(d, dict)))
    d_idx = {d: i for i, d in enumerate(dates)}
    for dd in days:
        if not isinstance(dd, dict) or "t" not in dd:
            continue
        offset = d_idx.get(dd.get("date", ""), 0) * TRADING_SECONDS_PER_DAY
        ts = dd["t"]
        rs = dd.get("re_spread", [0.0] * len(ts))
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
t0 = float(np.min(all_t))
events_list = [e - t0 if len(e) > 0 else e for e in events_list]
T = float(np.max(all_t) - t0)

for i in range(4):
    if len(events_list[i]) > 0:
        idx = np.argsort(events_list[i])
        events_list[i] = events_list[i][idx]
        intra_list[i] = intra_list[i][idx]
        spread_list[i] = spread_list[i][idx]

counts = [len(e) for e in events_list]
code = raw.get("code", "?")
print("code=%s, counts=%s, total=%d, T=%.1f, n_days=%d" % (code, counts, sum(counts), T, n_days))

beta_grid = np.array([1.0, 3.0, 5.0, 10.0])
prev_res = None
for m in ["A", "B", "C"]:
    t_s = time.time()
    r = fit_hawkes_additive(
        events_list, T, beta_grid, model=m, n_days=n_days,
        intraday_list=intra_list if m in ("B", "C") else None,
        spread_list=spread_list if m == "C" else None,
        maxiter=100, epsilon=1e-4, verbose=False,
        init_from=prev_res)
    elapsed = time.time() - t_s
    prev_res = r
    print("Model %s: LL=%.1f  AIC=%.1f  BIC=%.1f  BR=%.4f  GOF=%.3f  omega=%.1f  time=%.1fs" % (
        m, r["loglik"], r["aic"], r["bic"], r["branching_ratio"],
        r["gof_summary"]["mean_gof_score"], r["omega"], elapsed))
    if m == "C" and "gamma_spread" in r:
        print("  gamma_spread=%s" % r["gamma_spread"])

rA_ll = None
rB_ll = None
print("\nMonotonicity check:")
results = {}
prev_res = None
for m in ["A", "B", "C"]:
    r2 = fit_hawkes_additive(
        events_list, T, beta_grid, model=m, n_days=n_days,
        intraday_list=intra_list if m in ("B", "C") else None,
        spread_list=spread_list if m == "C" else None,
        maxiter=100, epsilon=1e-4, verbose=False,
        init_from=prev_res)
    results[m] = r2
    prev_res = r2
ll_a, ll_b, ll_c = results["A"]["loglik"], results["B"]["loglik"], results["C"]["loglik"]
print("LL: A=%.1f  B=%.1f  C=%.1f" % (ll_a, ll_b, ll_c))
print("B-A=%.1f  C-B=%.1f" % (ll_b - ll_a, ll_c - ll_b))
print("Monotonic: %s" % (ll_c >= ll_b >= ll_a))
print("Done.")
