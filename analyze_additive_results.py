"""分析 additive baseline 实盘实验结果"""
import json, numpy as np, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results_additive")

with open(os.path.join(OUT, "small_test_results.json"), encoding="utf-8") as f:
    sm = json.load(f)
print("=== Small Test (1 stock per group) ===")
for s in sm:
    a, b, c = s["A"], s["B"], s["C"]
    print("%s %s  LL: A=%12.1f  B=%12.1f  C=%12.1f" % (
        s["group"].upper(), s["code"], a["LL"], b["LL"], c["LL"]))
    print("  AIC: A=%12.1f  B=%12.1f  C=%12.1f" % (a["AIC"], b["AIC"], c["AIC"]))
    print("  B-A=%+.1f  C-B=%+.1f  Mono=%s" % (
        b["LL"] - a["LL"], c["LL"] - b["LL"],
        c["LL"] >= b["LL"] >= a["LL"]))
    aic_best = min("ABC", key=lambda m: s[m]["AIC"])
    bic_best = min("ABC", key=lambda m: s[m]["BIC"])
    print("  AIC_best=%s  BIC_best=%s" % (aic_best, bic_best))

print("\n=== Full Experiment (45 stocks) ===")
with open(os.path.join(OUT, "experiment_results.json"), encoding="utf-8") as f:
    full = json.load(f)

by_code = {}
for r in full:
    code = r["code"]
    model = r["model"]
    by_code.setdefault(code, {})[model] = r

mono_count = aic_c = bic_c = aic_b = bic_b = total = 0
for code, models in sorted(by_code.items()):
    if not all(m in models for m in "ABC"):
        continue
    la = models["A"]["loglik"]
    lb = models["B"]["loglik"]
    lc = models["C"]["loglik"]
    mono = lc >= lb >= la
    ab = min("ABC", key=lambda m: models[m]["aic"])
    bb = min("ABC", key=lambda m: models[m]["bic"])
    if mono:
        mono_count += 1
    if ab == "C":
        aic_c += 1
    if ab == "B":
        aic_b += 1
    if bb == "C":
        bic_c += 1
    if bb == "B":
        bic_b += 1
    total += 1

print("Total stocks: %d" % total)
print("LL Monotonic (C>=B>=A): %d/%d" % (mono_count, total))
print("AIC best: A=%d  B=%d  C=%d" % (total - aic_b - aic_c, aic_b, aic_c))
print("BIC best: A=%d  B=%d  C=%d" % (total - bic_b - bic_c, bic_b, bic_c))

for g in ["high", "mid", "low"]:
    codes = [c for c, m in by_code.items()
             if m.get("A", {}).get("group") == g and all(x in m for x in "ABC")]
    if not codes:
        continue
    mono = sum(1 for c in codes
               if by_code[c]["C"]["loglik"] >= by_code[c]["B"]["loglik"] >= by_code[c]["A"]["loglik"])
    ac = sum(1 for c in codes if min("ABC", key=lambda m: by_code[c][m]["aic"]) == "C")
    bc = sum(1 for c in codes if min("ABC", key=lambda m: by_code[c][m]["bic"]) == "C")
    ba = np.mean([by_code[c]["B"]["loglik"] - by_code[c]["A"]["loglik"] for c in codes])
    cb = np.mean([by_code[c]["C"]["loglik"] - by_code[c]["B"]["loglik"] for c in codes])
    br = np.mean([by_code[c]["C"]["branching_ratio"] for c in codes])
    gof = np.mean([by_code[c]["C"]["gof_summary"]["mean_gof_score"] for c in codes])
    print("  %s: n=%d mono=%d/%d AIC_C=%d BIC_C=%d B-A=%+.1f C-B=%+.1f BR=%.4f GOF=%.3f" % (
        g.upper(), len(codes), mono, len(codes), ac, bc, ba, cb, br, gof))

print("\nSample gamma_spread values (Model C):")
for code, models in sorted(by_code.items()):
    if "C" in models and "gamma_spread" in models["C"]:
        gs = models["C"]["gamma_spread"]
        g = models["C"].get("group", "?")
        print("  %s %s: gamma_spread=%s" % (g, code, ["%.4e" % v for v in gs]))
        break
