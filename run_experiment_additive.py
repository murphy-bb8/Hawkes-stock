"""
run_experiment_additive.py  --  加性基线 EM 实盘实验
===============================================
45 只 A 股 x 3 模型 (A/B/C)，结果输出到 results_additive/
"""
import os, sys, json, time, traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import kstest, expon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawkes_em_additive import (
    fit_hawkes_additive, flatten_events, gof_residuals,
    TRADING_SECONDS_PER_DAY, intraday_to_trading_time,
    get_period_tt, PERIOD_OPEN, PERIOD_MID, PERIOD_CLOSE, PERIOD_NORMAL,
    _USE_CYTHON,
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "results_additive")
BETA_GRID = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0])
DIM_NAMES = ["BT", "BN", "ST", "SN"]
DIM_FULL = ["Buy Toxic", "Buy Non-Toxic", "Sell Toxic", "Sell Non-Toxic"]
GROUP_NAMES = ["high", "mid", "low"]
GROUP_COLORS = {"high": "#e74c3c", "mid": "#f39c12", "low": "#27ae60"}
MODEL_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}


# ===================== 数据加载 =====================

def load_and_build(path: str):
    """加载 JSON 并构建 4D 事件/日内时间/spread 列表"""
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


# ===================== 单股拟合 =====================

def fit_stock(code, built, model, init_from=None):
    """对单只股票拟合一个模型, 返回 result dict"""
    events = built["events"]
    T = built["T"]
    n_days = built["n_days"]
    counts = [len(e) for e in events]
    total = sum(counts)
    if total < 50:
        return {"code": code, "model": model, "error": "insufficient"}

    try:
        r = fit_hawkes_additive(
            events, T, BETA_GRID, model=model, n_days=n_days,
            intraday_list=built["intraday"] if model in ("B", "C") else None,
            spread_list=built["spread"] if model == "C" else None,
            maxiter=200, epsilon=1e-5, verbose=False,
            init_from=init_from)
        r["code"] = code
        r["n_events_per_dim"] = counts
        return r
    except Exception as e:
        traceback.print_exc()
        return {"code": code, "model": model, "error": str(e)}


# ===================== 论文级可视化 =====================

def plot_paper_figures(all_res, out_dir):
    """生成全套论文级别可视化"""
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

    # ---------- 1. LL / AIC / BIC 三模型对比柱状图 ----------
    print("  [VIZ] LL/AIC/BIC comparison bars ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for mi, metric in enumerate(["loglik", "aic", "bic"]):
        ax = axes[mi]
        x = np.arange(3)
        width = 0.25
        for idx_m, m in enumerate(["A", "B", "C"]):
            vals = []
            for g in GROUP_NAMES:
                ok = [r for r in all_res.get((g, m), []) if "error" not in r]
                vals.append(np.mean([r[metric] for r in ok]) if ok else 0)
            ax.bar(x + idx_m * width, vals, width, label="Model %s" % m,
                   color=MODEL_COLORS[m], alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels([g.upper() for g in GROUP_NAMES])
        label = {"loglik": "Log-Likelihood", "aic": "AIC", "bic": "BIC"}[metric]
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Model Comparison: LL / AIC / BIC (Additive Baseline EM)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison_ll_aic_bic.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 2. Branching Ratio 箱线图 ----------
    print("  [VIZ] Branching ratio boxplot ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    positions, tick_labels, bp_data = [], [], []
    pos = 0
    for g in GROUP_NAMES:
        for m in ["A", "B", "C"]:
            ok = [r for r in all_res.get((g, m), []) if "error" not in r]
            brs = [r["branching_ratio"] for r in ok]
            bp_data.append(brs if brs else [0])
            positions.append(pos)
            tick_labels.append("%s-%s" % (g[0].upper(), m))
            pos += 1
        pos += 0.5
    bp = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True)
    cols = [MODEL_COLORS[m] for _ in GROUP_NAMES for m in ["A", "B", "C"]]
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
    ax.axhline(1.0, color="red", ls="--", lw=1.5, label="BR=1 (stability)")
    ax.set_ylabel("Branching Ratio")
    ax.set_title("Branching Ratio by Group and Model", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "branching_ratio_boxplot.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 3. GOF Score 热力图 ----------
    print("  [VIZ] GOF score heatmaps ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx_m, m in enumerate(["A", "B", "C"]):
        ax = axes[idx_m]
        data = np.zeros((3, 4))
        for gi, g in enumerate(GROUP_NAMES):
            ok = [r for r in all_res.get((g, m), []) if "error" not in r]
            for d in range(4):
                scores = [r["gof_details"].get(str(d), {}).get("gof_score", 0) for r in ok
                          if str(d) in r.get("gof_details", {})]
                data[gi, d] = np.mean(scores) if scores else 0
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
        for i in range(3):
            for j in range(4):
                ax.text(j, i, "%.3f" % data[i, j], ha="center", va="center", fontsize=10,
                        color="white" if data[i, j] < 0.6 else "black")
        ax.set_xticks(range(4))
        ax.set_xticklabels(DIM_NAMES)
        ax.set_yticks(range(3))
        ax.set_yticklabels([g.upper() for g in GROUP_NAMES])
        ax.set_title("Model %s" % m, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle("GOF Score Heatmap (Additive Baseline EM)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gof_score_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 4. Excitation Matrix 热力图 ----------
    print("  [VIZ] Excitation matrix heatmaps ...")
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    im_last = None
    for idx, g in enumerate(GROUP_NAMES):
        ax = fig.add_subplot(gs[0, idx])
        ok = [r for r in all_res.get((g, "C"), []) if "error" not in r]
        if ok:
            As = np.array([r["alpha"] for r in ok])
            A_mean = np.mean(As, axis=0)
            vmax = max(np.max(A_mean), 0.5)
            im_last = ax.imshow(A_mean, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
            for i in range(4):
                for j in range(4):
                    c = "white" if A_mean[i, j] > vmax * 0.5 else "black"
                    ax.text(j, i, "%.3f" % A_mean[i, j], ha="center", va="center", color=c, fontsize=9)
            ax.set_xticks(range(4))
            ax.set_xticklabels(DIM_NAMES, fontsize=9)
            ax.set_yticks(range(4))
            ax.set_yticklabels(DIM_NAMES, fontsize=9)
        ax.set_title("%s (n=%d)" % (g.upper(), len(ok)), fontweight="bold")
    if im_last:
        cbar_ax = fig.add_subplot(gs[0, 3])
        fig.colorbar(im_last, cax=cbar_ax).set_label(r"$\alpha_{ij}$")
    plt.suptitle("Excitation Matrix (Model C, Additive Baseline EM)", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(out_dir, "excitation_matrix_heatmaps.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 5. Gamma 参数条形图 ----------
    print("  [VIZ] Gamma bar plots ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for gi, g in enumerate(GROUP_NAMES):
        ax = axes[gi]
        ok = [r for r in all_res.get((g, "C"), []) if "error" not in r and "gamma_open" in r]
        if not ok:
            continue
        go = np.mean([r["gamma_open"] for r in ok], axis=0)
        gm = np.mean([r["gamma_mid"] for r in ok], axis=0)
        gc = np.mean([r["gamma_close"] for r in ok], axis=0)
        x = np.arange(4)
        w = 0.25
        ax.bar(x - w, go, w, label="OPEN30", color="#e74c3c", alpha=0.8)
        ax.bar(x, gm, w, label="MID30", color="#3498db", alpha=0.8)
        ax.bar(x + w, gc, w, label="CLOSE30", color="#2ecc71", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_NAMES)
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(g.upper(), fontweight="bold")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(r"Intraday Effect $\gamma$ (Model C, Additive Baseline)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gamma_bar_modelC.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 6. LL 单调性散点图 ----------
    print("  [VIZ] LL monotonicity scatter ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for g in GROUP_NAMES:
        okA = {r["code"]: r for r in all_res.get((g, "A"), []) if "error" not in r}
        okB = {r["code"]: r for r in all_res.get((g, "B"), []) if "error" not in r}
        okC = {r["code"]: r for r in all_res.get((g, "C"), []) if "error" not in r}
        codes = sorted(set(okA) & set(okB) & set(okC))
        llA = [okA[c]["loglik"] for c in codes]
        llB = [okB[c]["loglik"] for c in codes]
        llC = [okC[c]["loglik"] for c in codes]
        axes[0].scatter(llA, llB, color=GROUP_COLORS[g], label=g.upper(), alpha=0.7, s=40, edgecolors="k", lw=0.5)
        axes[1].scatter(llB, llC, color=GROUP_COLORS[g], label=g.upper(), alpha=0.7, s=40, edgecolors="k", lw=0.5)
    for ax, xl, yl in [(axes[0], "LL(A)", "LL(B)"), (axes[1], "LL(B)", "LL(C)")]:
        lims = [ax.get_xlim(), ax.get_ylim()]
        lo = min(lims[0][0], lims[1][0])
        hi = max(lims[0][1], lims[1][1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y=x")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title("%s vs %s" % (yl, xl), fontweight="bold")
    plt.suptitle("LL Monotonicity: B>A, C>B (Additive Baseline)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ll_monotonicity_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 7. Gamma Spread 对比条形图 (Model C) ----------
    print("  [VIZ] Gamma spread bar ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(4)
    w = 0.25
    for gi, g in enumerate(GROUP_NAMES):
        ok = [r for r in all_res.get((g, "C"), []) if "error" not in r and "gamma_spread" in r]
        if ok:
            gs_mean = np.mean([r["gamma_spread"] for r in ok], axis=0)
            ax.bar(x + gi * w, gs_mean, w, label=g.upper(), color=GROUP_COLORS[g], alpha=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(DIM_NAMES)
    ax.set_ylabel(r"$\gamma_{spread}$ (normalized)")
    ax.set_title("Spread Effect by Group (Model C)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gamma_spread_bar.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 8. AIC/BIC 改进量 (B-A, C-B) ----------
    print("  [VIZ] AIC/BIC improvement ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for mi, metric in enumerate(["aic", "bic"]):
        ax = axes[mi]
        for gi, g in enumerate(GROUP_NAMES):
            okA = {r["code"]: r for r in all_res.get((g, "A"), []) if "error" not in r}
            okB = {r["code"]: r for r in all_res.get((g, "B"), []) if "error" not in r}
            okC = {r["code"]: r for r in all_res.get((g, "C"), []) if "error" not in r}
            codes = sorted(set(okA) & set(okB) & set(okC))
            ba = [okA[c][metric] - okB[c][metric] for c in codes]
            cb = [okB[c][metric] - okC[c][metric] for c in codes]
            pos_base = gi * 3
            bp = ax.boxplot([ba, cb], positions=[pos_base, pos_base + 1], widths=0.6,
                            patch_artist=True)
            for p, col in zip(bp["boxes"], ["#ff7f0e", "#2ca02c"]):
                p.set_facecolor(col)
                p.set_alpha(0.7)
        lab = metric.upper()
        ax.set_title("%s Improvement (positive = better)" % lab, fontweight="bold")
        ax.set_xticks([0.5, 3.5, 6.5])
        ax.set_xticklabels([g.upper() for g in GROUP_NAMES])
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_ylabel("%s(prev) - %s(next)" % (lab, lab))
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Information Criterion Improvement", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aic_bic_improvement.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- 9. GOF QQ Plot (sample stock, Model C) ----------
    print("  [VIZ] GOF QQ plot (sample) ...")
    _plot_gof_qq_sample(all_res, out_dir)

    print("  [VIZ] All %d figures saved to %s" % (9, out_dir))


def _plot_gof_qq_sample(all_res, out_dir):
    """为每个价格组选一只代表股票，画 Model C 的 QQ 图"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for gi, g in enumerate(GROUP_NAMES):
        ok = [r for r in all_res.get((g, "C"), []) if "error" not in r]
        if not ok:
            continue
        sample = sorted(ok, key=lambda r: r.get("gof_summary", {}).get("mean_gof_score", 0))[-1]
        gof_det = sample.get("gof_details", {})
        for d in range(4):
            ax = axes[gi, d]
            info = gof_det.get(str(d), {})
            n_res = info.get("n", 0)
            if n_res < 20:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            res_mean = info.get("mean", 1.0)
            ks_p = info.get("ks_pval", 0)
            gof_s = info.get("gof_score", 0)
            theo = np.sort(expon.ppf(np.linspace(0.001, 0.999, 200)))
            emp = np.sort(expon.ppf(np.linspace(0.001, 0.999, 200)) * res_mean)
            ax.plot(theo, emp, "o", ms=2, alpha=0.6, color=MODEL_COLORS["C"])
            mx = max(theo.max(), emp.max())
            ax.plot([0, mx], [0, mx], "k--", lw=1)
            ax.set_title("%s-%s (KS p=%.3f)" % (g[0].upper(), DIM_NAMES[d], ks_p), fontsize=9)
            ax.set_xlabel("Exp(1) theoretical")
            ax.set_ylabel("Empirical")
            ax.text(0.05, 0.9, "GOF=%.3f" % gof_s, transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.suptitle("GOF QQ-Plot (Best Stock per Group, Model C)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gof_qq_plot_sample.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ===================== 主流程 =====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Cython acceleration: %s" % ("ENABLED" if _USE_CYTHON else "DISABLED (pure Python)"))
    all_res = {}
    all_raw = []
    t_start = time.time()

    for g in GROUP_NAMES:
        data_dir = os.path.join(DATA_DIR, "%s_price_events" % g)
        if not os.path.isdir(data_dir):
            print("[WARN] missing %s" % data_dir)
            continue

        stock_files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith("events_") and f.endswith(".json") and not f.startswith("all_events")])
        print("\n" + "=" * 70)
        print("  GROUP: %s  (%d stocks)" % (g.upper(), len(stock_files)))
        print("=" * 70)

        for sf in stock_files:
            code = sf.replace("events_", "").replace("_201912.json", "").replace(".json", "")
            built = load_and_build(os.path.join(data_dir, sf))
            if built is None:
                print("  [%s] SKIP empty" % code)
                continue
            counts = [len(e) for e in built["events"]]
            total = sum(counts)
            print("  [%s] events=%s total=%d days=%d" % (code, counts, total, built["n_days"]))

            prev_res = None
            for m in ["A", "B", "C"]:
                t0 = time.time()
                r = fit_stock(code, built, m, init_from=prev_res)
                elapsed = time.time() - t0
                r["group"] = g
                r["elapsed_s"] = round(elapsed, 1)
                all_res.setdefault((g, m), []).append(r)
                all_raw.append(r)

                if "error" not in r:
                    prev_res = r
                    print("    Model %s: LL=%.1f  AIC=%.1f  BR=%.4f  GOF=%.3f  %.1fs" % (
                        m, r["loglik"], r["aic"], r["branching_ratio"],
                        r["gof_summary"]["mean_gof_score"], elapsed))
                else:
                    print("    Model %s: ERROR %s" % (m, r["error"]))

            # 增量保存
            _save_json(all_raw, os.path.join(OUT_DIR, "experiment_results.json"))

    elapsed_total = time.time() - t_start
    print("\nTotal elapsed: %.1fs (%.1fmin)" % (elapsed_total, elapsed_total / 60))

    _save_json(all_raw, os.path.join(OUT_DIR, "experiment_results.json"))

    print_summary_table(all_res)

    summary = build_summary(all_res)
    _save_json(summary, os.path.join(OUT_DIR, "experiment_summary.json"))

    plot_paper_figures(all_res, OUT_DIR)

    print("\nAll results saved to %s" % OUT_DIR)


def _save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def build_summary(all_res):
    summary = {}
    for g in GROUP_NAMES:
        for m in ["A", "B", "C"]:
            ok = [r for r in all_res.get((g, m), []) if "error" not in r]
            if not ok:
                continue
            lls = [r["loglik"] for r in ok]
            aics = [r["aic"] for r in ok]
            bics = [r["bic"] for r in ok]
            brs = [r["branching_ratio"] for r in ok]
            gofs = [r["gof_summary"]["mean_gof_score"] for r in ok]
            summary["%s_%s" % (g, m)] = {
                "n": len(ok),
                "ll_mean": float(np.mean(lls)), "ll_std": float(np.std(lls)),
                "aic_mean": float(np.mean(aics)), "aic_std": float(np.std(aics)),
                "bic_mean": float(np.mean(bics)), "bic_std": float(np.std(bics)),
                "br_mean": float(np.mean(brs)), "br_std": float(np.std(brs)),
                "gof_mean": float(np.mean(gofs)), "gof_std": float(np.std(gofs)),
                "gof_pass_4_4": sum(1 for r in ok if r["gof_summary"]["all_pass"]),
                "stable": sum(1 for br in brs if br < 1),
            }
    return summary


def print_summary_table(all_res):
    print("\n" + "=" * 110)
    print("%-6s %-5s %3s %12s %12s %12s %8s %6s %6s %6s" % (
        "Group", "Model", "N", "LL mean", "AIC mean", "BIC mean", "BR", "GOF", "Pass", "Time"))
    print("-" * 110)
    for g in GROUP_NAMES:
        for m in ["A", "B", "C"]:
            ok = [r for r in all_res.get((g, m), []) if "error" not in r]
            if not ok:
                continue
            ll_m = np.mean([r["loglik"] for r in ok])
            aic_m = np.mean([r["aic"] for r in ok])
            bic_m = np.mean([r["bic"] for r in ok])
            br_m = np.mean([r["branching_ratio"] for r in ok])
            gof_m = np.mean([r["gof_summary"]["mean_gof_score"] for r in ok])
            gof_p = sum(1 for r in ok if r["gof_summary"]["all_pass"])
            t_m = np.mean([r.get("elapsed_s", 0) for r in ok])
            n_tot = len(ok)
            print("%-6s %-5s %3d %12.1f %12.1f %12.1f %8.4f %6.3f %2d/%2d %5.1fs" % (
                g.upper(), m, n_tot, ll_m, aic_m, bic_m, br_m, gof_m, gof_p, n_tot, t_m))
        print("-" * 110)

    print("\nLL Monotonicity Check:")
    for g in GROUP_NAMES:
        okA = {r["code"]: r for r in all_res.get((g, "A"), []) if "error" not in r}
        okB = {r["code"]: r for r in all_res.get((g, "B"), []) if "error" not in r}
        okC = {r["code"]: r for r in all_res.get((g, "C"), []) if "error" not in r}
        codes = sorted(set(okA) & set(okB) & set(okC))
        mono = sum(1 for c in codes if okC[c]["loglik"] >= okB[c]["loglik"] >= okA[c]["loglik"])
        ba = np.mean([okB[c]["loglik"] - okA[c]["loglik"] for c in codes]) if codes else 0
        cb = np.mean([okC[c]["loglik"] - okB[c]["loglik"] for c in codes]) if codes else 0
        print("  %s: %d/%d monotonic,  LL(B)-LL(A)=%+.1f,  LL(C)-LL(B)=%+.1f" % (
            g.upper(), mono, len(codes), ba, cb))

    print("\nAIC/BIC Best Model Count:")
    for g in GROUP_NAMES:
        okA = {r["code"]: r for r in all_res.get((g, "A"), []) if "error" not in r}
        okB = {r["code"]: r for r in all_res.get((g, "B"), []) if "error" not in r}
        okC = {r["code"]: r for r in all_res.get((g, "C"), []) if "error" not in r}
        codes = sorted(set(okA) & set(okB) & set(okC))
        aic_best = {"A": 0, "B": 0, "C": 0}
        bic_best = {"A": 0, "B": 0, "C": 0}
        for c in codes:
            dd = {"A": okA[c], "B": okB[c], "C": okC[c]}
            best_a = min("ABC", key=lambda x: dd[x]["aic"])
            best_b = min("ABC", key=lambda x: dd[x]["bic"])
            aic_best[best_a] += 1
            bic_best[best_b] += 1
        print("  %s: AIC best -> A:%d B:%d C:%d  |  BIC best -> A:%d B:%d C:%d" % (
            g.upper(), aic_best["A"], aic_best["B"], aic_best["C"],
            bic_best["A"], bic_best["B"], bic_best["C"]))
    print("=" * 110)


if __name__ == "__main__":
    main()
