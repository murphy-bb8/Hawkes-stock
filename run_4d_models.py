"""
run_4d_models.py  —  基于 EM 算法的 4D Hawkes 三种模型批量拟合
================================================================
Model A: 常数 μ
Model B: 时变 μ (哑变量 OPEN30/MID30/CLOSE30)
Model C: 时变 μ + re_spread 外生项

不依赖 tick 包，使用 hawkes_em.py 的 EM 算法。
β 通过网格搜索 + 对数似然选优。
拟合与 GOF 使用同一套强度函数口径。

运行：conda activate py385 && python run_4d_models.py
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawkes_em import (
    em_estimate, loglikelihood, grid_search_beta, fit_4d,
    compute_gof_residuals, estimate_gamma_from_events,
    correct_mu_for_gamma,
    intraday_to_trading_time, get_intraday_time,
    TRADING_SECONDS_PER_DAY,
    SpreadProcess, loglikelihood_loglink,
)


# ===================== 数据加载 =====================

def load_stock_data(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_4d_events(stock_data: Dict) -> Dict:
    """
    从股票数据构建 4D 事件。

    Returns
    -------
    dict:
      events       : List[np.ndarray]  归一化连续时间 (用于拟合)
      events_orig  : List[np.ndarray]  日内时间 (用于时段判断)
      T            : float             总时长
      counts       : List[int]
      spread_times : np.ndarray        re_spread 的时间点 (归一化)
      spread_vals  : np.ndarray        re_spread 值
    """
    event_types = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]

    events_offset = []   # 带日偏移的连续时间
    events_intraday = [] # 日内时间
    spread_t_list = []
    spread_v_list = []

    for et in event_types:
        evts = stock_data.get("events", {}).get(et, {})
        t_off = []
        t_intra = []

        if isinstance(evts, dict) and "days" in evts:
            days_list = evts["days"]
            dates = sorted(set(d.get("date", "") for d in days_list if isinstance(d, dict)))
            date_idx = {d: i for i, d in enumerate(dates)}

            for day_data in days_list:
                if not isinstance(day_data, dict) or "t" not in day_data:
                    continue
                d_idx = date_idx.get(day_data.get("date", ""), 0)
                offset = d_idx * TRADING_SECONDS_PER_DAY

                t_data = day_data["t"]
                if isinstance(t_data, list):
                    for item in t_data:
                        if isinstance(item, (int, float)):
                            t_id = float(item)
                            t_off.append(offset + intraday_to_trading_time(t_id))
                            t_intra.append(t_id)

                # re_spread: 从所有事件类型收集 (P0-1 修复内生采样)
                if "re_spread" in day_data:
                    rs = day_data["re_spread"]
                    ts = day_data["t"]
                    if isinstance(rs, list) and isinstance(ts, list) and len(rs) == len(ts):
                        for ti, ri in zip(ts, rs):
                            if isinstance(ti, (int, float)) and isinstance(ri, (int, float)):
                                spread_t_list.append(offset + intraday_to_trading_time(float(ti)))
                                spread_v_list.append(float(ri))

        events_offset.append(np.asarray(t_off, dtype=float))
        events_intraday.append(np.asarray(t_intra, dtype=float))

    # 计算交易日数
    n_days = 1
    first_evts = stock_data.get("events", {}).get("buy_toxic", {})
    if isinstance(first_evts, dict) and "days" in first_evts:
        dates_set = set(d.get("date", "") for d in first_evts["days"] if isinstance(d, dict))
        n_days = max(len(dates_set), 1)

    all_t = np.concatenate([e for e in events_offset if len(e) > 0]) if any(len(e) > 0 for e in events_offset) else np.array([])
    if len(all_t) == 0:
        return {"events": [], "events_orig": [], "T": 0.0, "counts": [0]*4,
                "spread_times": np.array([]), "spread_vals": np.array([]), "n_days": 1}

    t0 = float(np.min(all_t))
    events_norm = [e - t0 if len(e) > 0 else e for e in events_offset]
    T = float(np.max(all_t) - t0)
    counts = [len(e) for e in events_offset]

    # 排序
    events_sorted = []
    events_intra_sorted = []
    for i in range(4):
        if len(events_norm[i]) > 0:
            idx = np.argsort(events_norm[i])
            events_sorted.append(events_norm[i][idx])
            events_intra_sorted.append(events_intraday[i][idx])
        else:
            events_sorted.append(events_norm[i])
            events_intra_sorted.append(events_intraday[i])

    # spread: 去重（同一时刻多来源取均值）+ 构建 SpreadProcess
    sp_t = np.asarray(spread_t_list, dtype=float) - t0 if spread_t_list else np.array([])
    sp_v = np.asarray(spread_v_list, dtype=float) if spread_v_list else np.array([])
    spread_proc = None
    if len(sp_t) > 0:
        sp_idx = np.argsort(sp_t)
        sp_t = sp_t[sp_idx]
        sp_v = sp_v[sp_idx]
        # 去重：同一时刻取均值
        uniq_t, inv = np.unique(sp_t, return_inverse=True)
        uniq_v = np.zeros_like(uniq_t)
        cnt = np.zeros_like(uniq_t)
        for k in range(len(sp_t)):
            uniq_v[inv[k]] += sp_v[k]
            cnt[inv[k]] += 1.0
        uniq_v /= np.maximum(cnt, 1.0)
        sp_t = uniq_t
        sp_v = uniq_v
        spread_proc = SpreadProcess(sp_t, sp_v, method='previous',
                                    lag=0.0, standardize=True)

    return {
        "events": events_sorted,
        "events_orig": events_intra_sorted,
        "T": T,
        "counts": counts,
        "spread_times": sp_t,
        "spread_vals": sp_v,
        "spread_proc": spread_proc,
        "n_days": n_days,
    }


# ===================== 单只股票拟合 =====================

def fit_single_stock(stock_code: str, stock_data: Dict,
                     model: str, beta_grid: np.ndarray,
                     output_dir: str, maxiter: int = 80) -> Dict:
    """对单只股票拟合指定模型"""
    built = build_4d_events(stock_data)
    events_4d = built["events"]
    events_orig = built["events_orig"]
    T = built["T"]
    counts = built["counts"]
    total = sum(counts)

    print(f"  [{stock_code}] events={counts}, total={total}")

    if total < 20:
        print(f"    -> 跳过: 事件不足")
        return {"stock_code": stock_code, "model": model, "error": "insufficient_events", "n_events": counts}

    try:
        spread_t = built["spread_times"] if model == "C" else None
        spread_v = built["spread_vals"] if model == "C" else None
        sp_proc = built.get("spread_proc") if model == "C" else None
        n_days = built.get("n_days", 22)

        result = fit_4d(
            events_4d, T, beta_grid,
            model=model,
            events_4d_original=events_orig if model in ("B", "C") else None,
            spread_times=spread_t,
            spread_values=spread_v,
            spread_proc=sp_proc,
            n_days=n_days,
            maxiter=maxiter,
            verbose=False,
        )

        if "error" in result:
            print(f"    -> 错误: {result['error']}")
            result["stock_code"] = stock_code
            result["n_events"] = counts
            return result

        result["stock_code"] = stock_code
        result["n_events"] = counts
        result["T"] = T

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{stock_code}_4d.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        gof_score = result.get("gof", {}).get("summary", {}).get("gof_score_mean", 0)
        gof_pass = result.get("gof", {}).get("summary", {}).get("gof_pass_count", 0)
        br = result["full"]["branching_ratio"]
        beta = result["full"]["decay"]
        print(f"    -> β={beta:.2f}, BR={br:.4f}, GOF={gof_score:.3f}, pass={gof_pass}/4")
        return result

    except Exception as e:
        import traceback
        print(f"    -> 异常: {e}")
        traceback.print_exc()
        return {"stock_code": stock_code, "model": model, "error": str(e), "n_events": counts}


# ===================== 分组处理 =====================

def process_group(group_name: str, data_dir: str, output_dir: str,
                  model: str, beta_grid: np.ndarray) -> List[Dict]:
    """处理一个价格分组"""
    print(f"\n{'='*70}")
    print(f"  {group_name.upper()} 组 — Model {model}")
    print(f"  数据: {data_dir}")
    print(f"  输出: {output_dir}")
    print(f"{'='*70}")

    stock_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith("events_") and f.endswith(".json") and not f.startswith("all_events")
    ])
    print(f"  找到 {len(stock_files)} 只股票")

    results = []
    for sf in stock_files:
        code = sf.replace("events_", "").replace("_201912.json", "").replace(".json", "")
        data = load_stock_data(os.path.join(data_dir, sf))
        r = fit_single_stock(code, data, model, beta_grid, output_dir)
        results.append(r)

    # 汇总
    os.makedirs(output_dir, exist_ok=True)
    generate_summary(results, output_dir, group_name, model)
    return results


def generate_summary(results: List[Dict], output_dir: str,
                     group_name: str, model: str):
    """生成分组汇总"""
    ok = [r for r in results if "error" not in r]
    fail = [r for r in results if "error" in r]

    print(f"\n  --- {group_name} Model {model} 汇总 ---")
    print(f"  成功: {len(ok)}, 失败: {len(fail)}")

    if not ok:
        return

    brs = [r["full"]["branching_ratio"] for r in ok]
    betas = [r["full"]["decay"] for r in ok]
    gof_scores = [r["gof"]["summary"].get("gof_score_mean", 0) for r in ok]
    all_pass = sum(1 for r in ok if r["gof"]["summary"]["all_pass"])

    summary = {
        "group": group_name, "model": model,
        "n_stocks": len(results), "n_success": len(ok), "n_fail": len(fail),
        "branching_ratio": {
            "mean": float(np.mean(brs)), "std": float(np.std(brs)),
            "min": float(np.min(brs)), "max": float(np.max(brs)),
            "all_stable": all(br < 1.0 for br in brs),
        },
        "beta": {"mean": float(np.mean(betas)), "std": float(np.std(betas))},
        "gof": {
            "score_mean": float(np.mean(gof_scores)),
            "all_pass_count": all_pass,
        },
        "mu_mean": np.mean([r["full"]["mu"] for r in ok], axis=0).tolist(),
        "A_mean": np.mean([r["full"]["A"] for r in ok], axis=0).tolist(),
    }

    # gamma 统计 (Model B/C)
    if model in ("B", "C"):
        gammas_o = [r["gamma"]["gamma_open"] for r in ok if "gamma" in r]
        gammas_m = [r["gamma"]["gamma_mid"] for r in ok if "gamma" in r]
        gammas_c = [r["gamma"]["gamma_close"] for r in ok if "gamma" in r]
        if gammas_o:
            summary["gamma_open_mean"] = np.mean(gammas_o, axis=0).tolist()
            summary["gamma_mid_mean"] = np.mean(gammas_m, axis=0).tolist()
            summary["gamma_close_mean"] = np.mean(gammas_c, axis=0).tolist()

    # 分维度 GOF
    for d in range(4):
        dk = f"dim_{d}"
        scores = [r["gof"][dk].get("gof_score", 0) for r in ok if dk in r.get("gof", {})]
        w1s = [r["gof"][dk].get("wasserstein_1", 0) for r in ok if dk in r.get("gof", {})]
        means = [r["gof"][dk].get("mae_mean", 0) for r in ok if dk in r.get("gof", {})]
        if scores:
            summary[f"gof_{dk}"] = {
                "score_mean": float(np.mean(scores)),
                "w1_mean": float(np.mean(w1s)),
                "mae_mean_mean": float(np.mean(means)),
            }

    with open(os.path.join(output_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  BR: {summary['branching_ratio']['mean']:.4f} ± {summary['branching_ratio']['std']:.4f}")
    print(f"  GOF score: {summary['gof']['score_mean']:.3f}, all_pass: {all_pass}/{len(ok)}")


# ===================== 可视化 =====================

def plot_all_visualizations(all_results: Dict[str, List[Dict]], output_dir: str, model: str):
    """生成所有可视化"""
    from scipy.stats import expon

    group_names = ["high", "mid", "low"]
    dim_names = ["BT", "BN", "ST", "SN"]
    dim_full = ["BuyToxic", "BuyNotToxic", "SellToxic", "SellNotToxic"]
    colors = {"high": "#e74c3c", "mid": "#f39c12", "low": "#27ae60"}

    # --- 1. QQ 图 ---
    print("  生成 QQ 图...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    for row, gn in enumerate(group_names):
        ok = [r for r in all_results.get(gn, []) if "error" not in r and "gof" in r]
        for col in range(4):
            ax = axes[row, col]
            dk = f"dim_{col}"
            qq_emps = [r["gof"][dk]["qq_empirical"] for r in ok if dk in r["gof"] and "qq_empirical" in r["gof"][dk]]
            qq_theos = [r["gof"][dk]["qq_theoretical"] for r in ok if dk in r["gof"] and "qq_theoretical" in r["gof"][dk]]
            if qq_emps:
                emp = np.array(qq_emps)
                theo = np.array(qq_theos[0])
                med = np.median(emp, axis=0)
                q25 = np.percentile(emp, 25, axis=0)
                q75 = np.percentile(emp, 75, axis=0)
                ax.fill_between(theo, q25, q75, alpha=0.25, color=colors[gn])
                ax.plot(theo, med, 'o', color=colors[gn], markersize=2, alpha=0.8)
                mx = max(theo.max(), q75.max()) * 1.05
                ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.6)
                w1s = [r["gof"][dk].get("wasserstein_1", 0) for r in ok if dk in r["gof"]]
                if w1s:
                    ax.text(0.05, 0.92, f'W1={np.mean(w1s):.2f}', transform=ax.transAxes, fontsize=8, va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            if row == 0: ax.set_title(dim_full[col], fontsize=11, fontweight='bold')
            if row == 2: ax.set_xlabel("Exp(1) Theoretical", fontsize=9)
            if col == 0: ax.set_ylabel(f"{gn.upper()}\nEmpirical", fontsize=10)
            ax.grid(True, alpha=0.2)
    plt.suptitle(f"QQ-Exp(1) Diagnostic (Model {model}, EM)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gof_qq_plots.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 2. 距离热力图 ---
    print("  生成距离热力图...")
    metrics = ["wasserstein_1", "qq_mae", "mae_mean"]
    mlabels = ["Wasserstein-1", "QQ-MAE", "|Mean−1|"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for mi, (met, ml) in enumerate(zip(metrics, mlabels)):
        ax = axes[mi]
        data = np.zeros((3, 4))
        for gi, gn in enumerate(group_names):
            ok = [r for r in all_results.get(gn, []) if "error" not in r and "gof" in r]
            for d in range(4):
                dk = f"dim_{d}"
                vals = [r["gof"][dk].get(met, 0) for r in ok if dk in r["gof"] and met in r["gof"].get(dk, {})]
                data[gi, d] = np.mean(vals) if vals else 0
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center', fontsize=10)
        ax.set_xticks(range(4)); ax.set_xticklabels(dim_names)
        ax.set_yticks(range(3)); ax.set_yticklabels([g.upper() for g in group_names])
        ax.set_title(ml, fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle(f"Distance Metrics (Model {model}, EM)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_metrics_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 3. ACF 面板 ---
    print("  生成 ACF 面板...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for row, gn in enumerate(group_names):
        ok = [r for r in all_results.get(gn, []) if "error" not in r and "gof" in r]
        for col in range(4):
            ax = axes[row, col]
            dk = f"dim_{col}"
            all_acf = [r["gof"][dk]["acf_values"] for r in ok if dk in r["gof"] and "acf_values" in r["gof"][dk] and r["gof"][dk]["acf_values"]]
            if all_acf:
                acf_mean = np.mean(all_acf, axis=0)
                lags = np.arange(1, len(acf_mean) + 1)
                ax.bar(lags, acf_mean, color=colors[gn], alpha=0.7, width=0.8)
                ci = 1.96 / np.sqrt(500)
                ax.axhline(y=ci, color='blue', ls='--', alpha=0.5)
                ax.axhline(y=-ci, color='blue', ls='--', alpha=0.5)
                ax.axhline(y=0, color='black', lw=0.5)
            if row == 0: ax.set_title(dim_full[col], fontsize=11, fontweight='bold')
            if col == 0: ax.set_ylabel(f"{gn.upper()}\nACF", fontsize=10)
            ax.set_ylim(-0.15, 0.15)
            ax.grid(True, alpha=0.2, axis='y')
    plt.suptitle(f"ACF Independence (Model {model}, EM)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acf_independence_panel.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 4. 分枝比箱线图 ---
    print("  生成分枝比箱线图...")
    fig, ax = plt.subplots(figsize=(8, 6))
    bp_data = []
    for gn in group_names:
        ok = [r for r in all_results.get(gn, []) if "error" not in r]
        bp_data.append([r["full"]["branching_ratio"] for r in ok] if ok else [0])
    bp = ax.boxplot(bp_data, labels=[g.upper() for g in group_names], patch_artist=True)
    for patch, c in zip(bp['boxes'], [colors[g] for g in group_names]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(y=1.0, color='red', ls='--', lw=2, label='BR=1')
    ax.set_ylabel("Branching Ratio", fontsize=12)
    ax.set_title(f"Branching Ratio (Model {model}, EM)", fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "branching_ratio_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 5. 激励矩阵热力图 ---
    print("  生成激励矩阵热力图...")
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    dim_labels = ["Buy\nToxic", "Buy\nNot Toxic", "Sell\nToxic", "Sell\nNot Toxic"]
    im_last = None
    for idx, gn in enumerate(group_names):
        ax = fig.add_subplot(gs[0, idx])
        ok = [r for r in all_results.get(gn, []) if "error" not in r]
        if ok:
            As = np.array([r["full"]["A"] for r in ok])
            A_mean = np.mean(As, axis=0)
            im_last = ax.imshow(A_mean, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
            for i in range(4):
                for j in range(4):
                    c = 'white' if A_mean[i,j] > 0.5 else 'black'
                    ax.text(j, i, f'{A_mean[i,j]:.3f}', ha='center', va='center', color=c, fontsize=9)
            ax.set_xticks(range(4)); ax.set_xticklabels(dim_labels, fontsize=9)
            ax.set_yticks(range(4)); ax.set_yticklabels(dim_labels, fontsize=9)
            ax.set_title(f"{gn.upper()} (n={len(ok)})", fontsize=12, fontweight='bold')
    if im_last:
        cbar_ax = fig.add_subplot(gs[0, 3])
        fig.colorbar(im_last, cax=cbar_ax).set_label('α[i,j]')
    plt.suptitle(f"Excitation Matrix (Model {model}, EM)", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, "excitation_matrix_heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  可视化完成 → {output_dir}")


# ===================== 主流程 =====================

def run_model(model: str):
    """运行指定模型"""
    base_data = os.path.join(os.path.dirname(__file__), "data")

    groups = {
        "high": os.path.join(base_data, "high_price_events"),
        "mid":  os.path.join(base_data, "mid_price_events"),
        "low":  os.path.join(base_data, "low_price_events"),
    }

    # β 网格（三个模型统一）
    beta_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    # 输出目录
    suffix_map = {"A": "noexog_em", "B": "em", "C": "exog_em"}
    out_base = os.path.join(os.path.dirname(__file__), f"results_{suffix_map[model]}")

    all_results = {}
    t0 = time.time()

    for gn, data_dir in groups.items():
        if not os.path.isdir(data_dir):
            print(f"  ⚠ 数据目录不存在: {data_dir}")
            continue
        out_dir = os.path.join(out_base, f"{gn}_price_4d_{suffix_map[model]}")
        results = process_group(gn, data_dir, out_dir, model, beta_grid)
        all_results[gn] = results

    elapsed = time.time() - t0
    print(f"\n  Model {model} 总耗时: {elapsed:.1f}s")

    # 可视化
    if any(all_results.values()):
        plot_all_visualizations(all_results, out_base, model)

    # 跨组对比
    comparison = {}
    for gn, results in all_results.items():
        ok = [r for r in results if "error" not in r]
        if ok:
            comparison[gn] = {
                "n_stocks": len(ok),
                "br_mean": float(np.mean([r["full"]["branching_ratio"] for r in ok])),
                "br_std": float(np.std([r["full"]["branching_ratio"] for r in ok])),
                "gof_mean": float(np.mean([r["gof"]["summary"].get("gof_score_mean", 0) for r in ok])),
                "all_pass": sum(1 for r in ok if r["gof"]["summary"]["all_pass"]),
                "beta_mean": float(np.mean([r["full"]["decay"] for r in ok])),
            }

    comp_file = os.path.join(out_base, f"comparison_model_{model}.json")
    os.makedirs(out_base, exist_ok=True)
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"  对比结果: {comp_file}")

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="4D Hawkes EM 拟合")
    parser.add_argument("--model", type=str, default="A",
                        choices=["A", "B", "C", "all"],
                        help="模型选择: A/B/C/all")
    parser.add_argument("--stock", type=str, default=None,
                        help="单只股票代码 (调试用)")
    args = parser.parse_args()

    if args.stock:
        # 单只股票调试模式
        base_data = os.path.join(os.path.dirname(__file__), "data")
        # 搜索股票文件
        for sub in ["high_price_events", "mid_price_events", "low_price_events"]:
            d = os.path.join(base_data, sub)
            f = os.path.join(d, f"events_{args.stock}_201912.json")
            if os.path.exists(f):
                print(f"找到: {f}")
                data = load_stock_data(f)
                model = args.model if args.model != "all" else "B"
                beta_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
                r = fit_single_stock(args.stock, data, model, beta_grid, "results_debug")
                print(json.dumps({k: v for k, v in r.items() if k != "gof"}, indent=2, ensure_ascii=False))
                return
        print(f"未找到股票 {args.stock}")
        return

    if args.model == "all":
        for m in ["A", "B", "C"]:
            run_model(m)
    else:
        run_model(args.model)


if __name__ == "__main__":
    main()
