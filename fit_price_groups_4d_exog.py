"""
按价格分组的4D Hawkes模型拟合实验 - 带re_spread外生项版本
处理 high_price_events, mid_price_events, low_price_events 三组数据
每组15只股票，共45只股票

模型：λ_i(t) = μ_i(t) + Σ_j Σ_{t_k^j < t} A_{ij} e^{-β(t-t_k^j)} + γ_spread_i * re_spread(t)
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats

from fit_toxic_events import intraday_to_trading_time
from hawkes_4d_tick_exog import run_comparison_4d_tick_exog, load_events_4d


def load_single_stock_data(data_path: str) -> Dict:
    """加载单只股票的事件数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_event_times_and_spread_with_day_offset(
    stock_data: Dict, event_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从股票数据中提取事件时间和re_spread，包含日期偏移
    
    Returns:
    --------
    times_with_offset : np.ndarray
        带日期偏移的连续时间（用于tick拟合）
    times_intraday : np.ndarray
        日内时间（用于时段判断）
    re_spreads : np.ndarray
        每个事件的re_spread值
    """
    events = stock_data.get("events", {}).get(event_type, {})
    
    times_with_offset = []
    times_intraday = []
    re_spreads = []
    
    TRADING_SECONDS_PER_DAY = 14400
    
    if isinstance(events, dict) and "days" in events and isinstance(events["days"], list):
        days_list = events["days"]
        day_dates = []
        for day_data in days_list:
            if isinstance(day_data, dict) and "date" in day_data:
                day_dates.append(day_data["date"])
        day_dates_sorted = sorted(set(day_dates))
        date_to_index = {d: i for i, d in enumerate(day_dates_sorted)}
        
        for day_data in days_list:
            if not isinstance(day_data, dict) or "t" not in day_data:
                continue
            
            date_str = day_data.get("date", "")
            day_idx = date_to_index.get(date_str, 0)
            day_offset = day_idx * TRADING_SECONDS_PER_DAY
            
            t_data = day_data.get("t", [])
            spread_data = day_data.get("re_spread", [])
            
            if not isinstance(t_data, list):
                t_data = [t_data] if isinstance(t_data, (int, float)) else []
            if not isinstance(spread_data, list):
                spread_data = [spread_data] if isinstance(spread_data, (int, float)) else []
            
            # 确保spread数据与时间数据对齐
            if len(spread_data) < len(t_data):
                # 如果spread数据不足，用0填充
                spread_data = list(spread_data) + [0.0] * (len(t_data) - len(spread_data))
            
            for i, t_val in enumerate(t_data):
                if isinstance(t_val, (int, float)):
                    t_intraday = float(t_val)
                    t_trading = intraday_to_trading_time(t_intraday)
                    times_with_offset.append(day_offset + t_trading)
                    times_intraday.append(t_intraday)
                    
                    # 获取对应的spread值
                    if i < len(spread_data) and isinstance(spread_data[i], (int, float)):
                        re_spreads.append(float(spread_data[i]))
                    else:
                        re_spreads.append(0.0)
    
    return (np.asarray(times_with_offset, dtype=float), 
            np.asarray(times_intraday, dtype=float),
            np.asarray(re_spreads, dtype=float))


def build_4d_events_with_spread(stock_data: Dict) -> Dict:
    """
    构建4D事件数据，包含re_spread外生变量
    
    维度0: buy_toxic
    维度1: buy_not_toxic
    维度2: sell_toxic
    维度3: sell_not_toxic
    
    Returns:
    --------
    dict包含：
        events: 归一化后的事件时间
        events_original: 日内时间
        re_spread: 每个事件的re_spread值
        T: 总时间长度
        counts: 各维度事件数
    """
    event_types = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    events_with_offset = []
    events_intraday = []
    re_spreads = []
    
    for et in event_types:
        times_offset, times_intra, spreads = extract_event_times_and_spread_with_day_offset(stock_data, et)
        if len(times_offset) > 0:
            events_with_offset.append(times_offset.astype(float))
            events_intraday.append(times_intra.astype(float))
            re_spreads.append(spreads.astype(float))
        else:
            events_with_offset.append(np.asarray([], dtype=float))
            events_intraday.append(np.asarray([], dtype=float))
            re_spreads.append(np.asarray([], dtype=float))
    
    all_times = np.concatenate([ev for ev in events_with_offset if len(ev) > 0]) if any(len(ev) > 0 for ev in events_with_offset) else np.asarray([], dtype=float)
    
    if len(all_times) == 0:
        return {
            "events": [], 
            "events_original": [], 
            "re_spread": [],
            "T": 0.0, 
            "counts": [0, 0, 0, 0]
        }
    
    t0 = float(np.min(all_times))
    events_norm = [ev - t0 for ev in events_with_offset]
    T = float(np.max(all_times) - t0)
    counts = [int(len(ev)) for ev in events_with_offset]
    
    # 排序
    events_sorted = []
    events_intraday_sorted = []
    re_spreads_sorted = []
    for i in range(4):
        if len(events_norm[i]) > 0:
            sort_idx = np.argsort(events_norm[i])
            events_sorted.append(events_norm[i][sort_idx])
            events_intraday_sorted.append(events_intraday[i][sort_idx])
            re_spreads_sorted.append(re_spreads[i][sort_idx])
        else:
            events_sorted.append(events_norm[i])
            events_intraday_sorted.append(events_intraday[i])
            re_spreads_sorted.append(re_spreads[i])
    
    return {
        "events": events_sorted, 
        "events_original": events_intraday_sorted,
        "re_spread": re_spreads_sorted,
        "T": T, 
        "counts": counts
    }


def fit_stock_4d_tick_exog(stock_code: str, stock_data: Dict, output_dir: str) -> Dict:
    """
    对单只股票进行4D Hawkes拟合（带re_spread外生项）
    """
    built = build_4d_events_with_spread(stock_data)
    events_4d = built["events"]
    events_4d_original = built["events_original"]
    re_spread_4d = built["re_spread"]
    counts = built["counts"]
    total_events = sum(counts)
    
    # 统计spread信息
    spread_stats = []
    for d in range(4):
        if len(re_spread_4d[d]) > 0:
            spread_stats.append(f"{np.mean(re_spread_4d[d]):.6f}")
        else:
            spread_stats.append("N/A")
    
    print(f"  Stock {stock_code}: events={counts}, total={total_events}")
    print(f"    re_spread means: {spread_stats}")
    
    if total_events < 20:
        print(f"    -> Skipped: insufficient events")
        return {
            "stock_code": stock_code,
            "event_type": "4d_exog",
            "error": "insufficient_events",
            "n_events": counts,
        }
    
    os.makedirs("temp", exist_ok=True)
    temp_file = f"temp/events_{stock_code}_4d_exog.json"
    
    payload = []
    for dim, (ev_norm, ev_orig, ev_spread) in enumerate(zip(events_4d, events_4d_original, re_spread_4d)):
        for t_norm, t_orig, spread in zip(ev_norm, ev_orig, ev_spread):
            payload.append({
                "t": float(t_norm), 
                "i": int(dim),
                "t_orig": float(t_orig),
                "re_spread": float(spread)
            })
    payload.sort(key=lambda x: x["t"])
    
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    
    os.environ["OUT_TAG"] = f"{stock_code}_4d_exog"
    
    try:
        result = run_comparison_4d_tick_exog(
            temp_file, 
            events_4d_original=events_4d_original,
            re_spread_4d=re_spread_4d
        )
        
        result["stock_code"] = stock_code
        result["event_type"] = "4d_exog"
        result["n_events"] = counts
        result["T"] = float(built["T"])
        
        # 添加spread统计
        result["spread_stats"] = {
            f"dim_{d}": {
                "mean": float(np.mean(re_spread_4d[d])) if len(re_spread_4d[d]) > 0 else None,
                "std": float(np.std(re_spread_4d[d])) if len(re_spread_4d[d]) > 0 else None,
                "min": float(np.min(re_spread_4d[d])) if len(re_spread_4d[d]) > 0 else None,
                "max": float(np.max(re_spread_4d[d])) if len(re_spread_4d[d]) > 0 else None,
            }
            for d in range(4)
        }
        
        # 保存
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{stock_code}_4d_exog.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"    -> Saved to {output_file}")
        return result
        
    except Exception as e:
        print(f"    -> Error: {e}")
        return {
            "stock_code": stock_code,
            "event_type": "4d_exog",
            "error": str(e),
            "n_events": counts,
        }


def process_price_group_exog(data_dir: str, output_dir: str, group_name: str) -> List[Dict]:
    """处理一个价格组的所有股票"""
    print(f"\n{'='*70}")
    print(f"Processing {group_name} price group (with re_spread exogenous)")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找数据文件
    data_files = [f for f in os.listdir(data_dir) 
                  if f.startswith("events_") and f.endswith(".json") 
                  and not f.startswith("all_events")]
    
    print(f"Found {len(data_files)} stock files")
    
    all_results = []
    for data_file in sorted(data_files):
        stock_code = data_file.replace("events_", "").replace("_201912.json", "")
        data_path = os.path.join(data_dir, data_file)
        
        print(f"\nProcessing {stock_code}...")
        stock_data = load_single_stock_data(data_path)
        result = fit_stock_4d_tick_exog(stock_code, stock_data, output_dir)
        all_results.append(result)
    
    generate_group_summary_exog(all_results, output_dir, group_name)
    
    return all_results


def generate_group_summary_exog(all_results: List[Dict], output_dir: str, group_name: str) -> None:
    """生成组汇总报告"""
    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    
    print(f"\n{'-'*50}")
    print(f"Group {group_name} Summary (with re_spread exogenous):")
    print(f"  Total: {len(all_results)}, Successful: {len(successful)}, Failed: {len(failed)}")
    
    if len(successful) == 0:
        print("  No successful fits to summarize")
        return
    
    # 收集统计
    branching_ratios = [r["full"]["branching_ratio"] for r in successful]
    decays = [r["full"]["decay"] for r in successful]
    
    gamma_spreads = []
    gamma_opens = []
    gamma_mids = []
    gamma_closes = []
    
    for r in successful:
        if "exog" in r and r["exog"].get("gamma_spread"):
            gamma_spreads.append(r["exog"]["gamma_spread"])
        if "gof" in r and "gamma" in r["gof"]:
            gamma_opens.append(r["gof"]["gamma"]["gamma_open"])
            gamma_mids.append(r["gof"]["gamma"]["gamma_mid"])
            gamma_closes.append(r["gof"]["gamma"]["gamma_close"])
    
    # 收集GOF v2统计
    gof_scores = {d: [] for d in range(4)}
    wasserstein_vals = {d: [] for d in range(4)}
    qq_mae_vals = {d: [] for d in range(4)}
    mae_mean_vals = {d: [] for d in range(4)}
    lb_pass_counts = {d: [] for d in range(4)}
    per_day_not_reject_ratios = {d: [] for d in range(4)}
    for r in successful:
        gof = r.get("gof", {})
        pds = gof.get("per_day_summary", {})
        for d in range(4):
            dk = f"dim_{d}"
            if dk in gof:
                if "gof_score" in gof[dk]:
                    gof_scores[d].append(gof[dk]["gof_score"])
                if "wasserstein_1" in gof[dk]:
                    wasserstein_vals[d].append(gof[dk]["wasserstein_1"])
                if "qq_mae" in gof[dk]:
                    qq_mae_vals[d].append(gof[dk]["qq_mae"])
                if "mae_mean" in gof[dk]:
                    mae_mean_vals[d].append(gof[dk]["mae_mean"])
                if "ljung_box_pass" in gof[dk] and gof[dk]["ljung_box_pass"] is not None:
                    lb_pass_counts[d].append(1 if gof[dk]["ljung_box_pass"] else 0)
            if dk in pds and "not_reject_ratio" in pds[dk]:
                per_day_not_reject_ratios[d].append(pds[dk]["not_reject_ratio"])
    
    # 汇总
    summary = {
        "group": group_name,
        "summary": {
            "total_stocks": len(all_results),
            "successful_fits": len(successful),
            "failed_fits": len(failed),
        },
        "branching_ratio": {
            "mean": float(np.mean(branching_ratios)),
            "std": float(np.std(branching_ratios)),
            "min": float(np.min(branching_ratios)),
            "max": float(np.max(branching_ratios)),
            "median": float(np.median(branching_ratios)),
            "stable_count": int(sum(1 for br in branching_ratios if br < 1.0)),
        },
        "decay": {
            "mean": float(np.mean(decays)),
            "std": float(np.std(decays)),
            "min": float(np.min(decays)),
            "max": float(np.max(decays)),
        },
        "gof": {
            "mean_pass_count": float(np.mean([r["gof"]["summary"]["gof_pass_count"] for r in successful])),
            "all_pass_count": int(sum(1 for r in successful if r["gof"]["summary"]["all_pass"])),
            "gof_score_mean": float(np.mean([r["gof"]["summary"].get("gof_score_mean", 0) for r in successful])),
            "model_type": "constant_mu_per_day",
        },
    }
    
    # 分维度GOF v2汇总
    for d in range(4):
        dk = f"dim_{d}"
        if len(gof_scores[d]) > 0:
            summary["gof"][f"{dk}_gof_score_mean"] = float(np.mean(gof_scores[d]))
        if len(wasserstein_vals[d]) > 0:
            summary["gof"][f"{dk}_wasserstein_mean"] = float(np.mean(wasserstein_vals[d]))
        if len(qq_mae_vals[d]) > 0:
            summary["gof"][f"{dk}_qq_mae_mean"] = float(np.mean(qq_mae_vals[d]))
        if len(mae_mean_vals[d]) > 0:
            summary["gof"][f"{dk}_mae_mean_mean"] = float(np.mean(mae_mean_vals[d]))
        if len(lb_pass_counts[d]) > 0:
            summary["gof"][f"{dk}_ljung_box_pass_ratio"] = float(np.mean(lb_pass_counts[d]))
        if len(per_day_not_reject_ratios[d]) > 0:
            summary["gof"][f"{dk}_per_day_not_reject_ratio_mean"] = float(np.mean(per_day_not_reject_ratios[d]))
    
    # gamma_spread 统计
    if len(gamma_spreads) > 0:
        gamma_spreads_arr = np.array(gamma_spreads)
        summary["gamma_spread"] = {
            "mean": gamma_spreads_arr.mean(axis=0).tolist(),
            "std": gamma_spreads_arr.std(axis=0).tolist(),
        }
    
    # gamma_open/mid/close 统计
    if len(gamma_opens) > 0:
        summary["gof"]["gamma_open_mean"] = np.mean(gamma_opens, axis=0).tolist()
        summary["gof"]["gamma_open_std"] = np.std(gamma_opens, axis=0).tolist()
    if len(gamma_mids) > 0:
        summary["gof"]["gamma_mid_mean"] = np.mean(gamma_mids, axis=0).tolist()
        summary["gof"]["gamma_mid_std"] = np.std(gamma_mids, axis=0).tolist()
    if len(gamma_closes) > 0:
        summary["gof"]["gamma_close_mean"] = np.mean(gamma_closes, axis=0).tolist()
        summary["gof"]["gamma_close_std"] = np.std(gamma_closes, axis=0).tolist()
    
    # 保存
    summary_file = os.path.join(output_dir, "summary_report_exog.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"  Branching ratio: {summary['branching_ratio']['mean']:.4f} ± {summary['branching_ratio']['std']:.4f}")
    print(f"  Stable count: {summary['branching_ratio']['stable_count']}/{len(successful)}")
    print(f"  GOF score: {summary['gof']['gof_score_mean']:.3f}, pass={summary['gof']['all_pass_count']}/{len(successful)}")
    if "gamma_spread" in summary:
        print(f"  gamma_spread mean: {summary['gamma_spread']['mean']}")
    print(f"  Summary saved to {summary_file}")


def plot_gof_qq_residuals(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    论文级 QQ-Exp(1) 图：残差分位数 vs Exp(1) 理论分位数
    3组×4维面板，带45°线和95%置信带
    使用每只股票GOF中预计算的qq_empirical/qq_theoretical
    """
    from scipy.stats import expon
    print("Generating QQ-Exp(1) plots...")
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    group_names = ["high", "mid", "low"]
    group_labels = ["High Price", "Mid Price", "Low Price"]
    dim_names = ["Buy Toxic", "Buy Not Toxic", "Sell Toxic", "Sell Not Toxic"]
    group_colors = ['#c0392b', '#e67e22', '#27ae60']
    
    for row, group_name in enumerate(group_names):
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        for col in range(4):
            ax = axes[row, col]
            dim_key = f"dim_{col}"
            
            # 收集所有股票的QQ分位数，取组均值
            all_qq_emp = []
            all_qq_theo = []
            for r in successful:
                gof_d = r["gof"].get(dim_key, {})
                if "qq_empirical" in gof_d and "qq_theoretical" in gof_d:
                    all_qq_emp.append(gof_d["qq_empirical"])
                    all_qq_theo.append(gof_d["qq_theoretical"])
            
            if len(all_qq_emp) >= 1:
                qq_emp_arr = np.array(all_qq_emp)
                qq_theo = np.array(all_qq_theo[0])  # 理论分位数相同
                qq_emp_mean = np.mean(qq_emp_arr, axis=0)
                qq_emp_q25 = np.percentile(qq_emp_arr, 25, axis=0)
                qq_emp_q75 = np.percentile(qq_emp_arr, 75, axis=0)
                
                # 45°参考线
                max_val = min(np.max(qq_theo), 8.0)
                ax.plot([0, max_val], [0, max_val], 'k-', linewidth=1.5, alpha=0.8, label='$y=x$ (Exp(1))')
                
                # IQR带（股票间变异）
                if len(all_qq_emp) > 2:
                    ax.fill_between(qq_theo, qq_emp_q25, qq_emp_q75,
                                    alpha=0.2, color=group_colors[row], label='IQR across stocks')
                
                # 组均值QQ线
                ax.plot(qq_theo, qq_emp_mean, 'o-', color=group_colors[row],
                        markersize=1.5, linewidth=1.2, alpha=0.9, label='Mean QQ')
                
                # 标注距离度量
                w1_vals = [r["gof"][dim_key].get("wasserstein_1", 0) for r in successful
                           if dim_key in r["gof"] and "wasserstein_1" in r["gof"][dim_key]]
                qq_mae_vals = [r["gof"][dim_key].get("qq_mae", 0) for r in successful
                               if dim_key in r["gof"] and "qq_mae" in r["gof"][dim_key]]
                mean_vals = [r["gof"][dim_key].get("mean", 0) for r in successful
                             if dim_key in r["gof"] and "mean" in r["gof"][dim_key]]
                
                info_text = (f"$\\bar{{\\mu}}_r$={np.mean(mean_vals):.3f}\n"
                             f"$W_1$={np.mean(w1_vals):.3f}\n"
                             f"QQ-MAE={np.mean(qq_mae_vals):.3f}")
                ax.text(0.03, 0.97, info_text, transform=ax.transAxes,
                        fontsize=7.5, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
                
                ax.set_xlim(0, max_val * 1.05)
                ax.set_ylim(0, min(np.max(qq_emp_q75) * 1.1 if len(all_qq_emp) > 2 else np.max(qq_emp_mean) * 1.1, max_val * 2))
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
            
            if row == 0:
                ax.set_title(dim_names[col], fontsize=12, fontweight='bold')
            if row == 2:
                ax.set_xlabel("Exp(1) Theoretical Quantiles", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{group_labels[row]}\nEmpirical Quantiles", fontsize=9)
            
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            if row == 0 and col == 3:
                ax.legend(fontsize=7, loc='lower right')
    
    plt.suptitle("QQ Diagnostic: Time-Rescaling Residuals vs Exp(1)",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gof_qq_plots.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved QQ plots to {output_dir}/gof_qq_plots.png")


def plot_distance_metrics_heatmap(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    论文级距离度量热力图：3组×4维的 Wasserstein-1 / QQ-MAE / |mean-1| 矩阵
    """
    print("Generating distance metrics heatmap...")
    
    group_names = ["high", "mid", "low"]
    group_labels = ["High", "Mid", "Low"]
    dim_names = ["Buy Toxic", "Buy Not\nToxic", "Sell Toxic", "Sell Not\nToxic"]
    metric_names = ["Wasserstein-1", "QQ-MAE", "|Mean − 1|"]
    metric_keys = ["wasserstein_1", "qq_mae", "mae_mean"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    
    for midx, (metric_key, metric_name) in enumerate(zip(metric_keys, metric_names)):
        ax = axes[midx]
        matrix = np.full((3, 4), np.nan)
        
        for gidx, group_name in enumerate(group_names):
            results = all_group_results.get(group_name, [])
            successful = [r for r in results if "error" not in r and "gof" in r]
            
            for d in range(4):
                dim_key = f"dim_{d}"
                vals = [r["gof"][dim_key].get(metric_key, np.nan) for r in successful
                        if dim_key in r["gof"] and metric_key in r["gof"][dim_key]]
                if len(vals) > 0:
                    matrix[gidx, d] = np.mean(vals)
        
        # 选择合适的colormap（距离越小越好 → 绿色好，红色差）
        vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=vmax)
        
        # 标注数值
        for i in range(3):
            for j in range(4):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            color=color, fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(4))
        ax.set_yticks(range(3))
        ax.set_xticklabels(dim_names, fontsize=9)
        ax.set_yticklabels(group_labels, fontsize=10)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle("GOF Distance Metrics (lower = better fit to Exp(1))",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_metrics_heatmap.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved distance metrics heatmap to {output_dir}/distance_metrics_heatmap.png")


def plot_independence_acf_panel(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    论文级独立性诊断面板：3组×4维的ACF图 + Ljung-Box p-value标注
    ACF值来自GOF结果中预计算的acf_values (lag 1~20)
    """
    print("Generating ACF independence panel...")
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    group_names = ["high", "mid", "low"]
    group_labels = ["High Price", "Mid Price", "Low Price"]
    dim_names = ["Buy Toxic", "Buy Not Toxic", "Sell Toxic", "Sell Not Toxic"]
    group_colors = ['#c0392b', '#e67e22', '#27ae60']
    
    for row, group_name in enumerate(group_names):
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        for col in range(4):
            ax = axes[row, col]
            dim_key = f"dim_{col}"
            
            # 收集所有股票的ACF值
            all_acf = []
            all_lb_pass = []
            all_lb_pvals = []
            for r in successful:
                gof_d = r["gof"].get(dim_key, {})
                if "acf_values" in gof_d and len(gof_d["acf_values"]) > 0:
                    all_acf.append(gof_d["acf_values"])
                if "ljung_box_pass" in gof_d:
                    all_lb_pass.append(1 if gof_d["ljung_box_pass"] else 0)
                if "ljung_box_pvalues" in gof_d and len(gof_d["ljung_box_pvalues"]) > 0:
                    all_lb_pvals.append(gof_d["ljung_box_pvalues"])
            
            if len(all_acf) >= 1:
                acf_arr = np.array(all_acf)
                n_lags = acf_arr.shape[1]
                lags = np.arange(1, n_lags + 1)
                acf_mean = np.mean(acf_arr, axis=0)
                acf_q25 = np.percentile(acf_arr, 25, axis=0)
                acf_q75 = np.percentile(acf_arr, 75, axis=0)
                
                # 95%置信区间（近似 ±1.96/√n，取典型n=5000）
                ci_bound = 1.96 / np.sqrt(5000)
                ax.fill_between(lags, -ci_bound, ci_bound,
                                alpha=0.15, color='blue', label='95% CI')
                
                # IQR带
                if len(all_acf) > 2:
                    ax.fill_between(lags, acf_q25, acf_q75,
                                    alpha=0.2, color=group_colors[row])
                
                # 组均值ACF
                ax.bar(lags, acf_mean, width=0.6, color=group_colors[row],
                       alpha=0.7, edgecolor='none')
                ax.axhline(y=0, color='black', linewidth=0.5)
                
                # Ljung-Box标注
                lb_pass_ratio = np.mean(all_lb_pass) if len(all_lb_pass) > 0 else 0
                lb_mean_p = np.mean(all_lb_pvals, axis=0) if len(all_lb_pvals) > 0 else []
                lb_text = f"LB pass: {lb_pass_ratio:.0%}"
                if len(lb_mean_p) >= 3:
                    lb_text += f"\np(5)={lb_mean_p[0]:.3f}"
                    lb_text += f"\np(20)={lb_mean_p[2]:.3f}"
                ax.text(0.97, 0.97, lb_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
                
                ax.set_ylim(-0.15, 0.15)
                ax.set_xlim(0.5, n_lags + 0.5)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
            
            if row == 0:
                ax.set_title(dim_names[col], fontsize=12, fontweight='bold')
            if row == 2:
                ax.set_xlabel("Lag", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{group_labels[row]}\nACF", fontsize=9)
            
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, linewidth=0.5)
    
    plt.suptitle("Residual Independence: ACF + Ljung-Box Diagnostics",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acf_independence_panel.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved ACF independence panel to {output_dir}/acf_independence_panel.png")


def plot_gof_score_comparison(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    论文级组间对比图：
    左：GOF综合评分箱线图（3组×4维）
    右：多维度雷达图（每组一条线，轴=各GOF指标均值）
    """
    print("Generating GOF score comparison...")
    
    group_names = ["high", "mid", "low"]
    group_labels = ["High Price", "Mid Price", "Low Price"]
    dim_names = ["Buy Toxic", "Buy Not\nToxic", "Sell Toxic", "Sell Not\nToxic"]
    colors = ['#c0392b', '#e67e22', '#27ae60']
    
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    
    # ---- 左图：GOF score 箱线图 ----
    ax_left = fig.add_subplot(gs[0, 0])
    
    positions_all = []
    data_all = []
    colors_all = []
    tick_positions = []
    tick_labels = []
    
    for col in range(4):
        dim_key = f"dim_{col}"
        for gidx, group_name in enumerate(group_names):
            results = all_group_results.get(group_name, [])
            successful = [r for r in results if "error" not in r and "gof" in r]
            
            scores = [r["gof"][dim_key].get("gof_score", 0) for r in successful
                      if dim_key in r["gof"] and "gof_score" in r["gof"][dim_key]]
            
            pos = col * 4 + gidx + 1
            positions_all.append(pos)
            data_all.append(scores if len(scores) > 0 else [0])
            colors_all.append(colors[gidx])
        
        tick_positions.append(col * 4 + 2)
        tick_labels.append(dim_names[col])
    
    bp = ax_left.boxplot(data_all, positions=positions_all, widths=0.7, patch_artist=True,
                         showfliers=False)
    for patch, color in zip(bp['boxes'], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    # 散点
    rng = np.random.RandomState(42)
    for i, (data, pos) in enumerate(zip(data_all, positions_all)):
        x = rng.normal(pos, 0.1, size=len(data))
        ax_left.scatter(x, data, alpha=0.5, color=colors_all[i], s=15,
                        edgecolors='black', linewidth=0.3, zorder=3)
    
    ax_left.set_xticks(tick_positions)
    ax_left.set_xticklabels(tick_labels, fontsize=10)
    ax_left.set_ylabel("GOF Composite Score", fontsize=11)
    ax_left.set_title("GOF Score Distribution by Group × Dimension", fontsize=12, fontweight='bold')
    ax_left.set_ylim(0, 1.05)
    ax_left.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax_left.grid(axis='y', alpha=0.2)
    
    # 图例
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, group_labels)]
    ax_left.legend(handles=legend_patches, loc='lower left', fontsize=9)
    
    # ---- 右图：雷达图 ----
    ax_radar = fig.add_subplot(gs[0, 1], polar=True)
    
    # 雷达轴：Mean Score, W1 Score, LB Score, ACF Score (4维度均值)
    radar_labels = ['Mean\nAccuracy', 'Wasserstein\nFit', 'Ljung-Box\nIndep.', 'ACF\nIndep.']
    n_axes = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for gidx, group_name in enumerate(group_names):
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        if len(successful) == 0:
            continue
        
        # 计算各指标的组均值
        mean_scores = []
        w1_scores = []
        lb_scores = []
        acf_scores = []
        
        for r in successful:
            for d in range(4):
                dk = f"dim_{d}"
                gof_d = r["gof"].get(dk, {})
                if "mae_mean" in gof_d:
                    mean_scores.append(max(0, 1.0 - gof_d["mae_mean"]))
                if "wasserstein_1" in gof_d:
                    w1_scores.append(max(0, 1.0 - gof_d["wasserstein_1"] / 2.0))
                if "ljung_box_pvalues" in gof_d and len(gof_d["ljung_box_pvalues"]) > 0:
                    lb_scores.append(np.mean(gof_d["ljung_box_pvalues"]))
                if "acf_values" in gof_d and len(gof_d["acf_values"]) > 0:
                    acf_scores.append(1.0 - np.mean(np.abs(gof_d["acf_values"])))
        
        values = [
            np.mean(mean_scores) if len(mean_scores) > 0 else 0,
            np.mean(w1_scores) if len(w1_scores) > 0 else 0,
            np.mean(lb_scores) if len(lb_scores) > 0 else 0,
            np.mean(acf_scores) if len(acf_scores) > 0 else 0,
        ]
        values += values[:1]  # 闭合
        
        ax_radar.plot(angles, values, 'o-', color=colors[gidx], linewidth=2,
                      markersize=5, label=group_labels[gidx])
        ax_radar.fill(angles, values, alpha=0.15, color=colors[gidx])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax_radar.set_title("Multi-Metric GOF Radar", fontsize=12, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax_radar.grid(True, alpha=0.3)
    
    plt.suptitle("Cross-Group GOF Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, "gof_score_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved GOF score comparison to {output_dir}/gof_score_comparison.png")


def plot_excitation_matrix_heatmaps(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    绘制激励矩阵A的组均值热力图
    """
    print("Generating excitation matrix heatmaps...")
    
    # 使用GridSpec为colorbar预留空间，避免重叠
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    
    group_names = ["high", "mid", "low"]
    dim_labels = ["Buy\nToxic", "Buy\nNot Toxic", "Sell\nToxic", "Sell\nNot Toxic"]
    
    vmin, vmax = 0, 1.0
    im = None
    
    for idx, group_name in enumerate(group_names):
        ax = fig.add_subplot(gs[0, idx])
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r]
        
        if len(successful) > 0:
            As = np.array([r["full"]["A"] for r in successful], dtype=float)
            A_mean = np.mean(As, axis=0)
            
            im = ax.imshow(A_mean, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='equal')
            
            for i in range(4):
                for j in range(4):
                    val = A_mean[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)
            
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(dim_labels, fontsize=9)
            ax.set_yticklabels(dim_labels, fontsize=9)
            ax.set_title(f"{group_name.upper()} Price\n(n={len(successful)})", fontsize=12, fontweight='bold')
            ax.set_xlabel("Target Event (j)", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Source Event (i)", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{group_name.upper()} Price", fontsize=12)
    
    # 在预留的空间添加colorbar
    if im is not None:
        cbar_ax = fig.add_subplot(gs[0, 3])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Excitation Coefficient A[i,j]', fontsize=10)
    
    plt.suptitle("Excitation Matrix A (Group Mean) - with re_spread Exogenous", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, "excitation_matrix_heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved excitation matrix heatmaps to {output_dir}/excitation_matrix_heatmaps.png")


def plot_branching_ratio_boxplot(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    绘制分枝比箱线图
    """
    print("Generating branching ratio boxplot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    group_names = ["high", "mid", "low"]
    group_labels = ["High Price", "Mid Price", "Low Price"]
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    data_to_plot = []
    positions = []
    
    for idx, group_name in enumerate(group_names):
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r]
        
        if len(successful) > 0:
            br = [r["full"]["branching_ratio"] for r in successful]
            data_to_plot.append(br)
            positions.append(idx + 1)
    
    if len(data_to_plot) > 0:
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for idx, (data, pos) in enumerate(zip(data_to_plot, positions)):
            x = np.random.normal(pos, 0.08, size=len(data))
            ax.scatter(x, data, alpha=0.6, color=colors[idx], s=30, edgecolors='black', linewidth=0.5)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Stability threshold (BR=1)')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels[:len(positions)], fontsize=11)
        ax.set_ylabel("Branching Ratio", fontsize=12)
        ax.set_xlabel("Price Group", fontsize=12)
        ax.set_title("Branching Ratio Distribution (with re_spread Exogenous)", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        for idx, (data, pos) in enumerate(zip(data_to_plot, positions)):
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(pos, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                   ha='center', va='top', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "branching_ratio_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved branching ratio boxplot to {output_dir}/branching_ratio_boxplot.png")


def plot_gof_summary_dashboard(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    论文级GOF综合仪表板：
    上行：残差均值箱线图（3组，每组4维）
    下行：GOF多指标热力图（行=股票，列=指标，按组分面板）
    """
    print("Generating GOF summary dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    group_names = ["high", "mid", "low"]
    group_labels = ["High Price", "Mid Price", "Low Price"]
    dim_names = ["BT", "BN", "ST", "SN"]
    dim_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    # ---- 上行：残差均值箱线图 ----
    for idx, group_name in enumerate(group_names):
        ax = axes[0, idx]
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        residual_data = {d: [] for d in range(4)}
        for r in successful:
            for d in range(4):
                dim_key = f"dim_{d}"
                if dim_key in r["gof"] and "mean" in r["gof"][dim_key]:
                    residual_data[d].append(r["gof"][dim_key]["mean"])
        
        data_to_plot = []
        labels_to_plot = []
        for d in range(4):
            if len(residual_data[d]) > 0:
                data_to_plot.append(residual_data[d])
                labels_to_plot.append(dim_names[d])
        
        if len(data_to_plot) > 0:
            bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                           widths=0.6, showfliers=False)
            for patch, color in zip(bp['boxes'], dim_colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(0.5)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
            
            rng = np.random.RandomState(42)
            for didx, data in enumerate(data_to_plot):
                x = rng.normal(didx + 1, 0.06, size=len(data))
                ax.scatter(x, data, alpha=0.5, color=dim_colors[didx], s=20,
                          edgecolors='black', linewidth=0.3, zorder=3)
            
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='E[r]=1')
            ax.axhspan(0.8, 1.2, alpha=0.08, color='green', label='[0.8, 1.2]')
        
        ax.set_ylabel("Residual Mean" if idx == 0 else "", fontsize=10)
        ax.set_title(f"{group_labels[idx]}", fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    # ---- 下行：GOF评分热力图（每组一个面板）----
    for idx, group_name in enumerate(group_names):
        ax = axes[1, idx]
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        if len(successful) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue
        
        # 构建矩阵：行=股票，列=4维度的GOF score
        stock_labels = []
        score_matrix = []
        for r in successful:
            code = r.get("stock_code", "?")
            stock_labels.append(code)
            row_scores = []
            for d in range(4):
                dk = f"dim_{d}"
                row_scores.append(r["gof"].get(dk, {}).get("gof_score", 0))
            score_matrix.append(row_scores)
        
        score_arr = np.array(score_matrix)
        im = ax.imshow(score_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(4))
        ax.set_xticklabels(dim_names, fontsize=10)
        ax.set_yticks(range(len(stock_labels)))
        ax.set_yticklabels(stock_labels, fontsize=7)
        ax.set_title(f"{group_labels[idx]} — GOF Score", fontsize=11, fontweight='bold')
        
        # 标注数值
        for i in range(len(stock_labels)):
            for j in range(4):
                val = score_arr[i, j]
                color = 'white' if val < 0.4 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=7, fontweight='bold')
        
        if idx == 2:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('GOF Score', fontsize=9)
            cbar.ax.tick_params(labelsize=7)
    
    plt.suptitle("GOF Summary Dashboard: Residual Mean + Composite Score",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gof_summary_dashboard.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved GOF summary dashboard to {output_dir}/gof_summary_dashboard.png")


def plot_gamma_spread_comparison(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    绘制gamma_spread外生项系数对比图
    """
    print("Generating gamma_spread comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    group_names = ["high", "mid", "low"]
    group_labels = ["High Price", "Mid Price", "Low Price"]
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    dim_names = ["Buy Toxic", "Buy Not Toxic", "Sell Toxic", "Sell Not Toxic"]
    
    # 左图：各组gamma_spread均值对比
    ax1 = axes[0]
    x_pos = np.arange(4)
    width = 0.25
    
    for idx, group_name in enumerate(group_names):
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r]
        
        gamma_spreads = [r["exog"]["gamma_spread"] for r in successful 
                        if "exog" in r and r["exog"].get("gamma_spread")]
        
        if len(gamma_spreads) > 0:
            gamma_means = np.mean(gamma_spreads, axis=0)
            gamma_stds = np.std(gamma_spreads, axis=0)
            
            ax1.bar(x_pos + idx * width, gamma_means, width, 
                   label=group_labels[idx], color=colors[idx], alpha=0.7,
                   yerr=gamma_stds, capsize=3)
    
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(dim_names, fontsize=10)
    ax1.set_ylabel("γ_spread", fontsize=12)
    ax1.set_xlabel("Event Dimension", fontsize=12)
    ax1.set_title("γ_spread by Price Group and Dimension", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图：gamma_spread箱线图（按价格组）
    ax2 = axes[1]
    
    data_by_group = []
    for group_name in group_names:
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r]
        
        gamma_spreads = [r["exog"]["gamma_spread"] for r in successful 
                        if "exog" in r and r["exog"].get("gamma_spread")]
        
        if len(gamma_spreads) > 0:
            # 取所有维度的均值作为该股票的综合spread效应
            means = [np.mean(gs) for gs in gamma_spreads]
            data_by_group.append(means)
        else:
            data_by_group.append([])
    
    valid_data = [d for d in data_by_group if len(d) > 0]
    valid_labels = [group_labels[i] for i, d in enumerate(data_by_group) if len(d) > 0]
    
    if len(valid_data) > 0:
        bp = ax2.boxplot(valid_data, labels=valid_labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors[:len(valid_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for idx, (data, label) in enumerate(zip(valid_data, valid_labels)):
            x = np.random.normal(idx + 1, 0.08, size=len(data))
            ax2.scatter(x, data, alpha=0.6, color=colors[idx], s=30, edgecolors='black', linewidth=0.5)
    
    ax2.set_ylabel("Mean γ_spread (all dimensions)", fontsize=12)
    ax2.set_xlabel("Price Group", fontsize=12)
    ax2.set_title("γ_spread Distribution by Price Group", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle("re_spread Exogenous Effect Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gamma_spread_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved gamma_spread comparison to {output_dir}/gamma_spread_comparison.png")


def generate_all_visualizations(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    生成所有可视化图表（v2：QQ + 距离度量 + 独立性 + 组间对比）
    """
    print(f"\n{'='*70}")
    print("Generating visualizations (GOF v2)...")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. QQ-Exp(1) 诊断图（3组×4维面板）
    plot_gof_qq_residuals(all_group_results, output_dir)
    
    # 2. 距离度量热力图（Wasserstein / QQ-MAE / |mean-1|）
    plot_distance_metrics_heatmap(all_group_results, output_dir)
    
    # 3. ACF独立性面板（3组×4维 ACF + Ljung-Box）
    plot_independence_acf_panel(all_group_results, output_dir)
    
    # 4. 组间对比（GOF score箱线图 + 雷达图）
    plot_gof_score_comparison(all_group_results, output_dir)
    
    # 5. GOF综合仪表板（残差均值 + 评分热力图）
    plot_gof_summary_dashboard(all_group_results, output_dir)
    
    # 6. 激励矩阵热力图
    plot_excitation_matrix_heatmaps(all_group_results, output_dir)
    
    # 7. 分枝比箱线图
    plot_branching_ratio_boxplot(all_group_results, output_dir)
    
    # 8. gamma_spread对比图
    plot_gamma_spread_comparison(all_group_results, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}/")


def main():
    """主函数：处理三个价格分组"""
    
    price_groups = {
        "high": {
            "data_dir": "data/high_price_events",
            "output_dir": "results_exog/high_price_4d_exog",
        },
        "mid": {
            "data_dir": "data/mid_price_events", 
            "output_dir": "results_exog/mid_price_4d_exog",
        },
        "low": {
            "data_dir": "data/low_price_events",
            "output_dir": "results_exog/low_price_4d_exog",
        },
    }
    
    all_group_results = {}
    
    for group_name, config in price_groups.items():
        results = process_price_group_exog(
            config["data_dir"], 
            config["output_dir"], 
            group_name
        )
        all_group_results[group_name] = results
    
    # 保存组间比较（GOF v2指标）
    comparison = {}
    for group_name, results in all_group_results.items():
        successful = [r for r in results if "error" not in r]
        if len(successful) > 0:
            # GOF v2 指标
            gof_score_means = [r["gof"]["summary"].get("gof_score_mean", 0) for r in successful if "gof" in r]
            w1_all = []
            qq_mae_all = []
            for r in successful:
                for d in range(4):
                    dk = f"dim_{d}"
                    if "gof" in r and dk in r["gof"]:
                        if "wasserstein_1" in r["gof"][dk]:
                            w1_all.append(r["gof"][dk]["wasserstein_1"])
                        if "qq_mae" in r["gof"][dk]:
                            qq_mae_all.append(r["gof"][dk]["qq_mae"])
            
            comparison[group_name] = {
                "n_stocks": len(results),
                "n_successful": len(successful),
                "branching_ratio_mean": float(np.mean([r["full"]["branching_ratio"] for r in successful])),
                "branching_ratio_std": float(np.std([r["full"]["branching_ratio"] for r in successful])),
                "decay_mean": float(np.mean([r["full"]["decay"] for r in successful])),
                "stable_ratio": float(sum(1 for r in successful if r["full"]["branching_ratio"] < 1.0) / len(successful)),
                "gof_pass_ratio": float(sum(1 for r in successful if r["gof"]["summary"]["all_pass"]) / len(successful)),
                "gof_score_mean": float(np.mean(gof_score_means)) if len(gof_score_means) > 0 else 0,
                "wasserstein_mean": float(np.mean(w1_all)) if len(w1_all) > 0 else 0,
                "qq_mae_mean": float(np.mean(qq_mae_all)) if len(qq_mae_all) > 0 else 0,
            }
            
            # gamma_spread 统计
            gamma_spreads = [r["exog"]["gamma_spread"] for r in successful 
                           if "exog" in r and r["exog"].get("gamma_spread")]
            if len(gamma_spreads) > 0:
                comparison[group_name]["gamma_spread_mean"] = np.mean(gamma_spreads, axis=0).tolist()
    
    os.makedirs("results_exog", exist_ok=True)
    with open("results_exog/price_groups_comparison_exog.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # 生成可视化图表
    generate_all_visualizations(all_group_results, "results_exog")
    
    print(f"\n{'='*70}")
    print("All processing complete!")
    print(f"Results saved to results_exog/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
