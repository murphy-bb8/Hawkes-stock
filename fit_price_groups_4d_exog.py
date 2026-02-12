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
from scipy.stats import probplot

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
        },
    }
    
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
    print(f"  GOF pass count: {summary['gof']['all_pass_count']}/{len(successful)}")
    if "gamma_spread" in summary:
        print(f"  gamma_spread mean: {summary['gamma_spread']['mean']}")
    print(f"  Summary saved to {summary_file}")


def plot_gof_qq_residuals(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    绘制GOF残差Q-Q图
    对每个价格组的每个维度绘制Q-Q图，检验残差是否服从Exp(1)分布
    """
    print("Generating GOF Q-Q plots...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    group_names = ["high", "mid", "low"]
    dim_names = ["Buy Toxic", "Buy Not Toxic", "Sell Toxic", "Sell Not Toxic"]
    
    for row, group_name in enumerate(group_names):
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        for col in range(4):
            ax = axes[row, col]
            dim_key = f"dim_{col}"
            
            residual_means = []
            for r in successful:
                if dim_key in r["gof"] and "mean" in r["gof"][dim_key]:
                    residual_means.append(r["gof"][dim_key]["mean"])
            
            if len(residual_means) >= 3:
                residual_means = np.array(residual_means)
                probplot(residual_means, dist="norm", plot=ax)
                ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Expected mean=1')
                
                mean_val = np.mean(residual_means)
                ax.set_title(f"{group_name.upper()} - {dim_names[col]}\nMean={mean_val:.3f}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{group_name.upper()} - {dim_names[col]}", fontsize=10)
            
            if row == 2:
                ax.set_xlabel("Theoretical Quantiles")
            if col == 0:
                ax.set_ylabel(f"{group_name.upper()}\nSample Quantiles")
    
    plt.suptitle("GOF Residual Q-Q Plots by Price Group and Event Type (with re_spread)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gof_qq_plots.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved GOF Q-Q plots to {output_dir}/gof_qq_plots.png")


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


def plot_independence_diagnostics(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    绘制独立性诊断图
    """
    print("Generating independence diagnostics...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    group_names = ["high", "mid", "low"]
    group_labels = ["High Price", "Mid Price", "Low Price"]
    
    # 第一行：各组的KS检验p值分布
    for idx, group_name in enumerate(group_names):
        ax = axes[0, idx]
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        ks_pvalues = {f"dim_{d}": [] for d in range(4)}
        for r in successful:
            for d in range(4):
                dim_key = f"dim_{d}"
                if dim_key in r["gof"] and "ks_pvalue" in r["gof"][dim_key]:
                    ks_pvalues[dim_key].append(r["gof"][dim_key]["ks_pvalue"])
        
        dim_labels = ["Buy\nToxic", "Buy\nNot Toxic", "Sell\nToxic", "Sell\nNot Toxic"]
        x_pos = np.arange(4)
        
        means = []
        for d in range(4):
            pvals = ks_pvalues[f"dim_{d}"]
            if len(pvals) > 0:
                means.append(np.mean(pvals))
            else:
                means.append(0)
        
        bars = ax.bar(x_pos, means, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'], alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dim_labels, fontsize=9)
        ax.set_ylabel("Mean KS p-value", fontsize=10)
        ax.set_title(f"{group_labels[idx]}\nKS Test p-values", fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(0.1, max(means) * 1.2) if max(means) > 0 else 0.1)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # 第二行：残差均值偏离度
    for idx, group_name in enumerate(group_names):
        ax = axes[1, idx]
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        residual_data = {d: [] for d in range(4)}
        for r in successful:
            for d in range(4):
                dim_key = f"dim_{d}"
                if dim_key in r["gof"] and "mean" in r["gof"][dim_key]:
                    residual_data[d].append(r["gof"][dim_key]["mean"])
        
        data_to_plot = [residual_data[d] for d in range(4) if len(residual_data[d]) > 0]
        if len(data_to_plot) > 0:
            bp = ax.boxplot(data_to_plot, labels=[f"Dim {d}" for d in range(4) if len(residual_data[d]) > 0],
                           patch_artist=True)
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Expected mean=1')
        
        ax.set_ylabel("Residual Mean", fontsize=10)
        ax.set_xlabel("Event Dimension", fontsize=10)
        ax.set_title(f"{group_labels[idx]}\nResidual Mean Distribution", fontsize=11, fontweight='bold')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Independence Diagnostics (with re_spread Exogenous)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "independence_diagnostics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved independence diagnostics to {output_dir}/independence_diagnostics.png")


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
    生成所有可视化图表
    """
    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. GOF残差Q-Q图
    plot_gof_qq_residuals(all_group_results, output_dir)
    
    # 2. 激励矩阵热力图
    plot_excitation_matrix_heatmaps(all_group_results, output_dir)
    
    # 3. 分枝比箱线图
    plot_branching_ratio_boxplot(all_group_results, output_dir)
    
    # 4. 独立性诊断图
    plot_independence_diagnostics(all_group_results, output_dir)
    
    # 5. gamma_spread对比图（新增）
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
    
    # 保存组间比较
    comparison = {}
    for group_name, results in all_group_results.items():
        successful = [r for r in results if "error" not in r]
        if len(successful) > 0:
            comparison[group_name] = {
                "n_stocks": len(results),
                "n_successful": len(successful),
                "branching_ratio_mean": float(np.mean([r["full"]["branching_ratio"] for r in successful])),
                "branching_ratio_std": float(np.std([r["full"]["branching_ratio"] for r in successful])),
                "decay_mean": float(np.mean([r["full"]["decay"] for r in successful])),
                "stable_ratio": float(sum(1 for r in successful if r["full"]["branching_ratio"] < 1.0) / len(successful)),
                "gof_pass_ratio": float(sum(1 for r in successful if r["gof"]["summary"]["all_pass"]) / len(successful)),
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
