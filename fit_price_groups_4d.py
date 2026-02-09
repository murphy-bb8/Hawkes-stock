"""
按价格分组的4D Hawkes模型拟合实验
处理 high_price_events, mid_price_events, low_price_events 三组数据
每组15只股票，共45只股票
"""
import os
import json
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from scipy import stats

from fit_toxic_events import extract_event_times, extract_event_times_with_day_offset
from hawkes_4d_tick import run_comparison_4d_tick, load_events_4d


def load_single_stock_data(data_path: str) -> Dict:
    """加载单只股票的事件数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_4d_events(stock_data: Dict) -> Dict:
    """
    构建4D事件数据
    维度0: buy_toxic
    维度1: buy_not_toxic
    维度2: sell_toxic
    维度3: sell_not_toxic
    
    重要修复：
    - 使用带日期偏移的时间，确保多日数据正确排序
    - events: 归一化后的连续时间（用于tick拟合）
    - events_original: 日内时间（用于OPEN30/MID30/CLOSE30判断）
    
    Returns:
    --------
    dict包含：
        events: 归一化后的事件时间（用于tick拟合，从0开始连续）
        events_original: 日内时间（用于GOF的时段判断，34200-54000范围）
        T: 总时间长度
        counts: 各维度事件数
    """
    event_types = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    events_with_offset = []  # 带日期偏移的连续时间
    events_intraday = []     # 日内时间（用于时段判断）
    
    for et in event_types:
        # 使用新的提取函数，返回带偏移的时间和日内时间
        times_offset, times_intra = extract_event_times_with_day_offset(stock_data, et)
        if len(times_offset) > 0:
            events_with_offset.append(times_offset.astype(float))
            events_intraday.append(times_intra.astype(float))
        else:
            events_with_offset.append(np.asarray([], dtype=float))
            events_intraday.append(np.asarray([], dtype=float))
    
    # 合并所有事件计算全局时间范围
    all_times = np.concatenate([ev for ev in events_with_offset if len(ev) > 0]) if any(len(ev) > 0 for ev in events_with_offset) else np.asarray([], dtype=float)
    
    if len(all_times) == 0:
        return {"events": [], "events_original": [], "T": 0.0, "counts": [0, 0, 0, 0]}
    
    # 归一化：从0开始（用于tick拟合）
    t0 = float(np.min(all_times))
    events_norm = [ev - t0 for ev in events_with_offset]
    T = float(np.max(all_times) - t0)
    counts = [int(len(ev)) for ev in events_with_offset]
    
    # 对每个维度按归一化时间排序，同时保持日内时间的对应关系
    events_sorted = []
    events_intraday_sorted = []
    for i in range(4):
        if len(events_norm[i]) > 0:
            # 获取排序索引
            sort_idx = np.argsort(events_norm[i])
            events_sorted.append(events_norm[i][sort_idx])
            events_intraday_sorted.append(events_intraday[i][sort_idx])
        else:
            events_sorted.append(events_norm[i])
            events_intraday_sorted.append(events_intraday[i])
    
    return {
        "events": events_sorted, 
        "events_original": events_intraday_sorted,  # 日内时间用于时段判断
        "T": T, 
        "counts": counts
    }


def fit_stock_4d_tick(stock_code: str, stock_data: Dict, output_dir: str) -> Dict:
    """
    对单只股票进行4D Hawkes拟合（使用tick库）
    GOF检验使用分段基准强度（I_OPEN30, I_CLOSE30）
    """
    built = build_4d_events(stock_data)
    events_4d = built["events"]
    events_4d_original = built["events_original"]  # 原始时间用于GOF时段判断
    counts = built["counts"]
    total_events = sum(counts)
    
    print(f"  Stock {stock_code}: events={counts}, total={total_events}")
    
    if total_events < 20:
        print(f"    -> Skipped: insufficient events")
        return {
            "stock_code": stock_code,
            "event_type": "4d",
            "error": "insufficient_events",
            "n_events": counts,
        }
    
    # 创建临时文件（包含归一化时间和原始时间）
    os.makedirs("temp", exist_ok=True)
    temp_file = f"temp/events_{stock_code}_4d_price.json"
    
    # payload包含归一化时间t和原始时间t_orig
    payload = []
    for dim, (ev_norm, ev_orig) in enumerate(zip(events_4d, events_4d_original)):
        for t_norm, t_orig in zip(ev_norm, ev_orig):
            payload.append({
                "t": float(t_norm), 
                "i": int(dim),
                "t_orig": float(t_orig)  # 原始时间用于GOF
            })
    payload.sort(key=lambda x: x["t"])
    
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    
    # 设置输出标签
    os.environ["OUT_TAG"] = f"{stock_code}_4d_price"
    
    try:
        # 调用4D tick拟合，传入原始时间用于GOF
        result = run_comparison_4d_tick(temp_file, events_4d_original=events_4d_original)
        
        # 添加元信息
        result["stock_code"] = stock_code
        result["event_type"] = "4d"
        result["n_events"] = counts
        result["T"] = float(built["T"])
        
        # 保存单只股票结果
        save_per_stock = os.getenv("SAVE_PER_STOCK", "1")
        if save_per_stock != "0":
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{stock_code}_4d.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        gof_score = result.get("gof", {}).get("summary", {}).get("gof_score_mean", 0)
        gof_pass = result.get("gof", {}).get("summary", {}).get("gof_pass_count", 0)
        print(f"    -> Success: decay={result['full']['decay']:.4f}, "
              f"BR={result['full']['branching_ratio']:.4f}, GOF score={gof_score:.3f}, pass={gof_pass}/4")
        return result
        
    except Exception as e:
        print(f"    -> Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "stock_code": stock_code,
            "event_type": "4d",
            "error": str(e),
            "n_events": counts,
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def generate_group_summary(all_results: List[Dict], output_dir: str, group_name: str) -> None:
    """生成分组汇总报告 - GOF v2"""
    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    
    print(f"\n{'-'*50}")
    print(f"Group {group_name} Summary (time-varying μ):")
    print(f"  Total: {len(all_results)}, Successful: {len(successful)}, Failed: {len(failed)}")
    
    if len(successful) == 0:
        print("  No successful fits to summarize")
        return
    
    branching_ratios = [r["full"]["branching_ratio"] for r in successful]
    decays = [r["full"]["decay"] for r in successful]
    
    # 收集gamma参数
    gamma_opens = []
    gamma_mids = []
    gamma_closes = []
    for r in successful:
        if "gof" in r and "gamma" in r["gof"]:
            gamma_opens.append(r["gof"]["gamma"]["gamma_open"])
            if "gamma_mid" in r["gof"]["gamma"]:
                gamma_mids.append(r["gof"]["gamma"]["gamma_mid"])
            gamma_closes.append(r["gof"]["gamma"]["gamma_close"])
    
    # 收集GOF v2统计
    gof_scores = {d: [] for d in range(4)}
    wasserstein_vals = {d: [] for d in range(4)}
    qq_mae_vals = {d: [] for d in range(4)}
    mae_mean_vals = {d: [] for d in range(4)}
    lb_pass_counts = {d: [] for d in range(4)}
    for r in successful:
        gof = r.get("gof", {})
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
            "model_type": "time_varying_mu",
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
    
    # mu统计
    mus = np.array([r["full"]["mu"] for r in successful], dtype=float)
    summary["mu_mean"] = np.mean(mus, axis=0).tolist()
    summary["mu_std"] = np.std(mus, axis=0).tolist()
    
    # A矩阵统计
    As = np.array([r["full"]["A"] for r in successful], dtype=float)
    summary["A_mean"] = np.mean(As, axis=0).tolist()
    
    # AIC统计
    aics = [r["full"]["aic"] for r in successful]
    summary["aic"] = {"mean": float(np.mean(aics)), "std": float(np.std(aics))}
    
    # 保存
    report_file = os.path.join(output_dir, "summary_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"  Branching ratio: {summary['branching_ratio']['mean']:.4f} ± {summary['branching_ratio']['std']:.4f}")
    print(f"  Stable count: {summary['branching_ratio']['stable_count']}/{len(successful)}")
    print(f"  GOF score: {summary['gof']['gof_score_mean']:.3f}, pass={summary['gof']['all_pass_count']}/{len(successful)}")
    print(f"  Summary saved to {report_file}")


def process_price_group(group_name: str, data_dir: str, output_dir: str) -> List[Dict]:
    """
    处理一个价格分组的所有股票
    
    Parameters:
    -----------
    group_name : str
        分组名称（high, mid, low）
    data_dir : str
        数据目录路径
    output_dir : str
        输出目录路径
    
    Returns:
    --------
    List[Dict]: 所有拟合结果
    """
    print(f"\n{'='*70}")
    print(f"Processing {group_name} price group")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    # 获取所有股票文件（排除all_events文件）
    stock_files = [f for f in os.listdir(data_dir) 
                   if f.startswith("events_") and f.endswith(".json") 
                   and not f.startswith("all_events")]
    
    print(f"Found {len(stock_files)} stock files")
    
    all_results = []
    
    for stock_file in sorted(stock_files):
        # 提取股票代码
        # 文件名格式: events_600036_201912.json
        parts = stock_file.replace("events_", "").replace(".json", "").split("_")
        stock_code = parts[0]
        
        # 加载数据
        data_path = os.path.join(data_dir, stock_file)
        stock_data = load_single_stock_data(data_path)
        
        # 拟合
        result = fit_stock_4d_tick(stock_code, stock_data, output_dir)
        all_results.append(result)
    
    # 保存汇总结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有结果
    summary_file = os.path.join(output_dir, "summary_all_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nAll results saved to {summary_file}")
    
    # 生成汇总报告
    generate_group_summary(all_results, output_dir, group_name)
    
    return all_results


def plot_gof_qq_residuals(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """论文级 QQ-Exp(1) 图：3组×4维面板"""
    from scipy.stats import expon
    print("Generating QQ-Exp(1) plots...")
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    group_names = ["high", "mid", "low"]
    dim_names = ["BuyToxic", "BuyNotToxic", "SellToxic", "SellNotToxic"]
    group_colors = {"high": "#e74c3c", "mid": "#f39c12", "low": "#27ae60"}
    
    for row, group_name in enumerate(group_names):
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        for col in range(4):
            ax = axes[row, col]
            dim_key = f"dim_{col}"
            
            all_qq_emp = []
            all_qq_theo = []
            for r in successful:
                gof = r.get("gof", {})
                if dim_key in gof and "qq_empirical" in gof[dim_key]:
                    all_qq_emp.append(gof[dim_key]["qq_empirical"])
                    all_qq_theo.append(gof[dim_key]["qq_theoretical"])
            
            if len(all_qq_emp) > 0:
                qq_emp_arr = np.array(all_qq_emp)
                qq_theo = np.array(all_qq_theo[0])
                qq_emp_median = np.median(qq_emp_arr, axis=0)
                qq_emp_q25 = np.percentile(qq_emp_arr, 25, axis=0)
                qq_emp_q75 = np.percentile(qq_emp_arr, 75, axis=0)
                
                ax.fill_between(qq_theo, qq_emp_q25, qq_emp_q75, alpha=0.25, color=group_colors[group_name])
                ax.plot(qq_theo, qq_emp_median, 'o', color=group_colors[group_name], markersize=2, alpha=0.8)
                max_val = max(qq_theo.max(), qq_emp_q75.max()) * 1.05
                ax.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.6)
                
                w1_vals = [r["gof"][dim_key].get("wasserstein_1", 0) for r in successful if dim_key in r.get("gof", {})]
                qq_mae_vals = [r["gof"][dim_key].get("qq_mae", 0) for r in successful if dim_key in r.get("gof", {})]
                if w1_vals:
                    ax.text(0.05, 0.92, f'W1={np.mean(w1_vals):.2f}\nQQ={np.mean(qq_mae_vals):.2f}',
                            transform=ax.transAxes, fontsize=8, va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No QQ data", ha='center', va='center', transform=ax.transAxes)
            
            if row == 0:
                ax.set_title(dim_names[col], fontsize=11, fontweight='bold')
            if row == 2:
                ax.set_xlabel("Exp(1) Theoretical", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{group_name.upper()}\nEmpirical", fontsize=10)
            ax.grid(True, alpha=0.2)
    
    plt.suptitle("QQ-Exp(1) Diagnostic (Time-Varying μ Model)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gof_qq_plots.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved QQ plots to {output_dir}/gof_qq_plots.png")


def plot_distance_metrics_heatmap(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """距离度量热力图：Wasserstein / QQ-MAE / |mean-1|"""
    print("Generating distance metrics heatmap...")
    
    group_names = ["high", "mid", "low"]
    dim_names = ["BT", "BN", "ST", "SN"]
    metrics = ["wasserstein_1", "qq_mae", "mae_mean"]
    metric_labels = ["Wasserstein-1", "QQ-MAE", "|Mean−1|"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    for m_idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[m_idx]
        data = np.zeros((3, 4))
        for g_idx, gn in enumerate(group_names):
            results = all_group_results.get(gn, [])
            successful = [r for r in results if "error" not in r and "gof" in r]
            for d in range(4):
                dk = f"dim_{d}"
                vals = [r["gof"][dk].get(metric, 0) for r in successful if dk in r.get("gof", {}) and metric in r["gof"].get(dk, {})]
                data[g_idx, d] = np.mean(vals) if vals else 0
        
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center', fontsize=10)
        ax.set_xticks(range(4))
        ax.set_xticklabels(dim_names, fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels([g.upper() for g in group_names], fontsize=10)
        ax.set_title(mlabel, fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle("Distance Metrics (Time-Varying μ Model)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_metrics_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved distance metrics heatmap to {output_dir}/distance_metrics_heatmap.png")


def plot_independence_acf_panel(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """ACF独立性面板：3组×4维"""
    print("Generating ACF independence panel...")
    
    group_names = ["high", "mid", "low"]
    dim_names = ["BuyToxic", "BuyNotToxic", "SellToxic", "SellNotToxic"]
    group_colors = {"high": "#e74c3c", "mid": "#f39c12", "low": "#27ae60"}
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    
    for row, gn in enumerate(group_names):
        results = all_group_results.get(gn, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        for col in range(4):
            ax = axes[row, col]
            dk = f"dim_{col}"
            
            all_acf = []
            lb_pvals = []
            for r in successful:
                gof = r.get("gof", {})
                if dk in gof and "acf_values" in gof[dk] and len(gof[dk]["acf_values"]) > 0:
                    all_acf.append(gof[dk]["acf_values"])
                if dk in gof and "ljung_box_pvalues" in gof[dk]:
                    lb_pvals.append(gof[dk]["ljung_box_pvalues"])
            
            if len(all_acf) > 0:
                acf_arr = np.array(all_acf)
                acf_mean = np.mean(acf_arr, axis=0)
                lags = np.arange(1, len(acf_mean) + 1)
                n_mean = np.mean([len(r["gof"][dk].get("acf_values", [])) for r in successful if dk in r.get("gof", {})])
                ci = 1.96 / np.sqrt(max(n_mean, 1))
                
                ax.bar(lags, acf_mean, color=group_colors[gn], alpha=0.7, width=0.8)
                ax.axhline(y=ci, color='blue', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(y=-ci, color='blue', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(y=0, color='black', linewidth=0.5)
                
                if lb_pvals:
                    mean_lb = np.mean([np.mean(p) for p in lb_pvals if len(p) > 0])
                    lb_pass_ratio = np.mean([all(p > 0.05 for p in pv) for pv in lb_pvals if len(pv) > 0])
                    ax.text(0.95, 0.95, f'LB p={mean_lb:.3f}\nPass={lb_pass_ratio:.0%}',
                            transform=ax.transAxes, fontsize=7, ha='right', va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if row == 0:
                ax.set_title(dim_names[col], fontsize=11, fontweight='bold')
            if row == 2:
                ax.set_xlabel("Lag", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{gn.upper()}\nACF", fontsize=10)
            ax.set_ylim(-0.15, 0.15)
            ax.grid(True, alpha=0.2, axis='y')
    
    plt.suptitle("ACF Independence Panel (Time-Varying μ Model)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acf_independence_panel.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved ACF independence panel to {output_dir}/acf_independence_panel.png")


def plot_gof_score_comparison(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """GOF综合评分箱线图 + 雷达图"""
    print("Generating GOF score comparison...")
    
    group_names = ["high", "mid", "low"]
    dim_names = ["BT", "BN", "ST", "SN"]
    group_colors = {"high": "#e74c3c", "mid": "#f39c12", "low": "#27ae60"}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # 左图：GOF score箱线图
    all_data = []
    all_labels = []
    all_colors = []
    positions = []
    pos = 0
    for gn in group_names:
        results = all_group_results.get(gn, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        for d in range(4):
            dk = f"dim_{d}"
            scores = [r["gof"][dk].get("gof_score", 0) for r in successful if dk in r.get("gof", {}) and "gof_score" in r["gof"].get(dk, {})]
            if scores:
                all_data.append(scores)
                all_labels.append(f"{gn[0].upper()}-{dim_names[d]}")
                all_colors.append(group_colors[gn])
                positions.append(pos)
                pos += 1
        pos += 0.5
    
    if all_data:
        bp = ax1.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_xticks(positions)
        ax1.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel("GOF Score", fontsize=11)
        ax1.set_title("GOF Score by Group × Dimension", fontsize=12, fontweight='bold')
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2, axis='y')
    
    # 右图：雷达图
    radar_metrics = ["GOF Score", "1−W1/2", "1−QQ_MAE", "LB Pass%", "1−|Mean−1|"]
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    ax2 = fig.add_subplot(122, polar=True)
    for gn in group_names:
        results = all_group_results.get(gn, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        if not successful:
            continue
        
        gof_scores_all = []
        w1_all = []
        qq_mae_all = []
        lb_all = []
        mae_all = []
        for r in successful:
            for d in range(4):
                dk = f"dim_{d}"
                g = r.get("gof", {}).get(dk, {})
                if "gof_score" in g:
                    gof_scores_all.append(g["gof_score"])
                    w1_all.append(max(0, 1 - g.get("wasserstein_1", 0) / 2))
                    qq_mae_all.append(max(0, 1 - g.get("qq_mae", 0)))
                    lb_all.append(1.0 if g.get("ljung_box_pass", False) else 0.0)
                    mae_all.append(max(0, 1 - g.get("mae_mean", 0)))
        
        if gof_scores_all:
            values = [np.mean(gof_scores_all), np.mean(w1_all), np.mean(qq_mae_all), np.mean(lb_all), np.mean(mae_all)]
            values += values[:1]
            ax2.plot(angles, values, 'o-', color=group_colors[gn], label=gn.upper(), linewidth=2, markersize=4)
            ax2.fill(angles, values, color=group_colors[gn], alpha=0.1)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(radar_metrics, fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_title("Multi-Metric Radar", fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gof_score_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved GOF score comparison to {output_dir}/gof_score_comparison.png")


def plot_gof_summary_dashboard(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """综合GOF仪表板：残差均值箱线图 + GOF评分热力图"""
    print("Generating GOF summary dashboard...")
    
    group_names = ["high", "mid", "low"]
    dim_names = ["BT", "BN", "ST", "SN"]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 上行：残差均值箱线图
    for g_idx, gn in enumerate(group_names):
        ax = axes[0, g_idx]
        results = all_group_results.get(gn, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        data_to_plot = []
        for d in range(4):
            dk = f"dim_{d}"
            means = [r["gof"][dk]["mean"] for r in successful if dk in r.get("gof", {}) and "mean" in r["gof"].get(dk, {})]
            data_to_plot.append(means if means else [0])
        
        bp = ax.boxplot(data_to_plot, labels=dim_names, patch_artist=True)
        colors_dim = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        for patch, c in zip(bp['boxes'], colors_dim):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_title(f"{gn.upper()} Price", fontsize=12, fontweight='bold')
        ax.set_ylabel("Residual Mean" if g_idx == 0 else "")
        ax.grid(True, alpha=0.2, axis='y')
    
    # 下行：GOF评分热力图
    for g_idx, gn in enumerate(group_names):
        ax = axes[1, g_idx]
        results = all_group_results.get(gn, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        if successful:
            stock_codes = [r.get("stock_code", f"S{i}") for i, r in enumerate(successful)]
            score_matrix = np.zeros((len(successful), 4))
            for i, r in enumerate(successful):
                for d in range(4):
                    dk = f"dim_{d}"
                    score_matrix[i, d] = r.get("gof", {}).get(dk, {}).get("gof_score", 0)
            
            im = ax.imshow(score_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
            ax.set_xticks(range(4))
            ax.set_xticklabels(dim_names, fontsize=9)
            ax.set_yticks(range(len(stock_codes)))
            ax.set_yticklabels(stock_codes, fontsize=7)
            ax.set_title(f"{gn.upper()} GOF Scores", fontsize=11, fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle("GOF Summary Dashboard (Time-Varying μ Model)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gof_summary_dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved GOF summary dashboard to {output_dir}/gof_summary_dashboard.png")


def plot_excitation_matrix_heatmaps(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """绘制激励矩阵A的组均值热力图"""
    print("Generating excitation matrix heatmaps...")
    
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
            ax.set_title(f"{group_name.upper()} (n={len(successful)})", fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
    
    if im is not None:
        cbar_ax = fig.add_subplot(gs[0, 3])
        fig.colorbar(im, cax=cbar_ax).set_label('Excitation Coefficient A[i,j]')
    
    plt.suptitle("Excitation Matrix A (Time-Varying μ Model)", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, "excitation_matrix_heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved excitation matrix heatmaps to {output_dir}/excitation_matrix_heatmaps.png")


def plot_branching_ratio_boxplot(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """绘制分枝比箱线图"""
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
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Stability (BR=1)')
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels[:len(positions)], fontsize=11)
        ax.set_ylabel("Branching Ratio", fontsize=12)
        ax.set_title("Branching Ratio (Time-Varying μ Model)", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "branching_ratio_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved branching ratio boxplot to {output_dir}/branching_ratio_boxplot.png")


def generate_all_visualizations(all_group_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """生成所有可视化图表"""
    print(f"\n{'='*70}")
    print("Generating visualizations (Time-Varying μ Model)")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    plot_gof_qq_residuals(all_group_results, output_dir)
    plot_distance_metrics_heatmap(all_group_results, output_dir)
    plot_independence_acf_panel(all_group_results, output_dir)
    plot_gof_score_comparison(all_group_results, output_dir)
    plot_gof_summary_dashboard(all_group_results, output_dir)
    plot_excitation_matrix_heatmaps(all_group_results, output_dir)
    plot_branching_ratio_boxplot(all_group_results, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}/")


def main():
    """主函数：处理三个价格分组"""
    
    base_output = "results"
    
    price_groups = {
        "high": {
            "data_dir": "data/high_price_events",
            "output_dir": f"{base_output}/high_price_4d",
        },
        "mid": {
            "data_dir": "data/mid_price_events", 
            "output_dir": f"{base_output}/mid_price_4d",
        },
        "low": {
            "data_dir": "data/low_price_events",
            "output_dir": f"{base_output}/low_price_4d",
        },
    }
    
    all_group_results = {}
    
    print("\n" + "="*70)
    print("4D HAWKES MODEL FITTING - TIME-VARYING μ (I_OPEN30/I_MID/I_CLOSE30)")
    print("="*70)
    print("Model: λ_i(t) = μ_corrected_i * exp(γ·I(t)) + Σ_j Σ_k A_{ij} e^{-β(t-t_k^j)}")
    print("GOF v2: QQ + distance + independence + composite score")
    print("="*70 + "\n")
    
    for group_name, config in price_groups.items():
        results = process_price_group(
            group_name=group_name,
            data_dir=config["data_dir"],
            output_dir=config["output_dir"],
        )
        all_group_results[group_name] = results
    
    # 生成跨组对比报告
    print(f"\n{'='*70}")
    print("Generating cross-group comparison report")
    print(f"{'='*70}")
    
    comparison = {}
    for group_name, results in all_group_results.items():
        successful = [r for r in results if "error" not in r]
        if len(successful) > 0:
            br = [r["full"]["branching_ratio"] for r in successful]
            decays = [r["full"]["decay"] for r in successful]
            gof_counts = [r.get("gof", {}).get("summary", {}).get("gof_pass_count", 0) for r in successful]
            
            comparison[group_name] = {
                "n_stocks": len(results),
                "n_successful": len(successful),
                "branching_ratio_mean": float(np.mean(br)),
                "branching_ratio_std": float(np.std(br)),
                "decay_mean": float(np.mean(decays)),
                "stable_ratio": float(sum(1 for b in br if b < 1.0) / len(br)),
                "gof_pass_ratio": float(sum(1 for c in gof_counts if c == 4) / len(successful)),
                "gof_score_mean": float(np.mean([r["gof"]["summary"].get("gof_score_mean", 0) for r in successful])),
                "wasserstein_mean": float(np.mean([
                    np.mean([r["gof"].get(f"dim_{d}", {}).get("wasserstein_1", 0) for d in range(4)])
                    for r in successful
                ])),
                "qq_mae_mean": float(np.mean([
                    np.mean([r["gof"].get(f"dim_{d}", {}).get("qq_mae", 0) for d in range(4)])
                    for r in successful
                ])),
            }
    
    os.makedirs(base_output, exist_ok=True)
    comparison_file = f"{base_output}/price_groups_comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"Comparison report saved to {comparison_file}")
    
    # 打印汇总
    print("\n" + "="*70)
    print("SUMMARY (Time-Varying μ Model)")
    print("="*70)
    for group_name, st in comparison.items():
        print(f"\n{group_name.upper()} PRICE GROUP:")
        print(f"  Stocks: {st['n_successful']}/{st['n_stocks']} successful")
        print(f"  Branching ratio: {st['branching_ratio_mean']:.4f} ± {st['branching_ratio_std']:.4f}")
        print(f"  GOF score: {st['gof_score_mean']:.3f}, W1={st['wasserstein_mean']:.3f}, QQ_MAE={st['qq_mae_mean']:.3f}")
    
    generate_all_visualizations(all_group_results, base_output)


if __name__ == "__main__":
    os.environ.setdefault("BETA_STRATEGY", "grid")
    os.environ.setdefault("REQUIRE_STABLE", "1")
    os.environ.setdefault("DECAY_MIN", "0.3")
    os.environ.setdefault("DECAY_MAX", "10.0")
    
    main()
