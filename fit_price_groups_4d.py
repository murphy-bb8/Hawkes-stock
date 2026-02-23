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
from scipy.stats import probplot

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
        
        print(f"    -> Success: decay={result['full']['decay']:.4f}, "
              f"branching_ratio={result['full']['branching_ratio']:.4f}")
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
    """生成分组汇总报告"""
    successful = [r for r in all_results if "error" not in r]
    
    report = {
        "group": group_name,
        "summary": {
            "total_stocks": len(all_results),
            "successful_fits": len(successful),
            "failed_fits": len(all_results) - len(successful),
        },
    }
    
    if len(successful) > 0:
        # 分枝比统计
        br = [r["full"]["branching_ratio"] for r in successful]
        report["branching_ratio"] = {
            "mean": float(np.mean(br)),
            "std": float(np.std(br)),
            "min": float(np.min(br)),
            "max": float(np.max(br)),
            "median": float(np.median(br)),
            "stable_count": int(sum(1 for b in br if b < 1.0)),
        }
        
        # decay统计
        decays = [r["full"]["decay"] for r in successful]
        report["decay"] = {
            "mean": float(np.mean(decays)),
            "std": float(np.std(decays)),
            "min": float(np.min(decays)),
            "max": float(np.max(decays)),
        }
        
        # mu统计（4维）
        mus = np.array([r["full"]["mu"] for r in successful], dtype=float)
        report["mu_mean"] = np.mean(mus, axis=0).tolist()
        report["mu_std"] = np.std(mus, axis=0).tolist()
        
        # A矩阵统计
        As = np.array([r["full"]["A"] for r in successful], dtype=float)
        report["A_mean"] = np.mean(As, axis=0).tolist()
        
        # AIC统计
        aics = [r["full"]["aic"] for r in successful]
        report["aic"] = {
            "mean": float(np.mean(aics)),
            "std": float(np.std(aics)),
        }
        
        # GOF检验统计
        gof_pass_counts = []
        gamma_opens = []
        gamma_mids = []
        gamma_closes = []
        for r in successful:
            if "gof" in r and "summary" in r["gof"]:
                gof_pass_counts.append(r["gof"]["summary"]["gof_pass_count"])
                # 收集gamma参数
                if "gamma" in r["gof"]:
                    gamma_opens.append(r["gof"]["gamma"]["gamma_open"])
                    if "gamma_mid" in r["gof"]["gamma"]:
                        gamma_mids.append(r["gof"]["gamma"]["gamma_mid"])
                    gamma_closes.append(r["gof"]["gamma"]["gamma_close"])
        if gof_pass_counts:
            report["gof"] = {
                "mean_pass_count": float(np.mean(gof_pass_counts)),
                "all_pass_count": int(sum(1 for c in gof_pass_counts if c == 4)),
            }
            # 添加gamma参数统计（时变基准强度）
            if gamma_opens:
                gamma_opens_arr = np.array(gamma_opens)
                gamma_closes_arr = np.array(gamma_closes)
                report["gof"]["gamma_open_mean"] = np.mean(gamma_opens_arr, axis=0).tolist()
                report["gof"]["gamma_open_std"] = np.std(gamma_opens_arr, axis=0).tolist()
                report["gof"]["gamma_close_mean"] = np.mean(gamma_closes_arr, axis=0).tolist()
                report["gof"]["gamma_close_std"] = np.std(gamma_closes_arr, axis=0).tolist()
            if gamma_mids:
                gamma_mids_arr = np.array(gamma_mids)
                report["gof"]["gamma_mid_mean"] = np.mean(gamma_mids_arr, axis=0).tolist()
                report["gof"]["gamma_mid_std"] = np.std(gamma_mids_arr, axis=0).tolist()
    
    # 保存报告
    report_file = os.path.join(output_dir, "summary_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Summary report saved to {report_file}")


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
            
            # 收集该维度所有股票的残差均值（用于Q-Q图）
            residual_means = []
            for r in successful:
                if dim_key in r["gof"] and "mean" in r["gof"][dim_key]:
                    residual_means.append(r["gof"][dim_key]["mean"])
            
            if len(residual_means) >= 3:
                # 对残差均值做Q-Q图（理论分布为Exp(1)，均值应为1）
                residual_means = np.array(residual_means)
                
                # 使用正态Q-Q图展示（残差均值应接近1）
                probplot(residual_means, dist="norm", plot=ax)
                ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Expected mean=1')
                
                # 计算与理论值1的偏差
                mean_val = np.mean(residual_means)
                ax.set_title(f"{group_name.upper()} - {dim_names[col]}\nMean={mean_val:.3f}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{group_name.upper()} - {dim_names[col]}", fontsize=10)
            
            if row == 2:
                ax.set_xlabel("Theoretical Quantiles")
            if col == 0:
                ax.set_ylabel(f"{group_name.upper()}\nSample Quantiles")
    
    plt.suptitle("GOF Residual Q-Q Plots by Price Group and Event Type", fontsize=14, fontweight='bold')
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
    
    vmin, vmax = 0, 1.0  # 统一色标范围
    im = None
    
    for idx, group_name in enumerate(group_names):
        ax = fig.add_subplot(gs[0, idx])
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r]
        
        if len(successful) > 0:
            # 计算A矩阵均值
            As = np.array([r["full"]["A"] for r in successful], dtype=float)
            A_mean = np.mean(As, axis=0)
            
            # 绘制热力图
            im = ax.imshow(A_mean, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='equal')
            
            # 添加数值标注
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
    
    plt.suptitle("Excitation Matrix A (Group Mean) - Cross-Excitation Structure", fontsize=14, fontweight='bold', y=1.02)
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
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加散点
        for idx, (data, pos) in enumerate(zip(data_to_plot, positions)):
            x = np.random.normal(pos, 0.08, size=len(data))
            ax.scatter(x, data, alpha=0.6, color=colors[idx], s=30, edgecolors='black', linewidth=0.5)
        
        # 添加稳定性阈值线
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Stability threshold (BR=1)')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels[:len(positions)], fontsize=11)
        ax.set_ylabel("Branching Ratio", fontsize=12)
        ax.set_xlabel("Price Group", fontsize=12)
        ax.set_title("Branching Ratio Distribution by Price Group", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
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
    包括：残差自相关图和Ljung-Box检验p值分布
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
        
        # 收集所有维度的KS p值
        ks_pvalues = {f"dim_{d}": [] for d in range(4)}
        for r in successful:
            for d in range(4):
                dim_key = f"dim_{d}"
                if dim_key in r["gof"] and "ks_pvalue" in r["gof"][dim_key]:
                    ks_pvalues[dim_key].append(r["gof"][dim_key]["ks_pvalue"])
        
        # 绘制每个维度的p值柱状图
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
    
    # 第二行：残差均值偏离度（独立性指标）
    for idx, group_name in enumerate(group_names):
        ax = axes[1, idx]
        results = all_group_results.get(group_name, [])
        successful = [r for r in results if "error" not in r and "gof" in r]
        
        # 收集残差均值（理论值为1）
        residual_data = {d: [] for d in range(4)}
        for r in successful:
            for d in range(4):
                dim_key = f"dim_{d}"
                if dim_key in r["gof"] and "mean" in r["gof"][dim_key]:
                    residual_data[d].append(r["gof"][dim_key]["mean"])
        
        # 绘制箱线图
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
    
    plt.suptitle("Independence Diagnostics: GOF Test Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "independence_diagnostics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved independence diagnostics to {output_dir}/independence_diagnostics.png")


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
    
    print(f"\nAll visualizations saved to {output_dir}/")


def main():
    """主函数：处理三个价格分组"""
    
    # 定义三个分组
    price_groups = {
        "high": {
            "data_dir": "data/high_price_events",
            "output_dir": "results/high_price_4d",
        },
        "mid": {
            "data_dir": "data/mid_price_events", 
            "output_dir": "results/mid_price_4d",
        },
        "low": {
            "data_dir": "data/low_price_events",
            "output_dir": "results/low_price_4d",
        },
    }
    
    all_group_results = {}
    
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
            comparison[group_name] = {
                "n_stocks": len(results),
                "n_successful": len(successful),
                "branching_ratio_mean": float(np.mean(br)),
                "branching_ratio_std": float(np.std(br)),
                "decay_mean": float(np.mean(decays)),
                "stable_ratio": float(sum(1 for b in br if b < 1.0) / len(br)),
            }
    
    # 保存对比报告
    os.makedirs("results", exist_ok=True)
    comparison_file = "results/price_groups_comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"Comparison report saved to {comparison_file}")
    
    # 打印汇总
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for group_name, stats in comparison.items():
        print(f"\n{group_name.upper()} PRICE GROUP:")
        print(f"  Stocks: {stats['n_successful']}/{stats['n_stocks']} successful")
        print(f"  Branching ratio: {stats['branching_ratio_mean']:.4f} ± {stats['branching_ratio_std']:.4f}")
        print(f"  Decay: {stats['decay_mean']:.4f}")
        print(f"  Stable ratio: {stats['stable_ratio']*100:.1f}%")
    
    # 生成可视化图表
    generate_all_visualizations(all_group_results, "results")


if __name__ == "__main__":
    # 设置默认参数
    os.environ.setdefault("BETA_STRATEGY", "grid")
    os.environ.setdefault("REQUIRE_STABLE", "1")
    os.environ.setdefault("DECAY_MIN", "0.3")
    os.environ.setdefault("DECAY_MAX", "10.0")
    
    main()
