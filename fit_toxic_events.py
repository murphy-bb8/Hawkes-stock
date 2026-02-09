"""
批量处理50只股票的四类事件，应用Full模型并测试稳定性
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from full_vs_simple_1d import run_comparison_1d, load_events_1d
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def load_toxic_events_data(data_path: str) -> Dict:
    """加载toxic事件数据文件"""
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} stocks")
    return data

def extract_event_times(stock_data: Dict, event_type: str) -> np.ndarray:
    """
    从股票数据中提取指定类型的事件时间
    
    重要：返回的是日内时间（秒，从午夜开始，34200-54000范围）
    不包含日期偏移，需要配合 extract_event_times_with_day_offset 使用多日数据
    """
    events = stock_data.get("events", {}).get(event_type, {})
    
    if isinstance(events, list):
        # 如果已经是时间数组
        return np.asarray(events, dtype=float)
    elif isinstance(events, dict):
        all_times = []
        
        # 检查是否有"days"键（新格式：{"days": [{"date": "...", "t": [...], "T": ...}, ...]}）
        if "days" in events and isinstance(events["days"], list):
            # 遍历days列表中的每个日期数据
            for day_data in events["days"]:
                if isinstance(day_data, dict) and "t" in day_data:
                    t_data = day_data["t"]
                    if isinstance(t_data, list):
                        # 't'是列表，提取所有数字
                        for item in t_data:
                            if isinstance(item, (int, float)):
                                all_times.append(float(item))
                    elif isinstance(t_data, (int, float)):
                        all_times.append(float(t_data))
        else:
            # 旧格式：字典的每个值是一个包含't'字段的字典
            for key, value in events.items():
                if isinstance(value, dict):
                    # 如果值是字典，查找't'字段
                    if "t" in value:
                        t_data = value["t"]
                        if isinstance(t_data, list):
                            # 't'是列表，提取所有数字
                            for item in t_data:
                                if isinstance(item, (int, float)):
                                    all_times.append(float(item))
                        elif isinstance(t_data, (int, float)):
                            all_times.append(float(t_data))
                elif isinstance(value, list):
                    # 如果值直接是列表，提取所有数字
                    for item in value:
                        if isinstance(item, (int, float)):
                            all_times.append(float(item))
                elif isinstance(value, (int, float)):
                    # 如果值直接是数字
                    all_times.append(float(value))
        
        return np.asarray(all_times, dtype=float)
    else:
        # 其他情况，返回空数组
        return np.asarray([], dtype=float)


def extract_event_times_with_day_offset(stock_data: Dict, event_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从股票数据中提取指定类型的事件时间，包含日期偏移
    
    返回两个数组：
    - times_with_offset: 带日期偏移的连续时间（用于tick拟合）
    - times_intraday: 日内时间（用于时段判断，34200-54000范围）
    
    日期偏移计算：
    - 每个交易日有14400秒的交易时间（9:30-11:30 + 13:00-15:00）
    - 第N天的事件加上 N * 14400 的偏移
    
    Parameters:
    -----------
    stock_data : Dict
        股票数据，包含 events.{event_type}.days 结构
    event_type : str
        事件类型
    
    Returns:
    --------
    times_with_offset : np.ndarray
        带日期偏移的连续时间
    times_intraday : np.ndarray
        原始日内时间（用于OPEN30/MID30/CLOSE30判断）
    """
    events = stock_data.get("events", {}).get(event_type, {})
    
    times_with_offset = []
    times_intraday = []
    
    # 每个交易日的交易时长：4小时 = 14400秒
    TRADING_SECONDS_PER_DAY = 14400
    
    if isinstance(events, list):
        # 如果已经是时间数组，无法区分日期，回退到旧逻辑
        arr = np.asarray(events, dtype=float)
        return arr, arr.copy()
    
    elif isinstance(events, dict):
        # 检查是否有"days"键（新格式）
        if "days" in events and isinstance(events["days"], list):
            # 按日期排序
            days_list = events["days"]
            # 提取所有日期并排序
            day_dates = []
            for day_data in days_list:
                if isinstance(day_data, dict) and "date" in day_data:
                    day_dates.append(day_data["date"])
            day_dates_sorted = sorted(set(day_dates))
            date_to_index = {d: i for i, d in enumerate(day_dates_sorted)}
            
            # 遍历每个日期的数据
            for day_data in days_list:
                if not isinstance(day_data, dict) or "t" not in day_data:
                    continue
                
                # 获取日期索引
                date_str = day_data.get("date", "")
                day_idx = date_to_index.get(date_str, 0)
                day_offset = day_idx * TRADING_SECONDS_PER_DAY
                
                t_data = day_data["t"]
                if isinstance(t_data, list):
                    for item in t_data:
                        if isinstance(item, (int, float)):
                            t_intraday = float(item)
                            # 计算日内交易时间偏移（将34200-54000映射到0-14400）
                            t_trading = intraday_to_trading_time(t_intraday)
                            times_with_offset.append(day_offset + t_trading)
                            times_intraday.append(t_intraday)
                elif isinstance(t_data, (int, float)):
                    t_intraday = float(t_data)
                    t_trading = intraday_to_trading_time(t_intraday)
                    times_with_offset.append(day_offset + t_trading)
                    times_intraday.append(t_intraday)
        else:
            # 旧格式：无日期信息，回退到简单提取
            for key, value in events.items():
                if isinstance(value, dict) and "t" in value:
                    t_data = value["t"]
                    if isinstance(t_data, list):
                        for item in t_data:
                            if isinstance(item, (int, float)):
                                times_with_offset.append(float(item))
                                times_intraday.append(float(item))
                    elif isinstance(t_data, (int, float)):
                        times_with_offset.append(float(t_data))
                        times_intraday.append(float(t_data))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, (int, float)):
                            times_with_offset.append(float(item))
                            times_intraday.append(float(item))
    
    return np.asarray(times_with_offset, dtype=float), np.asarray(times_intraday, dtype=float)


def intraday_to_trading_time(t_intraday: float) -> float:
    """
    将日内时间（秒，从午夜开始）转换为交易时间（秒，从当日交易开始）
    
    A股交易时间：
    - 上午：9:30-11:30 (34200-41400) -> 0-7200
    - 下午：13:00-15:00 (46800-54000) -> 7200-14400
    
    Parameters:
    -----------
    t_intraday : float
        日内时间（秒，从午夜开始，范围34200-54000）
    
    Returns:
    --------
    float : 交易时间（秒，从当日开盘开始，范围0-14400）
    """
    MARKET_OPEN_AM = 34200    # 9:30
    MARKET_CLOSE_AM = 41400   # 11:30
    MARKET_OPEN_PM = 46800    # 13:00
    MARKET_CLOSE_PM = 54000   # 15:00
    
    if t_intraday < MARKET_OPEN_AM:
        # 早于开盘，按开盘处理
        return 0.0
    elif t_intraday <= MARKET_CLOSE_AM:
        # 上午时段
        return t_intraday - MARKET_OPEN_AM
    elif t_intraday < MARKET_OPEN_PM:
        # 午休时段，按上午收盘处理
        return MARKET_CLOSE_AM - MARKET_OPEN_AM  # 7200
    elif t_intraday <= MARKET_CLOSE_PM:
        # 下午时段
        return (MARKET_CLOSE_AM - MARKET_OPEN_AM) + (t_intraday - MARKET_OPEN_PM)
    else:
        # 晚于收盘，按收盘处理
        return 14400.0

def fit_stock_event_type(stock_code: str, event_type: str, events: np.ndarray, 
                        T: float, output_dir: str = "results/toxic_events") -> Dict:
    """
    对单只股票的单个事件类型应用Full模型
    
    Parameters:
    -----------
    stock_code : str
        股票代码
    event_type : str
        事件类型（buy_toxic, buy_not_toxic, sell_toxic, sell_not_toxic）
    events : np.ndarray
        事件时间数组
    T : float
        观测时间范围
    output_dir : str
        输出目录
    """
    if len(events) < 10:
        return {
            "stock_code": stock_code,
            "event_type": event_type,
            "error": "insufficient_events",
            "n_events": len(events)
        }
    
    # 创建临时文件
    os.makedirs("temp", exist_ok=True)
    temp_file = f"temp/events_{stock_code}_{event_type}.json"
    events_json = [{"t": float(t)} for t in events]
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(events_json, f)
    
    # 设置输出标签
    os.environ["OUT_TAG"] = f"{stock_code}_{event_type}"
    
    try:
        # 调用现有的拟合函数
        run_comparison_1d(temp_file)
        
        # 读取结果
        result_file = f"results/comparison_1d_{stock_code}_{event_type}.json"
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            result["stock_code"] = stock_code
            result["event_type"] = event_type
            result["n_events"] = len(events)
            result["T"] = float(T)
            
            # 根据环境变量决定是否保存单只股票结果
            # SAVE_PER_STOCK = "1"（默认）保存每只股票-事件类型一个文件
            # SAVE_PER_STOCK = "0" 只在内存中用于汇总，不落地单独文件
            save_per_stock = os.getenv("SAVE_PER_STOCK", "1")
            if save_per_stock != "0":
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{stock_code}_{event_type}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            return result
        else:
            return {
                "stock_code": stock_code,
                "event_type": event_type,
                "error": "result_file_not_found"
            }
    except Exception as e:
        return {
            "stock_code": stock_code,
            "event_type": event_type,
            "error": str(e)
        }
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

def stability_test(params_history: List[Dict], window_size: int = 5) -> Dict:
    """
    测试模型稳定性：使用滑动窗口方法
    
    Parameters:
    -----------
    params_history : List[Dict]
        参数历史记录（按时间顺序）
    window_size : int
        滑动窗口大小
    """
    if len(params_history) < window_size:
        return {"error": "insufficient_data"}
    
    # 提取参数
    mus = [p["full"]["mu"] for p in params_history]
    alphas = [p["full"]["alpha"] for p in params_history]
    beta0s = [p["full"]["beta0"] for p in params_history]
    beta1s = [p["full"]["beta1"] for p in params_history]
    
    # 计算滑动窗口的均值和标准差
    stability_metrics = {}
    for param_name, values in [("mu", mus), ("alpha", alphas), 
                               ("beta0", beta0s), ("beta1", beta1s)]:
        window_means = []
        window_stds = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i+window_size]
            window_means.append(np.mean(window))
            window_stds.append(np.std(window))
        
        stability_metrics[f"{param_name}_mean_cv"] = np.mean(window_stds) / np.mean(window_means) if np.mean(window_means) != 0 else np.inf
        stability_metrics[f"{param_name}_max_change"] = np.max(np.abs(np.diff(window_means)))
    
    return stability_metrics

def cross_validation_stability(events: np.ndarray, T: float, n_folds: int = 5) -> Dict:
    """
    使用K折交叉验证测试模型稳定性
    
    Parameters:
    -----------
    events : np.ndarray
        事件时间数组
    T : float
        总时间范围
    n_folds : int
        折数
    """
    if len(events) < n_folds * 10:
        return {"error": "insufficient_events"}
    
    # 按时间顺序分割
    fold_size = len(events) // n_folds
    params_folds = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(events)
        fold_events = events[start_idx:end_idx]
        
        if len(fold_events) < 10:
            continue
        
        # 重新计算时间范围（相对于第一个事件）
        fold_events_shifted = fold_events - fold_events[0]
        T_fold = fold_events_shifted[-1] if len(fold_events_shifted) > 0 else 1.0
        
        # 拟合该折的数据
        temp_file = f"temp/cv_fold_{i}.json"
        events_json = [{"t": float(t)} for t in fold_events_shifted]
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(events_json, f)
        
        try:
            os.environ["OUT_TAG"] = f"cv_fold_{i}"
            run_comparison_1d(temp_file)
            
            result_file = f"results/comparison_1d_cv_fold_{i}.json"
            if os.path.exists(result_file):
                with open(result_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                params_folds.append(result["full"])
        except Exception as e:
            print(f"Error in fold {i}: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    if len(params_folds) < 2:
        return {"error": "insufficient_folds"}
    
    # 计算参数稳定性指标
    mus = [p["mu"] for p in params_folds]
    alphas = [p["alpha"] for p in params_folds]
    beta0s = [p["beta0"] for p in params_folds]
    beta1s = [p["beta1"] for p in params_folds]
    
    stability = {
        "mu_mean": float(np.mean(mus)),
        "mu_std": float(np.std(mus)),
        "mu_cv": float(np.std(mus) / np.mean(mus)) if np.mean(mus) != 0 else np.inf,
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "alpha_cv": float(np.std(alphas) / np.mean(alphas)) if np.mean(alphas) != 0 else np.inf,
        "beta0_mean": float(np.mean(beta0s)),
        "beta0_std": float(np.std(beta0s)),
        "beta0_cv": float(np.std(beta0s) / np.abs(np.mean(beta0s))) if np.mean(beta0s) != 0 else np.inf,
        "beta1_mean": float(np.mean(beta1s)),
        "beta1_std": float(np.std(beta1s)),
        "beta1_cv": float(np.std(beta1s) / np.mean(beta1s)) if np.mean(beta1s) != 0 else np.inf,
        "n_folds": len(params_folds)
    }
    
    return stability

def main(enable_basic_viz: bool = True, enable_advanced_viz: bool = False):
    """主函数：批量处理所有股票和事件类型"""
    # 配置参数
    data_path = "data/events_toxic_all_201912.json"
    output_dir = "results/toxic_events"
    stability_output_dir = "results/stability_analysis"
    
    # 加载数据
    data = load_toxic_events_data(data_path)
    
    # 事件类型列表
    event_types = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    
    # 存储所有结果
    all_results = []
    stability_results = []
    
    # 遍历每只股票
    for stock_code, stock_data in data.items():
        print(f"\n{'='*60}")
        print(f"Processing stock: {stock_code}")
        print(f"{'='*60}")
        
        # 遍历每种事件类型
        for event_type in event_types:
            print(f"\n  Processing event type: {event_type}")
            
            # 提取事件时间
            events = extract_event_times(stock_data, event_type)
            
            if len(events) == 0:
                print(f"    No events found, skipping...")
                continue
            
            # 计算时间范围
            events_sorted = np.sort(events)
            T = events_sorted[-1] - events_sorted[0] if len(events_sorted) > 1 else 1.0
            
            # 将事件时间归一化到从0开始
            events_normalized = events_sorted - events_sorted[0]
            T_normalized = events_normalized[-1] if len(events_normalized) > 0 else 1.0
            
            print(f"    Found {len(events)} events, T={T_normalized:.2f}")
            
            # 拟合模型
            result = fit_stock_event_type(
                stock_code, event_type, events_normalized, T_normalized, output_dir
            )
            
            if "error" not in result:
                all_results.append(result)
                print(f"    ✓ Success: mu={result['full']['mu']:.4f}, "
                      f"alpha={result['full']['alpha']:.4f}, "
                      f"beta0={result['full']['beta0']:.4f}, "
                      f"beta1={result['full']['beta1']:.4f}")
                
                # 进行稳定性测试（K折交叉验证）
                print(f"    Running stability test (K-fold CV)...")
                stability = cross_validation_stability(events_normalized, T_normalized, n_folds=5)
                
                if "error" not in stability:
                    stability["stock_code"] = stock_code
                    stability["event_type"] = event_type
                    stability_results.append(stability)
                    print(f"    ✓ Stability test completed")
                else:
                    print(f"    ⚠ Stability test failed: {stability.get('error')}")
            else:
                print(f"    ✗ Failed: {result.get('error')}")
    
    # 保存汇总结果
    print(f"\n{'='*60}")
    print("Saving summary results...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stability_output_dir, exist_ok=True)
    
    # 保存所有拟合结果
    if len(all_results) > 0:
        summary_file = os.path.join(output_dir, "summary_all_results.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(all_results)} results to {summary_file}")
    else:
        print(f"  Warning: No successful fits found, skipping summary file creation")
    
    # 保存稳定性分析结果
    if len(stability_results) > 0:
        stability_file = os.path.join(stability_output_dir, "stability_analysis.json")
        with open(stability_file, "w", encoding="utf-8") as f:
            json.dump(stability_results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(stability_results)} stability results to {stability_file}")
    else:
        print(f"  Warning: No stability results found, skipping stability file creation")
    
    # 生成统计报告和整体可视化（仅当有结果时）
    if len(all_results) > 0:
        generate_summary_report(all_results, stability_results, output_dir)
        if enable_basic_viz:
            visualize_overall_results(all_results, output_dir)
        if enable_advanced_viz:
            visualize_advanced_results(all_results, stability_results, output_dir)
    else:
        print(f"  Warning: No results to generate summary report")

def generate_summary_report(all_results: List[Dict], stability_results: List[Dict], 
                           output_dir: str):
    """生成汇总统计报告"""
    report = {
        "summary": {
            "total_stocks": len(set(r["stock_code"] for r in all_results)),
            "total_event_types": len(set(r["event_type"] for r in all_results)),
            "total_fits": len(all_results),
            "timestamp": datetime.now().isoformat()
        },
        "parameter_statistics": {},
        "stability_statistics": {}
    }
    
    # 参数统计
    for event_type in ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]:
        type_results = [r for r in all_results if r.get("event_type") == event_type]
        if len(type_results) == 0:
            continue
        
        params = {
            "mu": [r["full"]["mu"] for r in type_results],
            "alpha": [r["full"]["alpha"] for r in type_results],
            "beta0": [r["full"]["beta0"] for r in type_results],
            "beta1": [r["full"]["beta1"] for r in type_results],
            "branching_ratio": [r["full"]["branching_ratio"] for r in type_results]
        }
        
        report["parameter_statistics"][event_type] = {
            param: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }
            for param, values in params.items()
        }
    
    # 稳定性统计
    if len(stability_results) > 0:
        for event_type in ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]:
            type_stability = [s for s in stability_results if s.get("event_type") == event_type]
            if len(type_stability) == 0:
                continue
            
            report["stability_statistics"][event_type] = {
                "mu_cv_mean": float(np.mean([s["mu_cv"] for s in type_stability if s["mu_cv"] != np.inf])),
                "alpha_cv_mean": float(np.mean([s["alpha_cv"] for s in type_stability if s["alpha_cv"] != np.inf])),
                "beta0_cv_mean": float(np.mean([s["beta0_cv"] for s in type_stability if s["beta0_cv"] != np.inf])),
                "beta1_cv_mean": float(np.mean([s["beta1_cv"] for s in type_stability if s["beta1_cv"] != np.inf])),
                "n_samples": len(type_stability)
            }
    
    # 保存报告
    report_file = os.path.join(output_dir, "summary_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Saved summary report to {report_file}")

def visualize_overall_results(all_results: List[Dict], output_dir: str):
    """
    基于所有拟合结果做整体可视化：
    1）分枝比分布
    2）各事件类型的验证集对数似然提升（Full - Simple）
    3）mu / alpha 的散点图（按事件类型着色）
    """
    if len(all_results) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 展平成表格，便于统计和可视化
    records = []
    for r in all_results:
        try:
            full = r["full"]
            val = r["validation"]
            rec = {
                "stock_code": r.get("stock_code"),
                "event_type": r.get("event_type"),
                "mu": float(full["mu"]),
                "alpha": float(full["alpha"]),
                "beta0": float(full["beta0"]),
                "beta1": float(full["beta1"]),
                "branching_ratio": float(full["branching_ratio"]),
                "ll_full": float(val["ll_full"]),
                "ll_simple": float(val["ll_simple"]),
            }
            rec["ll_improvement"] = rec["ll_full"] - rec["ll_simple"]
            records.append(rec)
        except Exception:
            # 某些结果可能缺少字段，直接跳过
            continue

    if len(records) == 0:
        return

    df = pd.DataFrame.from_records(records)

    # 1. 分枝比分布
    plt.figure(figsize=(6, 4))
    df["branching_ratio"].hist(bins=30, edgecolor="black")
    plt.xlabel("Branching ratio (alpha / beta)")
    plt.ylabel("Count")
    plt.title("Distribution of branching ratio (Full model)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_branching_ratio.png"), dpi=150)
    plt.close()

    # 2. 各事件类型验证集对数似然提升
    if "event_type" in df.columns:
        plt.figure(figsize=(6, 4))
        grouped = df.groupby("event_type")["ll_improvement"].mean().sort_values()
        grouped.plot(kind="bar")
        plt.ylabel("Mean validation log-likelihood improvement\n(Full - Simple)")
        plt.title("Validation performance improvement by event type")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "validation_improvement_by_type.png"), dpi=150)
        plt.close()

    # 3. mu vs alpha 散点图（按事件类型着色）
    plt.figure(figsize=(6, 4))
    if "event_type" in df.columns:
        for et, sub in df.groupby("event_type"):
            plt.scatter(sub["mu"], sub["alpha"], s=15, label=et, alpha=0.7)
        plt.legend(markerscale=1.5, fontsize=8)
    else:
        plt.scatter(df["mu"], df["alpha"], s=15, alpha=0.7)
    plt.xlabel("mu")
    plt.ylabel("alpha")
    plt.title("mu vs alpha (Full model)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mu_alpha_scatter.png"), dpi=150)
    plt.close()

def visualize_advanced_results(all_results: List[Dict],
                               stability_results: List[Dict],
                               output_dir: str):
    """
    高级可视化：
    1）验证对数似然提升的热力图（股票 × 事件类型）
    2）beta0 vs beta1 散点图（按事件类型着色）
    3）若提供稳定性结果：各事件类型参数CV的条形图
    """
    if len(all_results) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    records = []
    for r in all_results:
        try:
            full = r["full"]
            val = r["validation"]
            rec = {
                "stock_code": str(r.get("stock_code")),
                "event_type": r.get("event_type"),
                "beta0": float(full["beta0"]),
                "beta1": float(full["beta1"]),
                "ll_full": float(val["ll_full"]),
                "ll_simple": float(val["ll_simple"]),
            }
            rec["ll_improvement"] = rec["ll_full"] - rec["ll_simple"]
            records.append(rec)
        except Exception:
            continue

    if len(records) == 0:
        return

    df = pd.DataFrame.from_records(records)

    # 1. 验证对数似然提升热力图（股票 × 事件类型）
    try:
        pivot = df.pivot_table(
            index="stock_code",
            columns="event_type",
            values="ll_improvement",
            aggfunc="mean",
        )
        plt.figure(figsize=(8, max(4, 0.2 * len(pivot))))
        im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Validation ll_full - ll_simple")
        plt.yticks(range(len(pivot.index)), pivot.index, fontsize=6)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.title("Validation log-likelihood improvement\n(stock × event_type)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ll_improvement_heatmap.png"), dpi=150)
        plt.close()
    except Exception:
        # 若透视表失败则跳过该图
        pass

    # 2. beta0 vs beta1 散点图（按事件类型着色）
    plt.figure(figsize=(6, 4))
    if "event_type" in df.columns:
        for et, sub in df.groupby("event_type"):
            plt.scatter(sub["beta0"], sub["beta1"], s=15, alpha=0.7, label=et)
        plt.legend(markerscale=1.5, fontsize=8)
    else:
        plt.scatter(df["beta0"], df["beta1"], s=15, alpha=0.7)
    plt.xlabel("beta0")
    plt.ylabel("beta1")
    plt.title("beta0 vs beta1 (Full model)")
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beta0_beta1_scatter.png"), dpi=150)
    plt.close()

    # 3. 稳定性结果（CV）条形图
    if stability_results:
        try:
            df_stab = pd.DataFrame.from_records(stability_results)
            # 只保留需要的列
            cols = [
                "event_type",
                "mu_cv",
                "alpha_cv",
                "beta0_cv",
                "beta1_cv",
            ]
            df_stab = df_stab[cols]
            grouped = df_stab.groupby("event_type").mean(numeric_only=True)

            plt.figure(figsize=(7, 4))
            x = range(len(grouped.index))
            width = 0.2
            plt.bar([i - 1.5 * width for i in x], grouped["mu_cv"], width, label="mu_cv")
            plt.bar([i - 0.5 * width for i in x], grouped["alpha_cv"], width, label="alpha_cv")
            plt.bar([i + 0.5 * width for i in x], grouped["beta0_cv"], width, label="beta0_cv")
            plt.bar([i + 1.5 * width for i in x], grouped["beta1_cv"], width, label="beta1_cv")
            plt.xticks(list(x), grouped.index, rotation=45, ha="right")
            plt.ylabel("Coefficient of variation (CV)")
            plt.title("Parameter stability (mean CV by event type)")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "stability_cv_by_type.png"), dpi=150)
            plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit 1D Hawkes models with sinusoidal exogenous term on toxic events data."
    )
    parser.add_argument(
        "--no-basic-viz",
        action="store_true",
        help="禁用基础整体可视化（分枝比分布、按事件类型的LL提升、mu-alpha散点图）。",
    )
    parser.add_argument(
        "--advanced-viz",
        action="store_true",
        help="启用高级整体可视化（LL提升热力图、beta0-beta1散点、CV条形图）。",
    )

    args = parser.parse_args()

    # 设置模型参数（可根据需要调整）
    os.environ.setdefault("BETA_STRATEGY", "grid")
    os.environ.setdefault("REG_L2_MU", "1e-4")
    os.environ.setdefault("REG_L2_A", "1e-3")
    os.environ.setdefault("REG_L2_B0", "1e-3")
    os.environ.setdefault("REG_STAB", "1e-1")
    os.environ.setdefault("STAB_MARGIN", "0.01")

    main(enable_basic_viz=not args.no_basic_viz, enable_advanced_viz=args.advanced_viz)