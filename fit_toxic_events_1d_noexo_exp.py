"""
1D Toxic事件（无外生项）实验 - 使用 tick 实现
- 每种事件类型单独建模为1D Hawkes
- 网格搜索 decay
- 输出到 results/exp_1d_noexo_grid/
"""
import os
import json
import numpy as np
from typing import Dict, List
from datetime import datetime

from fit_toxic_events import load_toxic_events_data, extract_event_times
from hawkes_1d_tick import run_comparison_1d_tick


OUTPUT_DIR = "results/exp_1d_noexo_grid"


def fit_stock_event_1d(stock_code: str, event_type: str, events: np.ndarray, T: float) -> Dict:
    """拟合单只股票的单个事件类型"""
    if len(events) < 10:
        return {
            "stock_code": stock_code,
            "event_type": event_type,
            "error": "insufficient_events",
            "n_events": len(events),
        }

    # 创建临时文件
    os.makedirs("temp", exist_ok=True)
    temp_file = f"temp/events_{stock_code}_{event_type}_1d.json"
    payload = [{"t": float(t)} for t in events]
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    os.environ["OUT_TAG"] = f"{stock_code}_{event_type}_1d"

    try:
        result = run_comparison_1d_tick(temp_file)
        result["stock_code"] = stock_code
        result["event_type"] = event_type
        result["n_events"] = len(events)
        result["T"] = float(T)

        # 保存单只股票结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"{stock_code}_{event_type}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    except Exception as e:
        return {
            "stock_code": stock_code,
            "event_type": event_type,
            "error": str(e),
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def generate_summary_report(all_results: List[Dict]) -> None:
    """生成汇总报告"""
    report = {
        "summary": {
            "total_stocks": len(set(r.get("stock_code") for r in all_results)),
            "total_event_types": len(set(r.get("event_type") for r in all_results)),
            "total_fits": len(all_results),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "BETA_STRATEGY": os.environ.get("BETA_STRATEGY", "grid"),
                "DECAY_MIN": os.environ.get("DECAY_MIN", "0.3"),
                "DECAY_MAX": os.environ.get("DECAY_MAX", "10.0"),
                "DECAY_GRID_DENSE": "[0.6, 2.0] with 12 points",
                "REQUIRE_STABLE": os.environ.get("REQUIRE_STABLE", "1"),
            },
        },
        "branching_ratio": {},
        "decay_stats": {},
        "mu_stats": {},
        "alpha_stats": {},
        "validation_ll": {},
        "gof_summary": {},
        "by_event_type": {},
    }

    # 分枝比统计
    br = [r["full"]["branching_ratio"] for r in all_results if "full" in r]
    if len(br) > 0:
        report["branching_ratio"] = {
            "mean": float(np.mean(br)),
            "std": float(np.std(br)),
            "min": float(np.min(br)),
            "max": float(np.max(br)),
            "median": float(np.median(br)),
            "constraint_ok_count": sum(1 for r in all_results if r.get("full", {}).get("constraint_ok", False)),
        }

    # decay 统计
    decays = [r["full"]["decay"] for r in all_results if "full" in r]
    if len(decays) > 0:
        report["decay_stats"] = {
            "mean": float(np.mean(decays)),
            "std": float(np.std(decays)),
            "min": float(np.min(decays)),
            "max": float(np.max(decays)),
            "median": float(np.median(decays)),
        }

    # mu 统计
    mus = [r["full"]["mu"] for r in all_results if "full" in r]
    if len(mus) > 0:
        report["mu_stats"] = {
            "mean": float(np.mean(mus)),
            "std": float(np.std(mus)),
            "min": float(np.min(mus)),
            "max": float(np.max(mus)),
        }

    # alpha 统计
    alphas = [r["full"]["alpha"] for r in all_results if "full" in r]
    if len(alphas) > 0:
        report["alpha_stats"] = {
            "mean": float(np.mean(alphas)),
            "std": float(np.std(alphas)),
            "min": float(np.min(alphas)),
            "max": float(np.max(alphas)),
        }

    # 验证集 LL
    ll_vals = [r["validation"]["ll_val"] for r in all_results 
               if "validation" in r and not np.isnan(r["validation"]["ll_val"])]
    if len(ll_vals) > 0:
        report["validation_ll"] = {
            "mean": float(np.mean(ll_vals)),
            "std": float(np.std(ll_vals)),
        }

    # GOF 汇总
    gof_pass = [r["gof"].get("gof_pass", False) for r in all_results if "gof" in r]
    if len(gof_pass) > 0:
        report["gof_summary"] = {
            "pass_count": sum(gof_pass),
            "total": len(gof_pass),
            "pass_rate": float(sum(gof_pass) / len(gof_pass)),
        }
        pvals = [r["gof"]["ks_pvalue"] for r in all_results 
                 if "gof" in r and "ks_pvalue" in r["gof"]]
        if len(pvals) > 0:
            report["gof_summary"]["mean_pvalue"] = float(np.mean(pvals))

    # 按事件类型分组统计
    event_types = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    for et in event_types:
        et_results = [r for r in all_results if r.get("event_type") == et and "full" in r]
        if len(et_results) > 0:
            report["by_event_type"][et] = {
                "count": len(et_results),
                "mu_mean": float(np.mean([r["full"]["mu"] for r in et_results])),
                "alpha_mean": float(np.mean([r["full"]["alpha"] for r in et_results])),
                "decay_mean": float(np.mean([r["full"]["decay"] for r in et_results])),
                "branching_ratio_mean": float(np.mean([r["full"]["branching_ratio"] for r in et_results])),
            }

    report_file = os.path.join(OUTPUT_DIR, "summary_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Summary report saved to {report_file}")


def main():
    data_path = "data/events_toxic_all_201912.json"

    # 先用这三只股票做实验，全部跑完后可改为 None
    TEST_STOCKS = ["600036", "600519", "601288"]

    print(f"Loading data from {data_path}...")
    data = load_toxic_events_data(data_path)
    print(f"Loaded {len(data)} stocks")

    if TEST_STOCKS:
        data = {k: v for k, v in data.items() if k in TEST_STOCKS}
        print(f"Testing with {len(data)} stocks: {list(data.keys())}")

    event_types = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    all_results: List[Dict] = []
    failed = []

    total_tasks = len(data) * len(event_types)
    task_idx = 0

    for stock_code, stock_data in data.items():
        for event_type in event_types:
            task_idx += 1
            print(f"\n[{task_idx}/{total_tasks}] Processing {stock_code} - {event_type}")
            
            # 提取事件
            events = extract_event_times(stock_data, event_type)
            if len(events) == 0:
                print(f"  No events found, skipping...")
                continue
            
            # 归一化
            events_sorted = np.sort(events)
            t0 = events_sorted[0]
            events_norm = events_sorted - t0
            T = events_norm[-1] if len(events_norm) > 0 else 1.0
            
            result = fit_stock_event_1d(stock_code, event_type, events_norm, T)
            
            if "error" not in result:
                all_results.append(result)
                br = result["full"]["branching_ratio"]
                decay = result["full"]["decay"]
                gof = result.get("gof", {}).get("gof_pass", False)
                print(f"  ✓ Success: decay={decay:.4f}, branching_ratio={br:.4f}, GOF={'pass' if gof else 'fail'}")
            else:
                failed.append((stock_code, event_type, result.get("error")))
                print(f"  ✗ Failed: {result.get('error')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(all_results) > 0:
        summary_file = os.path.join(OUTPUT_DIR, "summary_all_results.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(all_results)} results to {summary_file}")
        generate_summary_report(all_results)

    if len(failed) > 0:
        failed_file = os.path.join(OUTPUT_DIR, "failed.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"Failed ({len(failed)}) saved to {failed_file}")

    print(f"\n=== Experiment completed ===")
    print(f"Success: {len(all_results)}, Failed: {len(failed)}")


if __name__ == "__main__":
    # 网格搜索 decay（0.6-2.0 区间加密）
    os.environ.setdefault("BETA_STRATEGY", "grid")
    os.environ.setdefault("DECAY_MIN", "0.3")
    os.environ.setdefault("DECAY_MAX", "10.0")
    os.environ.setdefault("REQUIRE_STABLE", "1")
    
    main()
