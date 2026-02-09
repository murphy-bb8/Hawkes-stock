"""
4D Toxic事件（无外生项）实验 - 使用 tick 实现，速度快
- 放宽α约束（tick 自动优化，无显式约束）
- 网格搜索 decay
- 输出到 results/exp_4d_noexo_grid/
"""
import os
import json
import numpy as np
from typing import Dict, List
from datetime import datetime

from fit_toxic_events import load_toxic_events_data, extract_event_times
from hawkes_4d_tick import run_comparison_4d_tick


OUTPUT_DIR = "results/exp_4d_noexo_grid"


def _build_4d_events(stock_data: Dict) -> Dict:
    """构建4D事件数据"""
    event_types = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    events_by_type = []
    for et in event_types:
        times = extract_event_times(stock_data, et)
        events_by_type.append(times.astype(float) if len(times) > 0 else np.asarray([], dtype=float))
    
    all_times = np.concatenate([ev for ev in events_by_type if len(ev) > 0]) if any(len(ev) > 0 for ev in events_by_type) else np.asarray([], dtype=float)
    if len(all_times) == 0:
        return {"events": [], "T": 0.0, "counts": [0, 0, 0, 0]}
    
    t0 = float(np.min(all_times))
    events_norm = [ev - t0 for ev in events_by_type]
    T = float(np.max(all_times) - t0)
    counts = [int(len(ev)) for ev in events_by_type]
    return {"events": events_norm, "T": T, "counts": counts}


def fit_stock_4d(stock_code: str, stock_data: Dict) -> Dict:
    """拟合单只股票的4D模型"""
    built = _build_4d_events(stock_data)
    events_4d = built["events"]
    counts = built["counts"]
    
    if sum(counts) < 10:
        return {
            "stock_code": stock_code,
            "error": "insufficient_events",
            "n_events": counts,
        }

    # 创建临时文件
    os.makedirs("temp", exist_ok=True)
    temp_file = f"temp/events_{stock_code}_4d_tick.json"
    payload = []
    for dim, ev in enumerate(events_4d):
        for t in ev:
            payload.append({"t": float(t), "i": int(dim)})
    payload.sort(key=lambda x: x["t"])
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    os.environ["OUT_TAG"] = f"{stock_code}_4d"

    try:
        result = run_comparison_4d_tick(temp_file)
        result["stock_code"] = stock_code
        result["n_events"] = counts
        result["T"] = float(built["T"])

        # 保存单只股票结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"{stock_code}_4d.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    except Exception as e:
        return {
            "stock_code": stock_code,
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
        "validation_ll": {},
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

    # mu 统计（4维）
    mus = np.array([r["full"]["mu"] for r in all_results if "full" in r], dtype=float)
    if mus.size > 0:
        report["mu_stats"] = {
            "mean": np.mean(mus, axis=0).tolist(),
            "std": np.std(mus, axis=0).tolist(),
        }

    # 验证集对数似然
    ll_vals = [r["validation"]["ll_val"] for r in all_results if "validation" in r and not np.isnan(r["validation"]["ll_val"])]
    if len(ll_vals) > 0:
        report["validation_ll"] = {
            "mean": float(np.mean(ll_vals)),
            "std": float(np.std(ll_vals)),
        }

    # GOF 检验汇总
    gof_pass_counts = [r["gof"]["summary"]["gof_pass_count"] for r in all_results if "gof" in r]
    gof_all_pass = [r["gof"]["summary"]["all_pass"] for r in all_results if "gof" in r]
    if len(gof_pass_counts) > 0:
        report["gof_summary"] = {
            "mean_pass_count": float(np.mean(gof_pass_counts)),
            "all_pass_count": sum(gof_all_pass),
            "total_stocks": len(gof_all_pass),
        }
        # 每个维度的平均 KS p 值
        for d in range(4):
            dim_key = f"dim_{d}"
            pvals = [r["gof"][dim_key]["ks_pvalue"] for r in all_results 
                     if "gof" in r and dim_key in r["gof"] and "ks_pvalue" in r["gof"][dim_key]]
            if len(pvals) > 0:
                report["gof_summary"][f"dim_{d}_mean_pvalue"] = float(np.mean(pvals))

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

    # 筛选测试股票
    if TEST_STOCKS:
        data = {k: v for k, v in data.items() if k in TEST_STOCKS}
        print(f"Testing with {len(data)} stocks: {list(data.keys())}")

    all_results: List[Dict] = []
    failed_stocks = []

    for idx, (stock_code, stock_data) in enumerate(data.items(), 1):
        print(f"\n[{idx}/{len(data)}] Processing stock: {stock_code}")
        result = fit_stock_4d(stock_code, stock_data)
        if "error" not in result:
            all_results.append(result)
            br = result["full"]["branching_ratio"]
            decay = result["full"]["decay"]
            gof_pass = result.get("gof", {}).get("summary", {}).get("gof_pass_count", 0)
            print(f"  ✓ Success: decay={decay:.4f}, branching_ratio={br:.4f}, GOF={gof_pass}/4")
        else:
            failed_stocks.append((stock_code, result.get("error")))
            print(f"  ✗ Failed: {result.get('error')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存所有结果
    if len(all_results) > 0:
        summary_file = os.path.join(OUTPUT_DIR, "summary_all_results.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(all_results)} results to {summary_file}")
        generate_summary_report(all_results)

    # 保存失败列表
    if len(failed_stocks) > 0:
        failed_file = os.path.join(OUTPUT_DIR, "failed_stocks.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_stocks, f, ensure_ascii=False, indent=2)
        print(f"Failed stocks ({len(failed_stocks)}) saved to {failed_file}")

    print(f"\n=== Experiment completed ===")
    print(f"Success: {len(all_results)}, Failed: {len(failed_stocks)}")


if __name__ == "__main__":
    # 网格搜索 decay（0.6-2.0 区间加密）
    os.environ.setdefault("BETA_STRATEGY", "grid")
    os.environ.setdefault("DECAY_MIN", "0.3")
    os.environ.setdefault("DECAY_MAX", "10.0")
    os.environ.setdefault("REQUIRE_STABLE", "1")  # 优先选择稳定解
    
    main()
