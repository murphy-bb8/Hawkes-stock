"""
4D Toxic事件：每只股票将四类事件作为四个维度拟合。
"""
import os
import json
import numpy as np
from typing import Dict, List

from fit_toxic_events import load_toxic_events_data, extract_event_times
from hawkes_full_vs_simple_4d import run_comparison_4d


def _build_4d_events(stock_data: Dict) -> Dict:
    event_types = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    events_by_type = []
    for et in event_types:
        times = extract_event_times(stock_data, et)
        events_by_type.append(times.astype(float) if len(times) > 0 else np.asarray([], dtype=float))
    # normalize by global min
    all_times = np.concatenate([ev for ev in events_by_type if len(ev) > 0]) if any(len(ev) > 0 for ev in events_by_type) else np.asarray([], dtype=float)
    if len(all_times) == 0:
        return {"events": [], "T": 0.0, "counts": [0, 0, 0, 0]}
    t0 = float(np.min(all_times))
    events_norm = [ev - t0 for ev in events_by_type]
    T = float(np.max(all_times) - t0)
    counts = [int(len(ev)) for ev in events_by_type]
    return {"events": events_norm, "T": T, "counts": counts}


def fit_stock_4d(stock_code: str, stock_data: Dict, output_dir: str) -> Dict:
    built = _build_4d_events(stock_data)
    events_4d = built["events"]
    counts = built["counts"]
    if sum(counts) < 10:
        return {
            "stock_code": stock_code,
            "event_type": "4d",
            "error": "insufficient_events",
            "n_events": counts,
        }

    os.makedirs("temp", exist_ok=True)
    temp_file = f"temp/events_{stock_code}_4d.json"
    payload = []
    for dim, ev in enumerate(events_4d):
        for t in ev:
            payload.append({"t": float(t), "i": int(dim)})
    payload.sort(key=lambda x: x["t"])
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    os.environ["OUT_TAG"] = f"{stock_code}_4d"

    try:
        run_comparison_4d(temp_file)
        result_file = f"results/comparison_4d_{stock_code}_4d.json"
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            result["stock_code"] = stock_code
            result["event_type"] = "4d"
            result["n_events"] = counts
            result["T"] = float(built["T"])

            save_per_stock = os.getenv("SAVE_PER_STOCK", "1")
            if save_per_stock != "0":
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{stock_code}_4d.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            return result
        return {
            "stock_code": stock_code,
            "event_type": "4d",
            "error": "result_file_not_found",
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def generate_summary_report_4d(all_results: List[Dict], output_dir: str) -> None:
    report = {
        "summary": {
            "total_stocks": len(set(r.get("stock_code") for r in all_results)),
            "total_fits": len(all_results),
        },
        "branching_ratio": {},
        "mu_mean": [],
        "A_mean": [],
        "beta0_mean": [],
        "beta1_mean": [],
    }

    br = [r["full"]["branching_ratio"] for r in all_results if "full" in r]
    if len(br) > 0:
        report["branching_ratio"] = {
            "mean": float(np.mean(br)),
            "std": float(np.std(br)),
            "min": float(np.min(br)),
            "max": float(np.max(br)),
            "median": float(np.median(br)),
        }

    mus = np.array([r["full"]["mu"] for r in all_results if "full" in r], dtype=float)
    if mus.size > 0:
        report["mu_mean"] = np.mean(mus, axis=0).tolist()

    As = np.array([r["full"]["A"] for r in all_results if "full" in r], dtype=float)
    if As.size > 0:
        report["A_mean"] = np.mean(As, axis=0).tolist()

    b0s = np.array([r["full"]["beta0"] for r in all_results if "full" in r], dtype=float)
    if b0s.size > 0:
        report["beta0_mean"] = np.mean(b0s, axis=0).tolist()

    b1s = np.array([r["full"]["beta1"] for r in all_results if "full" in r], dtype=float)
    if b1s.size > 0:
        report["beta1_mean"] = np.mean(b1s, axis=0).tolist()

    report_file = os.path.join(output_dir, "summary_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    data_path = "data/events_toxic_all_201912.json"
    output_dir = "results/toxic_events_4d"

    data = load_toxic_events_data(data_path)
    all_results: List[Dict] = []

    for stock_code, stock_data in data.items():
        result = fit_stock_4d(stock_code, stock_data, output_dir)
        if "error" not in result:
            all_results.append(result)

    os.makedirs(output_dir, exist_ok=True)
    if len(all_results) > 0:
        summary_file = os.path.join(output_dir, "summary_all_results.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        generate_summary_report_4d(all_results, output_dir)


if __name__ == "__main__":
    os.environ.setdefault("BETA_STRATEGY", "grid")
    os.environ.setdefault("REG_L2_MU", "1e-4")
    os.environ.setdefault("REG_L2_A", "1e-3")
    os.environ.setdefault("REG_L2_B0", "1e-3")
    os.environ.setdefault("REG_STAB", "1e-1")
    os.environ.setdefault("STAB_MARGIN", "0.01")
    main()
