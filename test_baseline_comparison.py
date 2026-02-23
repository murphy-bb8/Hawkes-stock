"""
基线链接方式对比测试：加性 vs log-link
严格控制变量，仅改变建模方式。

测试设计：
- 3只股票（从 high/mid/low 各选1只）
- 2种基线链接方式（additive, log-link）
- 3个模型（A, B, C）
- 统一 β 网格：[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
- 统一 n_days=22, maxiter=10, tol=1e-5

输出：baseline_comparison_results.json
"""

import json
import time
import numpy as np
from typing import List, Dict, Any

# 导入两种基线实现
from hawkes_em_additive import fit_4d_additive, loglikelihood_additive, gof_residuals_additive
from hawkes_em_loglink import fit_4d_loglink, loglikelihood_loglink_wrapper, gof_residuals_loglink_wrapper

# 导入数据加载工具
from run_4d_models import load_stock_data, build_4d_events
from hawkes_em import SpreadProcess, TRADING_SECONDS_PER_DAY

# 测试股票（每组选1只）
TEST_STOCKS = {
    "high": ["600036"],
    "mid": ["600000"],
    "low": ["600016"]
}

# 统一参数
BETA_GRID = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
N_DAYS = 22
MAXITER = 10
TOL = 1e-5
T = TRADING_SECONDS_PER_DAY * N_DAYS

def evaluate_gof(residuals: Dict[int, List[float]]) -> Dict[str, Any]:
    """计算 GOF 指标"""
    gof_scores = []
    gof_passes = []
    
    for d, res in residuals.items():
        if not res:
            continue
        res_arr = np.array(res)
        # KS 统计量（简化版）
        n = len(res_arr)
        ecdf = np.arange(1, n+1) / n
        theoretical = 1 - np.exp(-res_arr)
        ks_stat = np.max(np.abs(ecdf - theoretical))
        gof_scores.append(1 - ks_stat)  # 越接近1越好
        gof_passes.append(int(ks_stat < 0.1))  # KS<0.1 算通过
    
    return {
        "gof_score": np.mean(gof_scores) if gof_scores else 0.0,
        "gof_pass": np.sum(gof_passes) if gof_passes else 0,
        "gof_total": len(gof_passes)
    }

def run_single_test(group: str, code: str, baseline_type: str, model: str) -> Dict[str, Any]:
    """运行单次测试"""
    print(f"[{baseline_type}] {group} {code} Model {model} ...")
    
    # 加载数据
    data_path = f"data/{group}_price_events/events_{code}_201912.json"
    stock_data = load_stock_data(data_path)
    built = build_4d_events(stock_data)
    events_4d = built["events"]
    events_orig = built["events_orig"]
    T_actual = built["T"]
    n_days_actual = built.get("n_days", N_DAYS)
    if len(events_4d) != 4:
        return {"error": f"{code} 事件维度不为4"}
    
    # 选择拟合函数
    if baseline_type == "additive":
        fit_func = fit_4d_additive
        ll_func = loglikelihood_additive
        gof_func = gof_residuals_additive
    else:  # log-link
        fit_func = fit_4d_loglink
        ll_func = loglikelihood_loglink_wrapper
        gof_func = gof_residuals_loglink_wrapper
    
    # 拟合
    start_time = time.time()
    try:
        result = fit_func(
            events_4d, T_actual, BETA_GRID,
            model=model,
            events_4d_original=events_orig if model in ("B", "C") else None,
            maxiter=MAXITER,
            verbose=False,
            n_days=n_days_actual,
        )
    except Exception as e:
        return {"error": f"拟合失败: {str(e)}"}
    
    elapsed = time.time() - start_time
    result["time_s"] = elapsed
    
    # GOF 评估
    try:
        if baseline_type == "additive":
            residuals = gof_func(
                events_4d, 4,
                np.array(result["mu"]), np.array(result["alpha"]), result["beta"],
                T_actual, n_days_actual, model,
                gamma_open=np.array(result.get("gamma_open", [0,0,0,0])),
                gamma_mid=np.array(result.get("gamma_mid", [0,0,0,0])),
                gamma_close=np.array(result.get("gamma_close", [0,0,0,0])),
                gamma_spread=np.array(result.get("gamma_spread", [0,0,0,0])) if model == "C" else None,
                spread_proc=SpreadProcess(events_orig) if model == "C" else None,
            )
        else:  # log-link
            residuals = gof_func(
                events_4d, 4,
                np.array(result["mu"]), np.array(result["alpha"]), result["beta"],
                T_actual, n_days_actual, model,
                gamma_open=np.array(result.get("gamma_open", [0,0,0,0])),
                gamma_mid=np.array(result.get("gamma_mid", [0,0,0,0])),
                gamma_close=np.array(result.get("gamma_close", [0,0,0,0])),
                gamma_spread=np.array(result.get("gamma_spread", [0,0,0,0])) if model == "C" else None,
                spread_proc=SpreadProcess(events_orig) if model == "C" else None,
            )
        
        gof_stats = evaluate_gof(residuals)
        result.update(gof_stats)
    except Exception as e:
        print(f"GOF 计算失败: {e}")
        result["gof_score"] = 0.0
        result["gof_pass"] = 0
        result["gof_total"] = 4
    
    # 添加元信息
    result["group"] = group
    result["code"] = code
    result["baseline_type"] = baseline_type
    result["model"] = model
    
    return result

def main():
    """主测试函数"""
    results = []
    
    for group, codes in TEST_STOCKS.items():
        for code in codes:
            for baseline_type in ["additive", "log-link"]:
                for model in ["A", "B", "C"]:
                    result = run_single_test(group, code, baseline_type, model)
                    results.append(result)
                    ll_val = result.get('loglik')
                    br_val = result.get('branching_ratio')
                    gof_val = result.get('gof_score')
                    ll_str = f"{ll_val:.1f}" if isinstance(ll_val, (int, float)) else "N/A"
                    br_str = f"{br_val:.3f}" if isinstance(br_val, (int, float)) else "N/A"
                    gof_str = f"{gof_val:.3f}" if isinstance(gof_val, (int, float)) else "N/A"
                    print(f" -> LL={ll_str}, BR={br_str}, GOF={gof_str}")
    
    # 保存结果
    output_file = "baseline_comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    print(f"总测试数: {len(results)}")
    
    # 简单统计
    success_results = [r for r in results if "error" not in r]
    print(f"成功拟合: {len(success_results)}/{len(results)}")
    
    # 按 baseline_type 分组统计
    for bt in ["additive", "log-link"]:
        bt_results = [r for r in success_results if r["baseline_type"] == bt]
        print(f"\n{bt} 基线统计:")
        for model in ["A", "B", "C"]:
            m_results = [r for r in bt_results if r["model"] == model]
            if m_results:
                avg_ll = np.mean([r["loglik"] for r in m_results])
                avg_br = np.mean([r["branching_ratio"] for r in m_results])
                avg_gof = np.mean([r["gof_score"] for r in m_results])
                print(f"  Model {model}: LL={avg_ll:.1f}, BR={avg_br:.3f}, GOF={avg_gof:.3f}")

if __name__ == "__main__":
    main()
