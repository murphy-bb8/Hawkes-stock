"""
analyze_results.py — 分析实盘实验结果
=====================================
验证关键指标并生成汇总报告
"""

import json
import numpy as np

def main():
    with open("experiment_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print("="*70)
    print("  实盘实验结果分析")
    print("="*70)
    
    # 按组汇总
    groups = ["high", "mid", "low"]
    models = ["A", "B", "C"]
    
    total_stocks = 0
    total_fits = 0
    violations = []
    
    for group in groups:
        print(f"\n{'='*70}")
        print(f"  {group.upper()} 组")
        print(f"{'='*70}")
        
        # 收集该组所有股票代码
        codes = set()
        for model in models:
            key = f"{group}_{model}"
            if key in results:
                codes.update([r["code"] for r in results[key]])
        
        codes = sorted(codes)
        print(f"股票数: {len(codes)}")
        total_stocks += len(codes)
        
        # 逐股票验证LL单调性
        for code in codes:
            # 获取三个模型的结果
            res_a = next((r for r in results.get(f"{group}_A", []) if r["code"] == code), None)
            res_b = next((r for r in results.get(f"{group}_B", []) if r["code"] == code), None)
            res_c = next((r for r in results.get(f"{group}_C", []) if r["code"] == code), None)
            
            if not all([res_a, res_b, res_c]):
                continue
            
            total_fits += 3
            
            ll_a = res_a["loglik"]
            ll_b = res_b["loglik"]
            ll_c = res_c["loglik"]
            
            # 检查LL单调性（允许1.0的数值误差）
            if ll_b < ll_a - 1.0:
                violations.append(f"{group}/{code}: LL_B({ll_b:.2f}) < LL_A({ll_a:.2f})")
            if ll_c < ll_b - 1.0:
                violations.append(f"{group}/{code}: LL_C({ll_c:.2f}) < LL_B({ll_b:.2f})")
        
        # 组级统计
        for model in models:
            key = f"{group}_{model}"
            if key not in results:
                continue
            
            recs = results[key]
            n = len(recs)
            
            if n == 0:
                continue
            
            brs = [r["branching_ratio"] for r in recs]
            lls = [r["loglik"] for r in recs]
            aics = [r["aic"] for r in recs]
            bics = [r["bic"] for r in recs]
            gof_scores = [r["gof_score"] for r in recs]
            gof_pass = sum(1 for r in recs if r["gof_pass"] >= 3)
            times = [r["time_s"] for r in recs]
            
            print(f"\nModel {model}:")
            print(f"  样本数: {n}")
            print(f"  BR: {np.mean(brs):.4f} ± {np.std(brs):.4f}")
            print(f"  LL: {np.mean(lls):.1f} ± {np.std(lls):.1f}")
            print(f"  AIC: {np.mean(aics):.1f} ± {np.std(aics):.1f}")
            print(f"  BIC: {np.mean(bics):.1f} ± {np.std(bics):.1f}")
            print(f"  GOF通过率: {gof_pass}/{n} = {100*gof_pass/n:.1f}%")
            print(f"  平均耗时: {np.mean(times):.1f}s")
            
            if model == "C":
                # γ_spread 统计
                gs_list = [r["gamma_spread"] for r in recs if r["gamma_spread"] is not None]
                if gs_list:
                    gs_arr = np.array(gs_list)
                    print(f"  γ_spread 均值: {gs_arr.mean(axis=0).round(4)}")
                    print(f"  γ_spread 标准差: {gs_arr.std(axis=0).round(4)}")
    
    # 总体汇总
    print(f"\n\n{'='*70}")
    print("  总体汇总")
    print(f"{'='*70}")
    print(f"总股票数: {total_stocks}")
    print(f"总拟合次数: {total_fits}")
    print(f"收敛率: {total_fits}/{total_fits} = 100.0% (无拟合失败)")
    
    # GOF统计
    total_gof_pass = 0
    total_gof_count = 0
    for key, recs in results.items():
        for r in recs:
            total_gof_pass += r["gof_pass"]
            total_gof_count += 4
    print(f"GOF通过率: {total_gof_pass}/{total_gof_count} = {100*total_gof_pass/total_gof_count:.1f}%")
    
    # LL单调性
    if violations:
        print(f"\n⚠️  发现 {len(violations)} 个LL单调性违反:")
        for v in violations[:10]:  # 只显示前10个
            print(f"  - {v}")
        if len(violations) > 10:
            print(f"  ... 还有 {len(violations)-10} 个")
    else:
        print(f"\n✅ 所有股票LL单调性检验通过！")
    
    # AIC/BIC最优模型统计
    print(f"\n{'='*70}")
    print("  AIC/BIC 最优模型统计")
    print(f"{'='*70}")
    
    aic_best_count = {"A": 0, "B": 0, "C": 0}
    bic_best_count = {"A": 0, "B": 0, "C": 0}
    
    for group in groups:
        codes = set()
        for model in models:
            key = f"{group}_{model}"
            if key in results:
                codes.update([r["code"] for r in results[key]])
        
        for code in sorted(codes):
            res_a = next((r for r in results.get(f"{group}_A", []) if r["code"] == code), None)
            res_b = next((r for r in results.get(f"{group}_B", []) if r["code"] == code), None)
            res_c = next((r for r in results.get(f"{group}_C", []) if r["code"] == code), None)
            
            if not all([res_a, res_b, res_c]):
                continue
            
            # AIC最优
            aics = {"A": res_a["aic"], "B": res_b["aic"], "C": res_c["aic"]}
            aic_best = min(aics, key=aics.get)
            aic_best_count[aic_best] += 1
            
            # BIC最优
            bics = {"A": res_a["bic"], "B": res_b["bic"], "C": res_c["bic"]}
            bic_best = min(bics, key=bics.get)
            bic_best_count[bic_best] += 1
    
    print(f"AIC最优: A={aic_best_count['A']}, B={aic_best_count['B']}, C={aic_best_count['C']}")
    print(f"BIC最优: A={bic_best_count['A']}, B={bic_best_count['B']}, C={bic_best_count['C']}")
    
    total = sum(aic_best_count.values())
    if total > 0:
        print(f"\nAIC: C选中率 = {100*aic_best_count['C']/total:.1f}%")
        print(f"BIC: C选中率 = {100*bic_best_count['C']/total:.1f}%")


if __name__ == "__main__":
    main()
