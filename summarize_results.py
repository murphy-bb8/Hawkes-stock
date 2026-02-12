"""汇总实验结果"""
import json, numpy as np

with open('experiment_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

groups = ['high', 'mid', 'low']
models = ['A', 'B', 'C']

print('=' * 85)
print('                         汇 总 统 计')
print('=' * 85)
for gn in groups:
    print(f'\n  [{gn.upper()} 组]')
    hdr = f"  {'Model':<7} {'N':>3} {'BR_mean':>8} {'BR_std':>7} {'LL_mean':>12} {'AIC_mean':>12} {'GOF_mean':>9} {'GOF4/4':>7} {'Time':>6}"
    print(hdr)
    print('  ' + '-' * 80)
    for model in models:
        key = f'{gn}_{model}'
        recs = data.get(key, [])
        if not recs:
            continue
        n = len(recs)
        brs = [r['branching_ratio'] for r in recs]
        lls = [r['loglik'] for r in recs]
        aics = [r['aic'] for r in recs]
        gofs = [r['gof_score'] for r in recs]
        gof4 = sum(1 for r in recs if r['gof_pass'] == 4)
        ts = sum(r['time_s'] for r in recs)
        print(f"  {model:<7} {n:>3} {np.mean(brs):>8.4f} {np.std(brs):>7.4f} "
              f"{np.mean(lls):>12.0f} {np.mean(aics):>12.0f} "
              f"{np.mean(gofs):>9.3f} {gof4:>4}/{n:<2} {ts:>6.0f}s")

# 全样本汇总
print(f'\n\n{"=" * 85}')
print('                 全 样 本 汇 总 (45 只股票)')
print('=' * 85)
hdr2 = f"  {'Model':<7} {'BR_mean':>8} {'BR_std':>7} {'GOF_mean':>9} {'GOF4/4':>7}"
print(hdr2)
print('  ' + '-' * 40)
for model in models:
    all_recs = []
    for gn in groups:
        all_recs.extend(data.get(f'{gn}_{model}', []))
    n = len(all_recs)
    brs = [r['branching_ratio'] for r in all_recs]
    gofs = [r['gof_score'] for r in all_recs]
    gof4 = sum(1 for r in all_recs if r['gof_pass'] == 4)
    print(f"  {model:<7} {np.mean(brs):>8.4f} {np.std(brs):>7.4f} "
          f"{np.mean(gofs):>9.3f} {gof4:>4}/{n}")

# AIC 模型选择
print(f'\n\n{"=" * 85}')
print('              AIC 模型选择 (逐股票最优模型计数)')
print('=' * 85)
best_counts = {'A': 0, 'B': 0, 'C': 0}
for gn in groups:
    codes_a = set(r['code'] for r in data.get(f'{gn}_A', []))
    for code in codes_a:
        aics = {}
        for m in models:
            rec = next((r for r in data.get(f'{gn}_{m}', []) if r['code'] == code), None)
            if rec:
                aics[m] = rec['aic']
        if aics:
            best = min(aics, key=lambda k: aics[k])
            best_counts[best] += 1
for m in models:
    print(f"  Model {m}: {best_counts[m]} 只股票 AIC 最优")

# LL 提升: B vs A, C vs B
print(f'\n\n{"=" * 85}')
print('              LL 提升分析 (B vs A, C vs B)')
print('=' * 85)
for gn in groups:
    ll_improve_ba = []
    ll_improve_cb = []
    codes_a = {r['code']: r for r in data.get(f'{gn}_A', [])}
    codes_b = {r['code']: r for r in data.get(f'{gn}_B', [])}
    codes_c = {r['code']: r for r in data.get(f'{gn}_C', [])}
    for code in codes_a:
        if code in codes_b:
            ll_improve_ba.append(codes_b[code]['loglik'] - codes_a[code]['loglik'])
        if code in codes_b and code in codes_c:
            ll_improve_cb.append(codes_c[code]['loglik'] - codes_b[code]['loglik'])
    if ll_improve_ba:
        arr_ba = np.array(ll_improve_ba)
        arr_cb = np.array(ll_improve_cb)
        print(f"  [{gn.upper()}] B-A: mean={arr_ba.mean():>8.0f}, "
              f"median={np.median(arr_ba):>8.0f}, "
              f"all>0: {sum(x > 0 for x in arr_ba)}/{len(arr_ba)}")
        print(f"  [{gn.upper()}] C-B: mean={arr_cb.mean():>8.1f}, "
              f"median={np.median(arr_cb):>8.1f}, "
              f"all>0: {sum(x > 0 for x in arr_cb)}/{len(arr_cb)}")

# gamma_spread 汇总
print(f'\n\n{"=" * 85}')
print('           Model C gamma_spread 均值 (按组)')
print('=' * 85)
print(f"  {'Group':<7} {'g_buy_tox':>10} {'g_buy_not':>10} {'g_sell_tox':>10} {'g_sell_not':>10}")
print('  ' + '-' * 50)
for gn in groups:
    recs = data.get(f'{gn}_C', [])
    gs_all = [r['gamma_spread'] for r in recs if r.get('gamma_spread')]
    if gs_all:
        gs_arr = np.array(gs_all)
        print(f"  {gn:<7} {gs_arr[:,0].mean():>10.4f} {gs_arr[:,1].mean():>10.4f} "
              f"{gs_arr[:,2].mean():>10.4f} {gs_arr[:,3].mean():>10.4f}")
all_gs = []
for gn in groups:
    for r in data.get(f'{gn}_C', []):
        if r.get('gamma_spread'):
            all_gs.append(r['gamma_spread'])
gs_arr = np.array(all_gs)
print(f"  {'ALL':<7} {gs_arr[:,0].mean():>10.4f} {gs_arr[:,1].mean():>10.4f} "
      f"{gs_arr[:,2].mean():>10.4f} {gs_arr[:,3].mean():>10.4f}")

# BR 按组比较
print(f'\n\n{"=" * 85}')
print('           分支比 (Branching Ratio) 按组比较')
print('=' * 85)
print(f"  {'Group':<7} {'BR_A':>8} {'BR_B':>8} {'BR_C':>8}")
print('  ' + '-' * 35)
for gn in groups:
    br_vals = {}
    for m in models:
        recs = data.get(f'{gn}_{m}', [])
        br_vals[m] = np.mean([r['branching_ratio'] for r in recs]) if recs else 0
    print(f"  {gn:<7} {br_vals['A']:>8.4f} {br_vals['B']:>8.4f} {br_vals['C']:>8.4f}")
