import json, numpy as np

with open('experiment_results.json', 'r', encoding='utf-8') as f:
    R = json.load(f)

groups = ['high', 'mid', 'low']
models = ['A', 'B', 'C']

# 1. BR
print('=== BR ===')
for g in groups:
    for m in models:
        recs = R.get(f'{g}_{m}', [])
        brs = [r['branching_ratio'] for r in recs]
        print(f'{g} {m}: {np.mean(brs):.3f} +/- {np.std(brs):.3f}')

# 2. GOF Score
print('\n=== GOF Score ===')
for g in groups:
    for m in models:
        recs = R.get(f'{g}_{m}', [])
        gofs = [r['gof_score'] for r in recs]
        print(f'{g} {m}: {np.mean(gofs):.3f}')

# 3. GOF pass
print('\n=== GOF pass ===')
for g in groups:
    for m in models:
        recs = R.get(f'{g}_{m}', [])
        n = len(recs)
        p3 = sum(1 for r in recs if r['gof_pass'] >= 3)
        p4 = sum(1 for r in recs if r['gof_pass'] == 4)
        print(f'{g} {m}: 4/4={p4}/{n}, 3+/4={p3}/{n}')

# 4. LL improvement
print('\n=== LL improvement ===')
for g in groups:
    codes = set()
    for m in models:
        codes.update(r['code'] for r in R.get(f'{g}_{m}', []))
    diffs_ba, diffs_cb = [], []
    for code in sorted(codes):
        ra = next((r for r in R.get(f'{g}_A', []) if r['code'] == code), None)
        rb = next((r for r in R.get(f'{g}_B', []) if r['code'] == code), None)
        rc = next((r for r in R.get(f'{g}_C', []) if r['code'] == code), None)
        if ra and rb and rc:
            diffs_ba.append(rb['loglik'] - ra['loglik'])
            diffs_cb.append(rc['loglik'] - rb['loglik'])
    print(f'{g} B-A: mean={np.mean(diffs_ba):.1f}, min={np.min(diffs_ba):.1f}, max={np.max(diffs_ba):.1f}')
    print(f'{g} C-B: mean={np.mean(diffs_cb):.1f}, min={np.min(diffs_cb):.1f}, max={np.max(diffs_cb):.1f}')

# 5. AIC/BIC best
print('\n=== AIC/BIC best ===')
aic_cnt = {'A': 0, 'B': 0, 'C': 0}
bic_cnt = {'A': 0, 'B': 0, 'C': 0}
for g in groups:
    codes = set()
    for m in models:
        codes.update(r['code'] for r in R.get(f'{g}_{m}', []))
    for code in sorted(codes):
        ra = next((r for r in R.get(f'{g}_A', []) if r['code'] == code), None)
        rb = next((r for r in R.get(f'{g}_B', []) if r['code'] == code), None)
        rc = next((r for r in R.get(f'{g}_C', []) if r['code'] == code), None)
        if ra and rb and rc:
            aics = {'A': ra['aic'], 'B': rb['aic'], 'C': rc['aic']}
            bics = {'A': ra['bic'], 'B': rb['bic'], 'C': rc['bic']}
            aic_cnt[min(aics, key=aics.get)] += 1
            bic_cnt[min(bics, key=bics.get)] += 1
print(f'AIC best: {aic_cnt}')
print(f'BIC best: {bic_cnt}')

# 6. gamma_spread (Model C)
print('\n=== Model C gamma_spread ===')
for g in groups:
    recs = R.get(f'{g}_C', [])
    gs = [r['gamma_spread'] for r in recs if r['gamma_spread'] is not None]
    if gs:
        gs = np.array(gs)
        print(f'{g} mean: {gs.mean(axis=0).round(4)}')
        print(f'{g} std:  {gs.std(axis=0).round(4)}')
        print(f'{g} exp:  {np.exp(gs.mean(axis=0)).round(4)}')

# 7. mu (Model B)
print('\n=== Model B mu ===')
for g in groups:
    recs = R.get(f'{g}_B', [])
    mus = np.array([r['mu'] for r in recs])
    print(f'{g}: {mus.mean(axis=0).round(3)}')

# 8. gamma_open from Model B (stored in rec)
print('\n=== Model B gamma_open ===')
for g in groups:
    recs = R.get(f'{g}_B', [])
    go = [r['gamma_open'] for r in recs if 'gamma_open' in r and r['gamma_open'] is not None]
    if go:
        go = np.array(go)
        print(f'{g}: {go.mean(axis=0).round(3)}')
    else:
        print(f'{g}: not available in results')

# 9. Timing
print('\n=== Timing ===')
for m in models:
    ts = []
    for g in groups:
        recs = R.get(f'{g}_{m}', [])
        ts.extend([r['time_s'] for r in recs])
    print(f'{m}: mean={np.mean(ts):.1f}s, total={np.sum(ts):.0f}s')
total_time = sum(r['time_s'] for recs in R.values() for r in recs)
print(f'Total: {total_time:.0f}s = {total_time/60:.1f}min')

# 10. n_params (k)
print('\n=== n_params ===')
print('Model A: dim=4 -> k = 4(mu) + 16(alpha) + 1(beta) = 21')
print('Model B: k = 21 + 12(gamma_open/mid/close) = 33')
print('Model C: k = 33 + 4(gamma_spread) = 37')
