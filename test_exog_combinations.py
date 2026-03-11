"""
test_exog_combinations.py
=========================
快速测试 7 种外生变量组合 (C1–C7) 对 Model C 的效果。
高/中/低 各取 1 只股票，与 Model A / B 作为基准对比。

外生变量:
  re_spread      — 价差变化率
  OBI            — 订单簿不平衡
  log_opp_depth  — 对手方深度 (对数)

组合编号:
  C1  re_spread               (原始 Model C)
  C2  OBI
  C3  log_opp_depth
  C4  re_spread + OBI
  C5  re_spread + log_opp_depth
  C6  OBI + log_opp_depth
  C7  re_spread + OBI + log_opp_depth

用法:
  python test_exog_combinations.py
  python test_exog_combinations.py --configs C1 C2 C3
  python test_exog_combinations.py --configs all
"""

import os, sys, json, time, argparse
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawkes_em_additive import (
    fit_hawkes_additive, flatten_events, gof_residuals,
    _precompute_R, em_hawkes_recursive,
    TRADING_SECONDS_PER_DAY, intraday_to_trading_time,
    get_period_tt, N_PERIODS, PERIOD_SECS_PER_DAY,
    PERIOD_OPEN, PERIOD_MID, PERIOD_CLOSE, PERIOD_NORMAL,
    _USE_CYTHON,
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
BETA_GRID = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0])
DIM_NAMES = ["BT", "BN", "ST", "SN"]

EXOG_CONFIGS = {
    "C1": ["re_spread"],
    "C2": ["OBI"],
    "C3": ["log_opp_depth"],
    "C4": ["re_spread", "OBI"],
    "C5": ["re_spread", "log_opp_depth"],
    "C6": ["OBI", "log_opp_depth"],
    "C7": ["re_spread", "OBI", "log_opp_depth"],
}


# ==================== 数据加载 (支持多外生变量) ====================

def load_and_build_multi(path: str, exog_vars: List[str]):
    """
    加载 JSON, 构建 4D 事件 / 日内时间 / 多外生变量列表。
    返回 dict: events, intraday, exog={var: [ndarray x4]}, T, n_days, code
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    et_keys = ["buy_toxic", "buy_not_toxic", "sell_toxic", "sell_not_toxic"]
    events_list, intra_list = [], []
    exog_dict = {v: [] for v in exog_vars}

    first_et = raw.get("events", {}).get(et_keys[0], {})
    dates_set = set()
    if isinstance(first_et, dict) and "days" in first_et:
        dates_set = set(d.get("date", "") for d in first_et["days"] if isinstance(d, dict))
    n_days = max(len(dates_set), 1)

    for et in et_keys:
        evts = raw.get("events", {}).get(et, {})
        t_off, t_intra = [], []
        var_vals = {v: [] for v in exog_vars}

        if isinstance(evts, dict) and "days" in evts:
            days = evts["days"]
            dates = sorted(set(d.get("date", "") for d in days if isinstance(d, dict)))
            d_idx = {d: i for i, d in enumerate(dates)}
            for dd in days:
                if not isinstance(dd, dict) or "t" not in dd:
                    continue
                offset = d_idx.get(dd.get("date", ""), 0) * TRADING_SECONDS_PER_DAY
                ts = dd["t"]
                raw_vars = {}
                for v in exog_vars:
                    raw_vars[v] = dd.get(v, [None] * len(ts))
                for idx_t, ti in enumerate(ts):
                    if not isinstance(ti, (int, float)):
                        continue
                    t_id = float(ti)
                    t_off.append(offset + intraday_to_trading_time(t_id))
                    t_intra.append(t_id)
                    for v in exog_vars:
                        rv = raw_vars[v]
                        val = rv[idx_t] if idx_t < len(rv) else None
                        var_vals[v].append(float(val) if isinstance(val, (int, float)) else 0.0)

        events_list.append(np.asarray(t_off, dtype=float))
        intra_list.append(np.asarray(t_intra, dtype=float))
        for v in exog_vars:
            exog_dict[v].append(np.asarray(var_vals[v], dtype=float))

    all_t = np.concatenate([e for e in events_list if len(e) > 0])
    if len(all_t) == 0:
        return None
    t0 = float(np.min(all_t))
    events_list = [e - t0 if len(e) > 0 else e for e in events_list]
    T = float(np.max(all_t) - t0)

    for i in range(4):
        if len(events_list[i]) > 0:
            idx = np.argsort(events_list[i])
            events_list[i] = events_list[i][idx]
            intra_list[i] = intra_list[i][idx]
            for v in exog_vars:
                exog_dict[v][i] = exog_dict[v][i][idx]

    return {
        "events": events_list, "intraday": intra_list,
        "exog": exog_dict,
        "T": T, "n_days": n_days, "code": raw.get("code", ""),
    }


# ==================== 多外生变量 EM ====================

def _normalize_exog(arr: np.ndarray):
    """z-score + 非负平移 + max=1 归一化, 返回 (shifted, scale, shift)"""
    m = float(np.mean(arr))
    s = float(np.std(arr))
    scale = s if s > 1e-10 else 1.0
    z = (arr - m) / scale
    shift = float(np.min(z))
    shifted = z - shift
    mx = float(np.max(shifted))
    if mx > 1e-10:
        shifted = shifted / mx
    return shifted, scale, shift


def _compute_x_total(shifted: np.ndarray, times: np.ndarray, T: float):
    """近似 ∫₀ᵀ x⁺(t) dt"""
    N = len(times)
    if N < 2:
        return 1.0
    dt = np.diff(times)
    xt = float(np.sum(shifted[:-1] * dt) + shifted[-1] * max(T - times[-1], 0))
    return max(xt, 1.0)


def em_multi_exog(
    times, types, dim, omega, mu_init, alpha_init,
    T, n_days, periods,
    exog_shifted_dict: Dict[str, np.ndarray],
    x_total_dict: Dict[str, float],
    maxiter=100, epsilon=1e-4, verbose=False, R_all=None,
):
    """
    支持多外生变量的 EM。
    λ_i(t) = μ_{i,period(t)} + Σ_v γ_{v,i}·x_v⁺(t) + excitation
    """
    N = len(times)
    if N == 0:
        return {"mu": mu_init, "alpha": alpha_init, "loglik": -np.inf, "n_iter": 0}

    var_names = list(exog_shifted_dict.keys())
    n_vars = len(var_names)

    T_per = PERIOD_SECS_PER_DAY * n_days
    N_type = np.array([np.sum(types == d) for d in range(dim)])

    if R_all is None:
        R_all = _precompute_R(times, types, dim, omega)

    comp = np.zeros(dim)
    for d in range(dim):
        mask_d = types == d
        comp[d] = np.sum(1.0 - np.exp(-omega * (T - times[mask_d])))

    type_masks = [(types == d) for d in range(dim)]
    type_idx = types

    alpha = alpha_init.copy().astype(float)
    mu_p = mu_init.copy().astype(float)
    if mu_p.ndim == 1:
        mu_p = np.tile(mu_p.reshape(-1, 1), (1, N_PERIODS))

    init_base = N_type / T + 0.01
    gamma_dict = {v: init_base * 0.1 for v in var_names}

    period_masks = {}
    for i in range(dim):
        for p in range(N_PERIODS):
            period_masks[(i, p)] = type_masks[i] & (periods == p)

    old_ll = -1e30

    for it in range(maxiter):
        bases = mu_p[type_idx, periods]
        exc = np.sum(alpha[type_idx, :] * R_all, axis=1)

        sp_per_var = {}
        sp_total = np.zeros(N)
        for v in var_names:
            sp_v = gamma_dict[v][type_idx] * exog_shifted_dict[v]
            sp_per_var[v] = sp_v
            sp_total += sp_v

        lam = bases + exc + sp_total
        lam = np.maximum(lam, 1e-15)
        inv_lam = 1.0 / lam

        ll_sum = np.sum(np.log(lam))

        bg_w = bases * inv_lam
        new_mu_p = np.full((dim, N_PERIODS), 1e-10)
        for i in range(dim):
            for p in range(N_PERIODS):
                if T_per[p] > 0:
                    s = np.sum(bg_w[period_masks[(i, p)]])
                    new_mu_p[i, p] = max(s / T_per[p], 1e-10)
        mu_p = new_mu_p

        for v in var_names:
            sp_w_v = sp_per_var[v] * inv_lam
            xt = x_total_dict[v]
            if xt > 0:
                new_gv = np.zeros(dim)
                for i in range(dim):
                    new_gv[i] = max(np.sum(sp_w_v[type_masks[i]]) / xt, 0.0)
                gamma_dict[v] = new_gv

        new_alpha = np.zeros((dim, dim))
        for i in range(dim):
            m_i = type_masks[i]
            R_i = R_all[m_i, :]
            w_i = inv_lam[m_i]
            for j in range(dim):
                if N_type[j] > 0:
                    new_alpha[i, j] = max(
                        alpha[i, j] * np.dot(R_i[:, j], w_i) / N_type[j], 0.0)
        alpha = new_alpha

        int_base = np.sum(mu_p * T_per[np.newaxis, :])
        int_sp = sum(np.sum(gamma_dict[v]) * x_total_dict[v] for v in var_names)
        int_exc = np.sum(alpha * comp[np.newaxis, :])
        ll = ll_sum - int_base - int_sp - int_exc

        if verbose and it % 10 == 0:
            print(f"  EM iter {it}: LL={ll:.4f}")
        if it > 0 and abs(ll - old_ll) < epsilon:
            if verbose:
                print(f"  EM converged iter {it}: LL={ll:.4f}")
            break
        old_ll = ll

    mu_normal = mu_p[:, PERIOD_NORMAL].copy()
    res = {
        "alpha": alpha.copy(), "loglik": float(ll), "n_iter": it + 1,
        "mu": mu_normal,
        "mu_periods": mu_p.copy(),
        "gamma_open": mu_p[:, PERIOD_OPEN] - mu_normal,
        "gamma_mid": mu_p[:, PERIOD_MID] - mu_normal,
        "gamma_close": mu_p[:, PERIOD_CLOSE] - mu_normal,
        "gamma_exog": {v: gamma_dict[v].copy() for v in var_names},
    }
    return res


def gof_residuals_multi(
    times, types, dim, omega, alpha, T, n_days,
    mu_periods, periods,
    gamma_dict, exog_shifted_dict,
):
    """多外生变量版 GOF 残差"""
    from scipy.stats import kstest, wasserstein_distance

    N = len(times)
    var_names = list(gamma_dict.keys())

    residuals = {d: [] for d in range(dim)}
    R = np.zeros(dim)
    Lambda_run = np.zeros(dim)
    seen = np.zeros(dim, dtype=bool)
    last_t = 0.0

    for k in range(N):
        dt = times[k] - last_t
        i = types[k]
        if dt > 0:
            decay = np.exp(-omega * dt)
            for d in range(dim):
                b = mu_periods[d, periods[k]]
                base_int = b * dt
                exc_int = np.dot(alpha[d, :], R) * (1.0 - decay) / omega if omega > 0 else 0.0
                sp_int = 0.0
                for v in var_names:
                    sp_int += gamma_dict[v][d] * exog_shifted_dict[v][k] * dt
                Lambda_run[d] += base_int + exc_int + sp_int
            R *= decay
        if seen[i] and Lambda_run[i] > 0:
            residuals[i].append(Lambda_run[i])
        seen[i] = True
        Lambda_run[i] = 0.0
        R[i] += omega
        last_t = times[k]

    results = {}
    np.random.seed(0)
    for d in range(dim):
        arr = np.array(residuals[d])
        if len(arr) > 20:
            ks_stat, ks_pval = kstest(arr, "expon", args=(0, 1))
            m = float(np.mean(arr))
            n_sub = min(len(arr), 5000)
            w1 = float(wasserstein_distance(arr[:n_sub], np.random.exponential(1.0, n_sub)))
            score_mean = max(0.0, 1.0 - abs(m - 1.0))
            score_w1 = max(0.0, 1.0 - w1)
            score_lb = 1.0 if ks_pval > 0.05 else 0.0
            gof_score = 0.4 * score_mean + 0.4 * score_w1 + 0.2 * score_lb
            results[d] = {
                "n": len(arr), "mean": m, "mean_dev": abs(m - 1.0),
                "ks_stat": float(ks_stat), "ks_pval": float(ks_pval),
                "w1": w1, "gof_score": gof_score,
                "gof_pass": bool(ks_pval > 0.05),
            }
        else:
            results[d] = {"n": len(arr), "error": "insufficient"}

    n_pass = sum(1 for d in range(dim) if results.get(d, {}).get("gof_pass", False))
    mean_score = np.mean([results[d]["gof_score"] for d in range(dim)
                          if "gof_score" in results.get(d, {})] or [0.0])
    results["summary"] = {"pass_count": n_pass, "total": dim, "all_pass": n_pass == dim,
                          "mean_gof_score": float(mean_score)}
    return results


# ==================== 多外生变量拟合入口 ====================

def fit_multi_exog(
    events_list, T, beta_grid, n_days,
    intraday_list, exog_lists: Dict[str, List[np.ndarray]],
    maxiter=100, epsilon=1e-4, verbose=False,
    init_from=None,
):
    """
    Model C 多外生变量拟合: beta 网格搜索 + EM + LL + AIC/BIC + GOF
    """
    dim = len(events_list)
    var_names = list(exog_lists.keys())
    n_exog = len(var_names)

    times, types, intra_arr, _ = flatten_events(events_list, intraday_list, None)
    N = len(times)
    if N < 2:
        raise ValueError("events too few: %d" % N)

    N_type = np.array([np.sum(types == d) for d in range(dim)])
    periods = np.array([get_period_tt(t % TRADING_SECONDS_PER_DAY) for t in times], dtype=int)

    # flatten each exog var in the same order as times
    exog_flat = {}
    for v in var_names:
        _, _, _, exog_arr = flatten_events(events_list, None, exog_lists[v])
        exog_flat[v] = exog_arr

    exog_shifted_dict = {}
    x_total_dict = {}
    exog_meta = {}
    for v in var_names:
        shifted, scale, shift = _normalize_exog(exog_flat[v])
        exog_shifted_dict[v] = shifted
        x_total_dict[v] = _compute_x_total(shifted, times, T)
        exog_meta[v] = {"scale": scale, "shift": shift}

    base_rate = N_type / T + 0.01

    if init_from is not None and "omega" in init_from:
        prev_omega = init_from["omega"]
        if prev_omega not in beta_grid:
            beta_grid = np.unique(np.append(beta_grid, prev_omega))

    best_ll = -np.inf
    best_res = None
    best_omega = beta_grid[0]

    for omega in beta_grid:
        R_all = _precompute_R(times, types, dim, omega)

        if init_from is not None and "alpha" in init_from:
            alpha_init = np.array(init_from["alpha"], dtype=float)
            if "mu_periods" in init_from:
                mu_init = np.array(init_from["mu_periods"], dtype=float)
            else:
                mu_init = np.tile(np.array(init_from["mu"], dtype=float).reshape(-1, 1), (1, N_PERIODS))
        else:
            mu_init = np.tile(base_rate.reshape(-1, 1), (1, N_PERIODS))
            alpha_init = 0.05 * np.ones((dim, dim))

        res = em_multi_exog(
            times, types, dim, omega, mu_init, alpha_init,
            T, n_days, periods, exog_shifted_dict, x_total_dict,
            maxiter, epsilon, verbose=False, R_all=R_all)

        if res["loglik"] > best_ll:
            best_ll = res["loglik"]
            best_res = res
            best_omega = omega

    # 暖启动精炼
    R_all_best = _precompute_R(times, types, dim, best_omega)
    mu_warm = best_res["mu_periods"].copy()
    alpha_warm = best_res["alpha"].copy()
    best_res = em_multi_exog(
        times, types, dim, best_omega, mu_warm, alpha_warm,
        T, n_days, periods, exog_shifted_dict, x_total_dict,
        maxiter=maxiter * 2, epsilon=epsilon / 10, verbose=verbose,
        R_all=R_all_best)
    best_ll = best_res["loglik"]

    # LL 单调性兜底
    if init_from is not None and "loglik" in init_from:
        if best_ll < init_from["loglik"]:
            prev_omega = init_from["omega"]
            R_all_prev = _precompute_R(times, types, dim, prev_omega)
            if "mu_periods" in init_from:
                mu_prev = np.array(init_from["mu_periods"], dtype=float)
            else:
                mu_prev = np.tile(np.array(init_from["mu"], dtype=float).reshape(-1, 1), (1, N_PERIODS))
            alpha_prev = np.array(init_from["alpha"], dtype=float)
            res_retry = em_multi_exog(
                times, types, dim, prev_omega, mu_prev, alpha_prev,
                T, n_days, periods, exog_shifted_dict, x_total_dict,
                maxiter=maxiter * 3, epsilon=epsilon / 100, verbose=verbose,
                R_all=R_all_prev)
            if res_retry["loglik"] > best_ll:
                best_ll = res_retry["loglik"]
                best_res = res_retry
                best_omega = prev_omega

    br = float(np.max(np.abs(np.linalg.eigvals(best_res["alpha"]))))

    # k = period_mu + n_exog * dim (gamma) + alpha
    k = N_PERIODS * dim + n_exog * dim + dim * dim

    aic = -2 * best_ll + 2 * k
    bic = -2 * best_ll + k * np.log(N)

    gamma_exog = best_res.get("gamma_exog", {})

    gof = gof_residuals_multi(
        times, types, dim, best_omega, best_res["alpha"], T, n_days,
        best_res["mu_periods"], periods, gamma_exog, exog_shifted_dict)

    out = {
        "model": "C", "omega": float(best_omega),
        "mu": best_res["mu"].tolist(),
        "alpha": best_res["alpha"].tolist(),
        "loglik": float(best_ll), "aic": float(aic), "bic": float(bic),
        "branching_ratio": br, "n_params": k, "n_events": N,
        "n_iter": best_res["n_iter"],
        "gof_summary": gof["summary"],
        "mu_periods": best_res["mu_periods"].tolist(),
        "gamma_open": best_res["gamma_open"].tolist(),
        "gamma_mid": best_res["gamma_mid"].tolist(),
        "gamma_close": best_res["gamma_close"].tolist(),
    }
    for v in var_names:
        out["gamma_%s" % v] = gamma_exog[v].tolist()
        out["gamma_%s_raw" % v] = (gamma_exog[v] / exog_meta[v]["scale"]).tolist()
    out["exog_vars"] = var_names
    out["gof_details"] = {str(d): gof[d] for d in range(dim) if d in gof}
    return out


# ==================== 主测试流程 ====================

def pick_stocks():
    """每组选第 1 只股票"""
    picks = {}
    for g in ["high", "mid", "low"]:
        data_dir = os.path.join(DATA_DIR, "%s_price_events" % g)
        if not os.path.isdir(data_dir):
            print("[WARN] missing %s" % data_dir)
            continue
        sfs = sorted([f for f in os.listdir(data_dir)
                      if f.startswith("events_") and f.endswith(".json") and not f.startswith("all_")])
        if sfs:
            picks[g] = os.path.join(data_dir, sfs[0])
    return picks


def run_test(active_configs: List[str], verbose: bool = False):
    picks = pick_stocks()
    if not picks:
        print("ERROR: no data found in %s" % DATA_DIR)
        return

    all_needed_vars = set()
    for cfg in active_configs:
        all_needed_vars.update(EXOG_CONFIGS[cfg])
    all_needed_vars = sorted(all_needed_vars)

    print("=" * 90)
    print("  Exogenous Variable Combination Test")
    print("  Cython: %s" % ("ENABLED" if _USE_CYTHON else "DISABLED (pure Python)"))
    print("  Active configs: %s" % active_configs)
    print("  Exog vars needed: %s" % all_needed_vars)
    print("  Stocks: %s" % {g: os.path.basename(p) for g, p in picks.items()})
    print("=" * 90)

    all_results = []
    t_global = time.time()

    for g, fp in sorted(picks.items()):
        code = os.path.basename(fp).replace("events_", "").replace("_201912.json", "")
        print("\n" + "-" * 80)
        print("  [%s] %s  loading..." % (g.upper(), code))

        built = load_and_build_multi(fp, all_needed_vars)
        if built is None:
            print("  SKIP: empty data")
            continue

        counts = [len(e) for e in built["events"]]
        total = sum(counts)
        print("  events=%s  total=%d  days=%d  T=%.0f" % (counts, total, built["n_days"], built["T"]))

        # --- Model A ---
        t0 = time.time()
        res_A = fit_hawkes_additive(
            built["events"], built["T"], BETA_GRID, model="A", n_days=built["n_days"],
            maxiter=200, epsilon=1e-5, verbose=False)
        t_A = time.time() - t0
        print("  Model A : LL=%12.1f  AIC=%12.1f  BIC=%12.1f  BR=%.4f  GOF=%.3f  %5.1fs" % (
            res_A["loglik"], res_A["aic"], res_A["bic"],
            res_A["branching_ratio"], res_A["gof_summary"]["mean_gof_score"], t_A))

        # --- Model B ---
        t0 = time.time()
        res_B = fit_hawkes_additive(
            built["events"], built["T"], BETA_GRID, model="B", n_days=built["n_days"],
            intraday_list=built["intraday"],
            maxiter=200, epsilon=1e-5, verbose=False, init_from=res_A)
        t_B = time.time() - t0
        print("  Model B : LL=%12.1f  AIC=%12.1f  BIC=%12.1f  BR=%.4f  GOF=%.3f  %5.1fs" % (
            res_B["loglik"], res_B["aic"], res_B["bic"],
            res_B["branching_ratio"], res_B["gof_summary"]["mean_gof_score"], t_B))

        stock_row = {
            "group": g, "code": code, "n_events": total,
            "A": {"LL": res_A["loglik"], "AIC": res_A["aic"], "BIC": res_A["bic"],
                   "BR": res_A["branching_ratio"], "GOF": res_A["gof_summary"]["mean_gof_score"],
                   "k": res_A["n_params"], "time": round(t_A, 1)},
            "B": {"LL": res_B["loglik"], "AIC": res_B["aic"], "BIC": res_B["bic"],
                   "BR": res_B["branching_ratio"], "GOF": res_B["gof_summary"]["mean_gof_score"],
                   "k": res_B["n_params"], "time": round(t_B, 1)},
        }

        # --- Model C variants ---
        for cfg in active_configs:
            var_list = EXOG_CONFIGS[cfg]
            label = "%s(%s)" % (cfg, "+".join(var_list))
            t0 = time.time()

            if len(var_list) == 1:
                # 单外生变量: 直接用原始 fit_hawkes_additive
                v = var_list[0]
                res_C = fit_hawkes_additive(
                    built["events"], built["T"], BETA_GRID, model="C",
                    n_days=built["n_days"],
                    intraday_list=built["intraday"],
                    spread_list=built["exog"][v],
                    maxiter=200, epsilon=1e-5, verbose=False, init_from=res_B)
            else:
                # 多外生变量: 用扩展的 fit_multi_exog
                exog_subset = {v: built["exog"][v] for v in var_list}
                res_C = fit_multi_exog(
                    built["events"], built["T"], BETA_GRID,
                    n_days=built["n_days"],
                    intraday_list=built["intraday"],
                    exog_lists=exog_subset,
                    maxiter=200, epsilon=1e-5, verbose=False, init_from=res_B)

            t_C = time.time() - t0

            gamma_info = ""
            if len(var_list) == 1:
                gs = res_C.get("gamma_spread", [])
                gamma_info = "  gamma=%s" % [round(x, 5) for x in gs]
            else:
                for v in var_list:
                    gk = res_C.get("gamma_%s" % v, [])
                    gamma_info += "  g_%s=%s" % (v[:6], [round(x, 5) for x in gk])

            print("  %-26s: LL=%12.1f  AIC=%12.1f  BIC=%12.1f  BR=%.4f  GOF=%.3f  k=%2d  %5.1fs%s" % (
                label,
                res_C["loglik"], res_C["aic"], res_C["bic"],
                res_C["branching_ratio"], res_C["gof_summary"]["mean_gof_score"],
                res_C["n_params"], t_C, gamma_info))

            ll_mono_b = res_C["loglik"] >= res_B["loglik"]
            if not ll_mono_b:
                print("    *** WARNING: LL(C) < LL(B), diff=%.2f ***" % (res_C["loglik"] - res_B["loglik"]))

            stock_row[cfg] = {
                "LL": res_C["loglik"], "AIC": res_C["aic"], "BIC": res_C["bic"],
                "BR": res_C["branching_ratio"], "GOF": res_C["gof_summary"]["mean_gof_score"],
                "k": res_C["n_params"], "time": round(t_C, 1),
                "vars": var_list,
                "LL_mono_vs_B": ll_mono_b,
            }

        all_results.append(stock_row)

    elapsed = time.time() - t_global

    # ==================== 汇总表 ====================
    print("\n" + "=" * 130)
    print("  SUMMARY TABLE")
    print("=" * 130)

    header_models = ["A", "B"] + active_configs
    header = "%-5s %-6s %7s" % ("Group", "Code", "Events")
    for m in header_models:
        header += " | %12s %12s %6s" % ("%s_LL" % m, "%s_AIC" % m, "%s_GOF" % m)
    print(header)
    print("-" * len(header))

    for row in all_results:
        line = "%-5s %-6s %7d" % (row["group"].upper(), row["code"], row["n_events"])
        for m in header_models:
            d = row.get(m, {})
            line += " | %12.1f %12.1f %6.3f" % (d.get("LL", 0), d.get("AIC", 0), d.get("GOF", 0))
        print(line)

    # AIC/BIC 排名
    print("\n  AIC / BIC Ranking (lower is better):")
    print("  " + "-" * 80)
    for row in all_results:
        models_with_aic = [(m, row[m]["AIC"]) for m in header_models if m in row and "AIC" in row[m]]
        models_with_bic = [(m, row[m]["BIC"]) for m in header_models if m in row and "BIC" in row[m]]
        aic_rank = sorted(models_with_aic, key=lambda x: x[1])
        bic_rank = sorted(models_with_bic, key=lambda x: x[1])
        aic_best = aic_rank[0][0]
        bic_best = bic_rank[0][0]

        aic_detail = "  ".join(["%s:%.1f" % (m, v) for m, v in aic_rank[:4]])
        bic_detail = "  ".join(["%s:%.1f" % (m, v) for m, v in bic_rank[:4]])

        print("  [%s %s] AIC best=%-4s  BIC best=%-4s" % (
            row["group"].upper(), row["code"], aic_best, bic_best))
        print("           AIC: %s" % aic_detail)
        print("           BIC: %s" % bic_detail)

    # LL 提升对比
    print("\n  LL Improvement over Model B:")
    print("  " + "-" * 80)
    for row in all_results:
        ll_b = row["B"]["LL"]
        improvements = []
        for cfg in active_configs:
            if cfg in row:
                delta = row[cfg]["LL"] - ll_b
                improvements.append((cfg, delta))
        improvements.sort(key=lambda x: -x[1])
        parts = ["  %s: %+.1f" % (c, d) for c, d in improvements]
        print("  [%s %s] %s" % (row["group"].upper(), row["code"], "  |".join(parts)))

    print("\n  Total elapsed: %.1fs" % elapsed)
    print("=" * 130)

    # 保存结果
    out_path = os.path.join(BASE, "results_additive", "exog_combination_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print("\nResults saved to %s" % out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test exogenous variable combinations for Model C")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Configs to test: C1 C2 C3 C4 C5 C6 C7 or 'all'. Default=all.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.configs is None or "all" in (args.configs or []):
        active = list(EXOG_CONFIGS.keys())
    else:
        active = [c for c in args.configs if c in EXOG_CONFIGS]
        if not active:
            print("ERROR: no valid configs. Choose from:", list(EXOG_CONFIGS.keys()))
            sys.exit(1)

    run_test(active, verbose=args.verbose)
