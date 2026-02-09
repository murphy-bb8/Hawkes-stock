import os
import json
import math
import numpy as np
from typing import Dict

from full_vs_simple_1d import run_comparison_1d


def simulate_1d(mu: float, alpha: float, beta: float, beta0: float, beta1: float, target_events: int, seed: int = 0, max_events: int = 2_000_000):
    """Simulate 1D Hawkes with sine exogenous term until target_events are reached.

    lambda(t) = mu + alpha * r(t) + beta0 * sin(beta1 * t),
    r(t) = sum_{s < t} exp(-beta (t - s)).
    """
    rng = np.random.RandomState(seed)
    t = 0.0
    r = 0.0
    out = []
    while len(out) < target_events and len(out) < max_events:
        lam_bar = mu + abs(beta0) + alpha * r
        if lam_bar <= 1e-12:
            break
        dt = rng.exponential(1.0 / lam_bar)
        t_next = t + dt
        decay = math.exp(-beta * dt)
        r *= decay
        lam = mu + alpha * r + beta0 * math.sin(beta1 * t_next)
        lam = max(lam, 0.0)
        if rng.rand() * lam_bar <= lam:
            out.append(t_next)
            r += 1.0
        t = t_next
    return np.asarray(out, dtype=float)


def main():
    # True parameters for simulation (stable: alpha/beta < 1)
    true_params: Dict[str, float] = {
        "mu": float(os.environ.get("SIM_MU", "0.6")),
        "alpha": float(os.environ.get("SIM_ALPHA", "0.45")),
        "beta": float(os.environ.get("SIM_BETA", "0.8")),
        "beta0": float(os.environ.get("SIM_BETA0", "0.6")),
        "beta1": float(os.environ.get("SIM_BETA1", "0.12")),  # period ~ 52.3
    }
    target_events = int(os.environ.get("SIM_N_EVENTS", "100000"))
    seed = int(os.environ.get("SIM_SEED", "0"))

    print("=== Simulation (1D Hawkes + sine) ===")
    print({**true_params, "target_events": target_events})
    # Direct built-in simulator to avoid API mismatch issues
    events = simulate_1d(
        true_params["mu"],
        true_params["alpha"],
        true_params["beta"],
        true_params["beta0"],
        true_params["beta1"],
        target_events=target_events,
        seed=seed,
    )
    data_path = os.environ.get("SIM_OUT", "events_sim_1d.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump([{"t": float(t), "i": 0} for t in events], f, ensure_ascii=False)
    print(f"Generated {len(events)} events -> {data_path}")

    # Save true parameters for downstream comparison
    os.makedirs("results", exist_ok=True)
    with open("results/true_params_1d.json", "w", encoding="utf-8") as f:
        json.dump(true_params, f, ensure_ascii=False, indent=2)

    # Fit using our 1D script and report
    print("\n=== Fitting on simulated data ===")
    run_comparison_1d(data_path, true_params=true_params)


if __name__ == "__main__":
    main()


