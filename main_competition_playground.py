from mable.examples import environment, fleets, companies
from mable.examples import environment, fleets, shipping
import group1
import group2
import group3
import group4
import group7

def build_specification():
    number_of_month = 12
    trades_per_auction = 5
    specifications_builder = environment.get_specification_builder(
        trades_per_occurrence=trades_per_auction,
        # fixed_trades=shipping.example_trades_1()
        num_auctions=number_of_month)
    my_fleet = fleets.mixed_fleet(num_suezmax=3, num_aframax=3, num_vlcc=3)
    specifications_builder.add_company(group1.Company1.Data(group1.Company1, my_fleet, group1.Company1.__name__))
    
    
    # group2_fleet = fleets.mixed_fleet(num_suezmax=2, num_aframax=2, num_vlcc=2)
    # specifications_builder.add_company(group2.Company2.Data(group2.Company2, group2_fleet, group2.Company2.__name__))
    # group3_fleet = fleets.mixed_fleet(num_suezmax=2, num_aframax=2, num_vlcc=2)
    # specifications_builder.add_company(group3.Company3.Data(group3.Company3, group3_fleet, group3.Company3.__name__))
    # group7_fleet = fleets.mixed_fleet(num_suezmax=2, num_aframax=2, num_vlcc=2)
    # specifications_builder.add_company(group7.Company7.Data(group7.Company7, group7_fleet, group7.Company7.__name__))
    # group4_fleet = fleets.mixed_fleet(num_suezmax=2, num_aframax=2, num_vlcc=2)
    # specifications_builder.add_company(group4.Company4.Data(group4.Company4, group4_fleet, group4.Company4.__name__))

    arch_enemy_fleet = fleets.mixed_fleet(num_suezmax=3, num_aframax=3, num_vlcc=3)
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(
            companies.MyArchEnemy, arch_enemy_fleet, "Arch Enemy Ltd.",
            profit_factor=5))
    the_scheduler_fleet = fleets.mixed_fleet(num_suezmax=3, num_aframax=3, num_vlcc=3)
    specifications_builder.add_company(
        companies.TheScheduler.Data(
            companies.TheScheduler, the_scheduler_fleet, "The Scheduler LP",
            profit_factor=5))
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=True,
        global_agent_timeout=60)
    return sim

n= 1
if __name__ == '__main__':
    build_specification()

import os, glob, json, random, statistics
from typing import Dict, Any, List, Tuple

try:
    import numpy as np
except Exception:
    np = None

# ----------------------------
# Helper: find newly created metrics file after a run
# ----------------------------
def _list_metrics_files() -> List[str]:
    return sorted(glob.glob("metrics_competition_*.json"))

def _pick_newest(new_files: List[str]) -> str:
    return max(new_files, key=lambda p: os.path.getmtime(p))

# ----------------------------
# Parse MABLE metrics_competition_*.json
# cost = fuel_cost, revenue = sum(payment), income = revenue - cost - penalty
# wins = number of contracts won
# ----------------------------
def parse_metrics(path: str) -> Dict[str, Dict[str, float]]:
    data = json.load(open(path, "r", encoding="utf-8"))

    # company_names: {"0":"Company4", ...}
    idx_to_name = data.get("company_names", {})
    company_metrics = data.get("company_metrics", {})
    penalties = data.get("global_metrics", {}).get("penalty", {})
    auction_outcomes = data.get("global_metrics", {}).get("auction_outcomes", [])

    out: Dict[str, Dict[str, float]] = {}

    for idx, name in idx_to_name.items():
        fuel_cost = float(company_metrics.get(idx, {}).get("fuel_cost", 0.0))
        penalty = float(penalties.get(idx, 0.0))

        wins = 0
        revenue = 0.0
        for auc in auction_outcomes:
            won_list = auc.get(idx, [])
            wins += len(won_list)
            for c in won_list:
                revenue += float(c.get("payment", 0.0))

        income = revenue - fuel_cost - penalty
        out[name] = {
            "wins": float(wins),
            "revenue": revenue,
            "cost": fuel_cost,
            "penalty": penalty,
            "income": income,
        }

    return out

# ============================
# Batch runner (append at EOF)
# ============================

import os, glob, json, random, statistics
from contextlib import redirect_stdout, redirect_stderr

try:
    import numpy as np
except Exception:
    np = None

BATCH_LOG_PATH = "batch_run_full.log"      # 全部 stdout/stderr 都写入这里
BATCH_SUMMARY_PATH = "batch_run_summary.txt"  # 每个 seed 一行 + 最终均值/方差

def _list_metrics_files():
    return sorted(glob.glob("metrics_competition_*.json"))

def _newest(paths):
    return max(paths, key=lambda p: os.path.getmtime(p))

def _parse_metrics(path: str):
    """
    Parse MABLE metrics_competition_*.json
    Returns: {company_name: {wins, revenue, cost, penalty, income}}
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    idx_to_name = data.get("company_names", {})           # {"0": "Company4", ...}
    company_metrics = data.get("company_metrics", {})     # {"0": {...}, ...}
    penalties = data.get("global_metrics", {}).get("penalty", {})
    auction_outcomes = data.get("global_metrics", {}).get("auction_outcomes", [])

    out = {}
    for idx, name in idx_to_name.items():
        fuel_cost = float(company_metrics.get(idx, {}).get("fuel_cost", 0.0))
        penalty = float(penalties.get(idx, 0.0))

        wins = 0
        revenue = 0.0
        for auc in auction_outcomes:
            won_list = auc.get(idx, [])
            wins += len(won_list)
            for c in won_list:
                revenue += float(c.get("payment", 0.0))

        income = revenue - fuel_cost - penalty
        out[name] = dict(
            wins=float(wins),
            revenue=float(revenue),
            cost=float(fuel_cost),
            penalty=float(penalty),
            income=float(income),
        )
    return out

def _run_one(seed: int):
    """
    Run one simulation; returns (metrics_path, parsed_metrics_dict)
    """
    # Make runs more reproducible
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    before = set(_list_metrics_files())

    sim = build_specification()  # IMPORTANT: must return sim (not run internally)
    if sim is None or not hasattr(sim, "run"):
        raise RuntimeError(
            "build_specification() did not return a simulation with .run(). "
            "Fix it to: sim = environment.generate_simulation(...); return sim"
        )

    sim.run()

    after = set(_list_metrics_files())
    new_files = sorted(list(after - before))
    if new_files:
        path = _newest(new_files)
    else:
        all_files = _list_metrics_files()
        if not all_files:
            raise RuntimeError("No metrics_competition_*.json produced after sim.run().")
        path = _newest(all_files)

    return path, _parse_metrics(path)

def _mean_std(xs):
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))

def run_many_to_file(n: int = 10, seed0: int = 0, target: str = "Company4"):
    # Clear summary file
    with open(BATCH_SUMMARY_PATH, "w", encoding="utf-8") as fsum:
        fsum.write(f"Batch run summary (n={n}, seed0={seed0}, target={target})\n")

    # Redirect *everything* (prints, logs, tracebacks) into BATCH_LOG_PATH
    with open(BATCH_LOG_PATH, "w", encoding="utf-8") as flog, open(BATCH_SUMMARY_PATH, "a", encoding="utf-8") as fsum:
        fsum.write(f"\nPer-seed results:\n")
        wins_list, income_list, pen_list, rev_list, cost_list = [], [], [], [], []

        for i in range(n):
            seed = seed0 + i
            # Redirect stdout/stderr for this run (including sim internal prints)
            with redirect_stdout(flog), redirect_stderr(flog):
                path, metrics = _run_one(seed)

            if target not in metrics:
                raise RuntimeError(f"Target '{target}' not found in {path}. Found: {list(metrics.keys())}")

            m = metrics[target]
            wins_list.append(m["wins"])
            income_list.append(m["income"])
            pen_list.append(m["penalty"])
            rev_list.append(m["revenue"])
            cost_list.append(m["cost"])

            line = (f"[seed={seed}] wins={int(m['wins'])} "
                    f"income={m['income']:.3f} rev={m['revenue']:.3f} "
                    f"cost={m['cost']:.3f} pen={m['penalty']:.3f}  metrics={path}")
            fsum.write(line + "\n")
            fsum.flush()

            # Minimal console progress (avoid flooding)
            try:
                os.write(1, (line + "\n").encode("utf-8", errors="ignore"))
            except Exception:
                pass

        w_mu, w_sd = _mean_std(wins_list)
        i_mu, i_sd = _mean_std(income_list)
        p_mu, p_sd = _mean_std(pen_list)
        r_mu, r_sd = _mean_std(rev_list)
        c_mu, c_sd = _mean_std(cost_list)

        ipw_list = [inc / max(w, 1.0) for inc, w in zip(income_list, wins_list)]
        ipw_mu, ipw_sd = _mean_std(ipw_list)
        fsum.write("\n=== SUMMARY (mean ± std) ===\n")
        fsum.write(f"wins   : {w_mu:.2f} ± {w_sd:.2f}\n")
        fsum.write(f"income : {i_mu:.2f} ± {i_sd:.2f}\n")
        fsum.write(f"penalty: {p_mu:.2f} ± {p_sd:.2f}\n")
        fsum.write(f"revenue: {r_mu:.2f} ± {r_sd:.2f}\n")
        fsum.write(f"cost   : {c_mu:.2f} ± {c_sd:.2f}\n")
        fsum.write(f"inc/win: {ipw_mu:.2f} ± {ipw_sd:.2f}\n")
        fsum.flush()

        try:
            os.write(1, (f"\nDone. Full log: {BATCH_LOG_PATH}\nSummary: {BATCH_SUMMARY_PATH}\n").encode("utf-8"))
        except Exception:
            pass

if __name__ == "__main__":
    # Adjust n/seed0/target as you like
    # run_many_to_file(n=2, seed0=0, target="Company4")
    run_many_to_file(n, seed0=0, target="Company1")