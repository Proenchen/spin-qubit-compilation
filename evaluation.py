"""
Evaluation script for comparing DefaultRoutingPlanner vs RerouteRoutingPlanner
across grid sizes 1x1 .. 10x10 and qubit counts 2..#SN with rounds=5.

Metrics collected per run:
- movements: total number of qubit coordinate changes across all timelines
- timesteps: the maximum timestamp found in timelines
- runtime_s: wall-clock seconds for the routing call
- exception: 1 if the planner raised an exception (e.g., no valid routing), else 0

Outputs:
- evaluation_results.csv (full raw records)
- plots: movements.png, timesteps.png, runtime.png, exception_rate.png
"""

import sys
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from routing.routing_with_reroute import RerouteRoutingPlanner 
from routing.default_routing import DefaultRoutingPlanner
from routing.common import Qubit, TimedNode
from utils.network import NetworkBuilder

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def pick_qubits_and_pairs(
    G: nx.Graph,
    n_qubits: int,
    rounds: int,
    seed: int
) -> Tuple[List[Qubit], List[Tuple[Qubit, Qubit]]]:
    """
    Self-contained generator for qubits & pairs if you don't want to rely on an external NetworkBuilder.
    - Places qubits on distinct SN nodes uniformly at random.
    - For each round, randomly pairs qubits (last one dropped if odd).
    - Returns a *flat list* of pairs (concatenated across rounds), matching the expectation in your planners.
    """
    rng = random.Random(seed)
    sn_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "SN"]
    if n_qubits > len(sn_nodes):
        raise ValueError(f"Requested {n_qubits} qubits but only {len(sn_nodes)} SN nodes exist.")
    chosen = rng.sample(sn_nodes, n_qubits)
    qubits = [Qubit(id=i, pos=chosen[i]) for i in range(n_qubits)]

    pairs: List[Tuple[Qubit, Qubit]] = []
    ids = list(range(n_qubits))
    for _ in range(rounds):
        rng.shuffle(ids)
        # Pair consecutive ids
        for i in range(0, len(ids) - 1, 2):
            qa = qubits[ids[i]]
            qb = qubits[ids[i + 1]]
            pairs.append((qa, qb))
    return qubits, pairs


def count_movements(timelines: Dict[int, List[TimedNode]]) -> int:
    """
    Count total coordinate changes across all qubit timelines.
    Assumes timelines are sequences of (coord, t) with non-decreasing t and possibly repeated coords.
    """
    moves = 0
    for qid, path in timelines.items():
        # Group by consecutive time steps and compare coordinates
        for (c1, _t1), (c2, _t2) in zip(path[:-1], path[1:]):
            if c1 != c2:
                moves += 1
    return moves


def total_timesteps(timelines: Dict[int, List[TimedNode]]) -> int:
    max_t = 0
    for path in timelines.values():
        if path:
            max_t = max(max_t, path[-1][1])
    return max_t


def run_one(
    PlannerClass,
    G: nx.Graph,
    qubits: List[Qubit],
    pairs: List[Tuple[Qubit, Qubit]],
    p_success: float,
    p_repair: float,
):
    start = time.perf_counter()
    timelines, _ = PlannerClass.route(G, qubits, pairs, p_success=p_success, p_repair=p_repair)
    end = time.perf_counter()
    return {
        "movements": count_movements(timelines),
        "timesteps": total_timesteps(timelines),
        "runtime_s": end - start,
        "exception": 0,
    }


def evaluate(
    max_size: int,
    rounds: int,
    base_seed: int,
    n_seed_samples: int,
    p_success: float,
    p_repair: float,
    limit_qubits: Optional[int],
) -> pd.DataFrame:
    """
    Iterate grid sizes and qubit counts, run both planners, and collect metrics.
    For each (grid, n_qubits) combination, evaluate n_seed_samples random seeds.
    Prints progress to the console.
    """
    records = []
    grid_list = [(n, n) for n in range(2, max_size + 1)]

    # Rough total count for progress tracking
    total_tasks = 0
    for (w, h) in grid_list:
        G_tmp = NetworkBuilder.build_network(w, h)
        sn_nodes_tmp = [n for n in G_tmp.nodes if G_tmp.nodes[n].get("type") == "SN"]
        max_qubits_tmp = len(sn_nodes_tmp) // 2
        if limit_qubits is not None:
            max_qubits_tmp = min(max_qubits_tmp, limit_qubits)
        total_tasks += (max_qubits_tmp - 1) * n_seed_samples * 2  # both planners

    task_counter = 0
    print(f"Starting evaluation across {len(grid_list)} grids, total ~{total_tasks} planner runs\n")

    for (w, h) in grid_list:
        print(f"\n=== Evaluating grid {w}x{h} ===")
        G = NetworkBuilder.build_network(w, h)
        sn_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "SN"]
        max_qubits = len(sn_nodes) // 2
        if limit_qubits is not None:
            max_qubits = min(max_qubits, limit_qubits)

        for n_qubits in range(2, max_qubits + 1):
            for seed_offset in range(n_seed_samples):
                seed = base_seed + seed_offset

                try:
                    qubits, pairs = pick_qubits_and_pairs(G, n_qubits=n_qubits, rounds=rounds, seed=seed)
                except Exception as e:
                    print(f"[WARN] Skipping seed {seed} for {w}x{h}, n_qubits={n_qubits}: {e}")
                    continue

                for algo_name, PlannerClass in [
                    ("Default", DefaultRoutingPlanner),
                    ("Reroute", RerouteRoutingPlanner),
                ]:
                    task_counter += 1
                    print(
                        f"[{task_counter:5d}/{total_tasks}] "
                        f"Grid={w}x{h}, n_qubits={n_qubits}, seed={seed}, algo={algo_name}...",
                        end=""
                    )
                    sys.stdout.flush()

                    try:
                        m = run_one(PlannerClass, G, qubits, pairs, p_success, p_repair)
                        status = "OK"
                    except Exception as e:
                        m = {"movements": np.nan, "timesteps": np.nan, "runtime_s": np.nan, "exception": 1}
                        status = f"FAIL ({type(e).__name__})"

                    records.append({
                        "algo": algo_name,
                        "width": w,
                        "height": h,
                        "n_qubits": n_qubits,
                        "seed": seed,
                        **m
                    })

                    print(f" {status}, runtime={m.get('runtime_s', np.nan):.3f}s")

    print("\nAll evaluations completed.\n")
    return pd.DataFrame.from_records(records)



def make_plots(df: pd.DataFrame, out_prefix: str = "eval"):
    """
    Erstellt für JEDE Netzwerkgröße (width x height) getrennte Plots:
    - Mean movements vs n_qubits per algo
    - Mean timesteps vs n_qubits per algo
    - Mean runtime vs n_qubits per algo
    - Exception rate vs n_qubits per algo

    Dateien werden als:
      {out_prefix}_{w}x{h}_movements.png
      {out_prefix}_{w}x{h}_timesteps.png
      {out_prefix}_{w}x{h}_runtime.png
      {out_prefix}_{w}x{h}_exception_rate.png
    gespeichert.
    """

    # Alle vorhandenen Grid-Größen (sortiert)
    grids = (
        df[["width", "height"]]
        .drop_duplicates()
        .sort_values(["width", "height"])
        .itertuples(index=False, name=None)
    )

    for (w, h) in grids:
        df_grid = df[(df["width"] == w) & (df["height"] == h)].copy()
        if df_grid.empty:
            continue

        # Aggregation pro Algo und n_qubits für DIESES Grid
        agg = df_grid.groupby(["algo", "n_qubits"], as_index=False).agg({
            "movements": "mean",
            "timesteps": "mean",
            "runtime_s": "mean",
            "exception": "mean",
        })

        # Falls durch Ausreißer/NaNs alles weg wäre, Überspringen
        if agg.empty:
            continue

        # ---------- Movements ----------
        plt.figure()
        for algo in sorted(agg["algo"].unique()):
            sub = agg[agg["algo"] == algo]
            plt.plot(sub["n_qubits"], sub["movements"], marker="o", label=algo)
        plt.xlabel("n_qubits")
        plt.ylabel("mean movements")
        plt.title(f"Mean movements vs n_qubits ({w}x{h})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{w}x{h}_movements.png", dpi=150)
        plt.close()

        # ---------- Timesteps ----------
        plt.figure()
        for algo in sorted(agg["algo"].unique()):
            sub = agg[agg["algo"] == algo]
            plt.plot(sub["n_qubits"], sub["timesteps"], marker="o", label=algo)
        plt.xlabel("n_qubits")
        plt.ylabel("mean timesteps")
        plt.title(f"Mean timesteps vs n_qubits ({w}x{h})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{w}x{h}_timesteps.png", dpi=150)
        plt.close()

        # ---------- Runtime ----------
        plt.figure()
        for algo in sorted(agg["algo"].unique()):
            sub = agg[agg["algo"] == algo]
            plt.plot(sub["n_qubits"], sub["runtime_s"], marker="o", label=algo)
        plt.xlabel("n_qubits")
        plt.ylabel("mean runtime (s)")
        plt.title(f"Mean runtime vs n_qubits ({w}x{h})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{w}x{h}_runtime.png", dpi=150)
        plt.close()

        # ---------- Exception rate ----------
        plt.figure()
        for algo in sorted(agg["algo"].unique()):
            sub = agg[agg["algo"] == algo]
            plt.plot(sub["n_qubits"], sub["exception"], marker="o", label=algo)
        plt.xlabel("n_qubits")
        plt.ylabel("exception rate")
        plt.title(f"Exception rate vs n_qubits ({w}x{h})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{w}x{h}_exception_rate.png", dpi=150)
        plt.close()



def main():
    df = evaluate(
        max_size=4,
        rounds=5,
        base_seed=42,
        n_seed_samples=20,
        p_success=0.98,
        p_repair=0.25,
        limit_qubits=None,
    )
    df.to_csv("eval_results.csv", index=False)
    make_plots(df, out_prefix="eval")
    print("Wrote eval_results.csv and plots with prefix eval_*.png")


# --- NEU/GEÄNDERT: Utility für inklusiven Zahlenbereich ---
def _range_inclusive(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    k = int(round((stop - start) / step))
    xs = [round(start + i * step, 10) for i in range(k + 1)]
    # Falls durch Rundung der Endpunkt knapp verfehlt wird, füge ihn hinzu
    if xs and abs(xs[-1] - stop) > 1e-9 and stop > xs[-1]:
        xs.append(round(stop, 10))
    return sorted([x for x in xs if x >= start - 1e-12 and x <= stop + 1e-12])

# --- ERSETZT die alte param_grid(...) ---
def param_grid(
    ps_start: float = 0.05, ps_stop: float = 1.0, ps_step: float = 0.05,
    pr_start: Optional[float] = None, pr_stop: Optional[float] = None, pr_step: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Liefert alle (p_success, p_repair)-Kombinationen mit getrennten Rastern.
    Wenn pr_* nicht gesetzt ist, wird das p_success-Raster übernommen.
    """
    if pr_start is None: pr_start = ps_start
    if pr_stop  is None: pr_stop  = ps_stop
    if pr_step  is None: pr_step  = ps_step

    ps_vals = _range_inclusive(ps_start, ps_stop, ps_step)
    pr_vals = _range_inclusive(pr_start, pr_stop, pr_step)

    return [(float(ps), float(pr)) for ps in ps_vals for pr in pr_vals]

# --- SIGNATUR/LOGIK anpassen, damit getrennte Rasters & mehrere Circuits laufen ---
def evaluate_fixed_circuit_ps_pr(
    width: int = 2,
    height: int = 2,
    n_qubits: int = 4,
    rounds: int = 5,
    seed: int = 44,  # Basis-Seed; pro Circuit wird offsettet
    # getrennte Parameter-Raster:
    ps_start: float = 0.05,
    ps_stop: float = 1.0,
    ps_step: float = 0.05,
    pr_start: Optional[float] = None,
    pr_stop: Optional[float] = None,
    pr_step: Optional[float] = None,
    # NEU: mehrere Circuits je (ps, pr)
    n_circuits: int = 10,
    out_prefix: str = "pspr_eval"
) -> pd.DataFrame:
    """
    Sweep über getrennte Rasters für p_success (ps_*) und p_repair (pr_*).
    Für jedes (ps, pr) werden n_circuits verschiedene Circuits (neue Platzierungen+Paare) getestet.
    Schreibt:
      - {out_prefix}_results.csv (RAW, alle Replikate)
      - {out_prefix}_means.csv   (pro algo, ps, pr gemittelt)
      - Heatmaps (aus den Mittelwerten)
    """
    # Netzwerk fixieren (nur die Circuits variieren)
    G = NetworkBuilder.build_network(width, height)
    sn_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "SN"]
    if n_qubits > len(sn_nodes):
        raise ValueError(
            f"Requested {n_qubits} qubits but only {len(sn_nodes)} SN nodes exist in a {width}x{height} network."
        )

    combos = param_grid(
        ps_start=ps_start, ps_stop=ps_stop, ps_step=ps_step,
        pr_start=pr_start, pr_stop=pr_stop, pr_step=pr_step
    )
    total_runs = len(combos) * 2 * n_circuits  # 2 Planner * Replikate
    print(f"Total (p_success, p_repair) combos: {len(combos)}; runs (with {n_circuits} circuits each, 2 algos): {total_runs}")

    records = []
    run_idx = 0

    for (ps, pr) in combos:
        for circuit_idx in range(n_circuits):
            # pro Circuit neue Platzierungen/Paare (Seed variieren)
            seed_used = seed + circuit_idx
            qubits, pairs = pick_qubits_and_pairs(G, n_qubits=n_qubits, rounds=rounds, seed=seed_used)

            for algo_name, PlannerClass in [
                ("Default", DefaultRoutingPlanner),
                ("Reroute", RerouteRoutingPlanner),
            ]:
                run_idx += 1
                print(
                    f"[{run_idx:5d}/{total_runs}] ps={ps:.2f}, pr={pr:.2f}, "
                    f"circuit={circuit_idx+1}/{n_circuits}, algo={algo_name} ... ",
                    end=""
                )
                sys.stdout.flush()
                try:
                    m = run_one(PlannerClass, G, qubits, pairs, p_success=ps, p_repair=pr)
                    status = "OK"
                except Exception as e:
                    m = {"movements": np.nan, "timesteps": np.nan, "runtime_s": np.nan, "exception": 1}
                    status = f"FAIL ({type(e).__name__})"

                records.append({
                    "algo": algo_name,
                    "width": width,
                    "height": height,
                    "n_qubits": n_qubits,
                    "rounds": rounds,
                    "seed_base": seed,
                    "seed_used": seed_used,
                    "circuit_idx": circuit_idx,
                    "p_success": ps,
                    "p_repair": pr,
                    **m
                })
                print(status)

    df = pd.DataFrame.from_records(records)
    raw_csv = f"{out_prefix}_results.csv"
    df.to_csv(raw_csv, index=False)
    print(f"Wrote {raw_csv}")

    # --- Mittelwerte je (algo, ps, pr) berechnen ---
    group_cols = ["algo", "width", "height", "n_qubits", "rounds", "p_success", "p_repair"]
    metrics = ["movements", "timesteps", "runtime_s", "exception"]
    df_means = (
        df.groupby(group_cols, as_index=False)[metrics]
          .mean(numeric_only=True)
          .rename(columns={"exception": "exception_rate"})
    )
    mean_csv = f"{out_prefix}_means.csv"
    df_means.to_csv(mean_csv, index=False)
    print(f"Wrote {mean_csv}")

    # ---- Heatmaps (aus den Mittelwerten) ----
    def _plot_heatmap(df_algo: pd.DataFrame, metric: str, title: str, out_path: str):
        d = df_algo.pivot_table(index="p_success", columns="p_repair", values=metric, aggfunc="mean")
        d = d.sort_index().sort_index(axis=1)
        plt.figure()
        im = plt.imshow(
            d.values, aspect="auto", origin="lower",
            extent=[d.columns.min(), d.columns.max(), d.index.min(), d.index.max()]
        )
        plt.colorbar(im, label=metric)
        plt.xlabel("p_repair")
        plt.ylabel("p_success")
        plt.title(title)
        xticks = np.round(np.linspace(d.columns.min(), d.columns.max(), 8), 2)
        yticks = np.round(np.linspace(d.index.min(), d.index.max(), 8), 2)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    for algo in sorted(df_means["algo"].unique()):
        df_a = df_means[df_means["algo"] == algo]
        _plot_heatmap(df_a, "movements",
                      f"Mean movements ({width}x{height}, {n_qubits} qubits) – {algo}",
                      f"{out_prefix}_{algo}_movements.png")
        _plot_heatmap(df_a, "timesteps",
                      f"Mean timesteps ({width}x{height}, {n_qubits} qubits) – {algo}",
                      f"{out_prefix}_{algo}_timesteps.png")
        _plot_heatmap(df_a, "runtime_s",
                      f"Mean runtime (s) ({width}x{height}, {n_qubits} qubits) – {algo}",
                      f"{out_prefix}_{algo}_runtime.png")
        _plot_heatmap(df_a, "exception_rate",
                      f"Exception rate ({width}x{height}, {n_qubits} qubits) – {algo}",
                      f"{out_prefix}_{algo}_exception_rate.png")

    print("Wrote heatmaps with prefix "
          f"{out_prefix}_<algo>_{{movements,timesteps,runtime,exception_rate}}.png")

    return df



if __name__ == "__main__":
    evaluate_fixed_circuit_ps_pr(
        width=2,
        height=2,
        n_qubits=4,
        rounds=5,
        seed=42,
        ps_start=0.90, ps_stop=1.00, ps_step=0.01,  # p_success: 0.90 .. 1.00
        pr_start=0.10, pr_stop=1.00, pr_step=0.10,  # p_repair: 0.10 .. 1.00
        n_circuits=10,                               # <<— 10 Circuits pro (ps, pr)
        out_prefix="pspr_eval"
    )
