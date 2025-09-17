from statistics import mean
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from routing import Qubit, TimedNode, RoutingPlanner
from network import NetworkBuilder


def make_qubits_and_pairs():
    qubits: List[Qubit] = [
        Qubit(0, (1, -2)),
        Qubit(1, (1,  0)),
        Qubit(2, (-1, 0)),
        Qubit(3, (3, -2)),
        Qubit(4, (0, -3)),
        Qubit(5, (2,  1))
    ]

    pairs: List[Tuple[Qubit, Qubit]] = [
        (qubits[0], qubits[1]),
        (qubits[0], qubits[2]),
        (qubits[2], qubits[3]),
        (qubits[4], qubits[5]),
        (qubits[0], qubits[2]),
        (qubits[1], qubits[4]),
        (qubits[3], qubits[5]),
        (qubits[0], qubits[5]),
        (qubits[1], qubits[3]),
        (qubits[4], qubits[5]),
    ]
    return qubits, pairs


def makespan(plan: Dict[int, List[TimedNode]]) -> int:
    return max(path[-1][1] for path in plan.values()) if plan else 0


def run_once(seed: int, strategy: str, p_success: float) -> int:
    random.seed(seed)
    G = NetworkBuilder.build_network()
    qubits, pairs = make_qubits_and_pairs()
    planner = RoutingPlanner()

    if strategy == "try_until_success":
        plan = planner.try_until_success(G, qubits, pairs, p_success=p_success)
    elif strategy == "interleaved_routing":
        plan = planner.interleaved_routing(G, qubits, pairs, p_success=p_success)
    else:
        raise ValueError("Unknown strategy")
    return makespan(plan)


def sweep(p_values, seeds_per_p: int = 100):

    avg_try = []
    avg_inter = []

    for p in p_values:
        try_steps = []
        inter_steps = []
        for s in range(seeds_per_p):
            # try_until_success
            try:
                steps_try = run_once(s, "try_until_success", p)
                try_steps.append(steps_try)
            except Exception as e:
                # Keep going; you can log if you like
                pass

            # interleaved_routing
            try:
                steps_inter = run_once(s, "interleaved_routing", p)
                inter_steps.append(steps_inter)
            except Exception as e:
                pass

        avg_try.append(mean(try_steps) if try_steps else float("nan"))
        avg_inter.append(mean(inter_steps) if inter_steps else float("nan"))

        print(f"p={p:0.2f}  try_until_success avg={avg_try[-1]:.2f}  "
              f"interleaved_routing avg={avg_inter[-1]:.2f}  "
              f"(runs: {len(try_steps)}/{len(inter_steps)})")

    return p_values, avg_try, avg_inter


def plot_results(p_values, avg_try, avg_inter, out_path="D:\spin-qubit-compilation\p_success_sweep.png"):
    plt.figure()
    plt.plot(p_values, avg_try, marker="o", label="try_until_success")
    plt.plot(p_values, avg_inter, marker="s", label="interleaved_routing")
    plt.xlabel("p_success")
    plt.ylabel("Average makespan (time steps)")
    plt.title("Average makespan vs. p_success (100 seeds per point)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    p_values, avg_try, avg_inter = sweep(p_values=[(i+1)/10 for i in range(10)], seeds_per_p=100)
    plot_results(p_values, avg_try, avg_inter)
