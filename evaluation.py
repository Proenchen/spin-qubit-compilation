import os
import csv
import random
import time
from statistics import mean, stdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scienceplots 

from routing.common import TimedNode  
from routing.routing_strategy import RoutingStrategy
from routing.default_routing import DefaultRoutingPlanner
from routing.routing_with_reroute import RerouteRoutingPlanner
from routing.rotation_routing import RotationRoutingPlanner
from routing.rotation_bypass_routing import HybridRotationRoutingPlanner
from placements.placement_strategy import PlacementStrategy
from placements.random_strategy import RandomPlacementStrategy
from placements.reverse_traversal_strategy import ReverseTraversalPlacementStrategy
from placements.interaction_placement_strategy import InteractionPlacementStrategy
from simulation import SimulationConfig, RoutingSimulator
from utils.network import NetworkBuilder  

def count_movements(timelines: Dict[int, List[TimedNode]]) -> int:
    moves = 0
    for path in timelines.values():
        for (c1, _), (c2, _) in zip(path[:-1], path[1:]):
            if c1 != c2:
                moves += 1
    return moves


def total_timesteps(timelines: Dict[int, List[TimedNode]]) -> int:
    max_t = 0
    for path in timelines.values():
        if path:
            max_t = max(max_t, path[-1][1])
    return max_t


def get_max_sn_nodes(width: int, height: int) -> int:
    """
    Baue das Netzwerk mit NetworkBuilder und zähle alle Knoten,
    deren Attribut 'type' == 'SN' ist.
    """
    G: nx.Graph = NetworkBuilder.build_network(width, height)
    n_sn = sum(1 for _, data in G.nodes(data=True) if data.get("type") == "SN")
    return n_sn



def evaluate_strategy(
    routing_strategy: RoutingStrategy,
    n_qubits_list: List[int],
    n_samples: int = 20,
    width: int = 4,
    height: int = 4,
    rounds: int = 5,
    p_success: float = 0.99,
    p_repair: float = 0.25,
) -> Tuple[List[float], List[float]]:

    placement: PlacementStrategy = RandomPlacementStrategy()

    avg_timesteps: List[float] = []
    avg_movements: List[float] = []

    strategy_dead = False

    print(f"\n=== Starte Evaluation für {routing_strategy.__class__.__name__} ===")

    for idx_q, n_qubits in enumerate(n_qubits_list, start=1):
        if strategy_dead:
            print(
                f"\nQubits: {n_qubits} ({idx_q}/{len(n_qubits_list)}) "
                f"→ skipped (already dead), setze NaN."
            )
            avg_timesteps.append(float("nan"))
            avg_movements.append(float("nan"))
            continue

        timesteps_samples: List[int] = []
        movements_samples: List[int] = []

        print(f"\nQubits: {n_qubits} ({idx_q}/{len(n_qubits_list)})")

        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx+1}/{n_samples} ...", end="", flush=True)

            base_seed = 1000 * n_qubits + sample_idx
            random.seed(base_seed)

            config = SimulationConfig(
                width=width,
                height=height,
                n_qubits=n_qubits,
                rounds=rounds,
                p_success=p_success,
                p_repair=p_repair,
                seed=base_seed,
            )

            simulator = RoutingSimulator(
                placement_strategy=placement,
                routing_strategy=routing_strategy,
                config=config,
            )

            try:
                timelines, _ = simulator.run()
            except Exception as e:
                print(f" FAILED ({e})", flush=True)
                continue

            timesteps_samples.append(total_timesteps(timelines))
            movements_samples.append(count_movements(timelines))

            print(" done", flush=True)

        if timesteps_samples:
            avg_timesteps.append(mean(timesteps_samples))
            avg_movements.append(mean(movements_samples))
        else:
            print(
                f"  WARNUNG: Keine erfolgreichen Samples für n_qubits={n_qubits}, "
                f"setze Wert auf NaN und markiere Strategie als DEAD."
            )
            avg_timesteps.append(float("nan"))
            avg_movements.append(float("nan"))
            strategy_dead = True  

    return avg_timesteps, avg_movements


def evaluate_strategy_vs_edge_expectation(
    routing_strategy: RoutingStrategy,
    expectation_values: List[float],
    n_qubits: int = 8,
    n_samples: int = 100,
    width: int = 3,
    height: int = 3,
    rounds: int = 5,
    min_expectation: float = 0.0,   
) -> Tuple[List[float], List[float]]:

    placement: PlacementStrategy = RandomPlacementStrategy()

    avg_timesteps: List[float] = []
    avg_movements: List[float] = []

    print(f"\n=== Starte Evaluation für {routing_strategy.__class__.__name__} "
          f"über Erwartungswert der funktionierenden Kanten ===")

    for idx_e, expectation in enumerate(expectation_values, start=1):
        if expectation < min_expectation:
            print(f"\nE={expectation:.2f} < {min_expectation} → skip → NaN")

            avg_timesteps.append(float("nan"))
            avg_movements.append(float("nan"))
            continue

        timesteps_samples: List[int] = []
        movements_samples: List[int] = []

        p_success = expectation
        p_repair = expectation

        print(f"\nE={expectation:.2f} → p_success=p_repair={expectation:.2f}")

        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx+1}/{n_samples} ...", end="", flush=True)

            base_seed = 10_000 * idx_e + sample_idx
            random.seed(base_seed)

            config = SimulationConfig(
                width=width,
                height=height,
                n_qubits=n_qubits,
                rounds=rounds,
                p_success=p_success,
                p_repair=p_repair,
                seed=base_seed,
            )

            simulator = RoutingSimulator(
                placement_strategy=placement,
                routing_strategy=routing_strategy,
                config=config,
            )

            try:
                timelines, _ = simulator.run()
            except Exception:
                print(" FAILED", flush=True)
                continue

            timesteps_samples.append(total_timesteps(timelines))
            movements_samples.append(count_movements(timelines))

            print(" done", flush=True)

        if timesteps_samples:
            avg_timesteps.append(mean(timesteps_samples))
            avg_movements.append(mean(movements_samples))
        else:
            avg_timesteps.append(float("nan"))
            avg_movements.append(float("nan"))

    return avg_timesteps, avg_movements


def evaluate_strategies_over_grids(
    routing_strategies: Dict[str, RoutingStrategy],
    grid_sizes: List[Tuple[int, int]],
    n_samples: int = 50,
    rounds: int = 5,
    p_success: float = 0.9,
    p_repair: float = 0.25,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[int]]:

    placement: PlacementStrategy = RandomPlacementStrategy()

    avg_timesteps: Dict[str, List[float]] = {name: [] for name in routing_strategies}
    avg_movements: Dict[str, List[float]] = {name: [] for name in routing_strategies}
    qubits_per_grid: List[int] = []

    for (width, height) in grid_sizes:
        # Anzahl SN-Nodes bestimmen
        n_sn = get_max_sn_nodes(width, height)
        if (width, height) == (5, 5):
            n_qubits = 12
        else:
            n_qubits = max(2, int(0.25 * n_sn))
        qubits_per_grid.append(n_qubits)

        print(f"\n=== Grid {width}x{height}, SN={n_sn}, Qubits={n_qubits} ===")

        for strat_name, routing_strategy in routing_strategies.items():
            print(f"\n--- Strategie: {strat_name} ---")

            timesteps_samples: List[int] = []
            movements_samples: List[int] = []

            for sample_idx in range(n_samples):
                print(f"  Sample {sample_idx+1}/{n_samples} ...", end="", flush=True)

                base_seed = 100_000 * (width * 10 + height) + sample_idx
                random.seed(base_seed)

                config = SimulationConfig(
                    width=width,
                    height=height,
                    n_qubits=n_qubits,
                    rounds=rounds,
                    p_success=p_success,
                    p_repair=p_repair,
                    seed=base_seed,
                )

                simulator = RoutingSimulator(
                    placement_strategy=placement,
                    routing_strategy=routing_strategy,
                    config=config,
                )

                try:
                    timelines, _ = simulator.run()
                except Exception as e:
                    print(f" FAILED ({e})", flush=True)
                    continue

                timesteps_samples.append(total_timesteps(timelines))
                movements_samples.append(count_movements(timelines))

                print(" done", flush=True)

            if timesteps_samples:
                avg_timesteps[strat_name].append(mean(timesteps_samples))
                avg_movements[strat_name].append(mean(movements_samples))
            else:
                avg_timesteps[strat_name].append(float("nan"))
                avg_movements[strat_name].append(float("nan"))

    return avg_timesteps, avg_movements, qubits_per_grid


def evaluate_placements_for_routing(
    routing_strategy: RoutingStrategy,
    placement_strategies: Dict[str, PlacementStrategy],
    width: int = 3,
    height: int = 3,
    n_samples: int = 100,
    rounds: int = 5,
    p_success: float = 0.998,
    p_repair: float = 0.25,
    n_qubits: int = 8
) -> Tuple[Dict[str, float], Dict[str, float], int]:

    n_sn = get_max_sn_nodes(width, height)

    print(f"\n=== Router: {routing_strategy.__class__.__name__} ===")
    print(f"Grid: {width}x{height}, SN={n_sn}, Qubits={n_qubits}")
    print(f"Samples={n_samples}, rounds={rounds}, p_success={p_success}, p_repair={p_repair}")

    avg_timesteps: Dict[str, float] = {}
    avg_movements: Dict[str, float] = {}

    for pname, placement in placement_strategies.items():
        print(f"\n--- Placement-Strategie: {pname} ---")

        timesteps_samples: List[int] = []
        movements_samples: List[int] = []

        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx+1}/{n_samples} ...", end="", flush=True)

            base_seed = 1_000_000 * hash(pname) % (2**31 - 1) + sample_idx
            random.seed(base_seed)

            config = SimulationConfig(
                width=width,
                height=height,
                n_qubits=n_qubits,
                rounds=rounds,
                p_success=p_success,
                p_repair=p_repair,
                seed=base_seed,
            )

            simulator = RoutingSimulator(
                placement_strategy=placement,
                routing_strategy=routing_strategy,
                config=config,
            )

            try:
                timelines, _ = simulator.run()
            except Exception as e:
                print(f" FAILED ({e})", flush=True)
                continue

            timesteps_samples.append(total_timesteps(timelines))
            movements_samples.append(count_movements(timelines))

            print(" done", flush=True)

        if timesteps_samples:
            avg_timesteps[pname] = mean(timesteps_samples)
            avg_movements[pname] = mean(movements_samples)
        else:
            avg_timesteps[pname] = float("nan")
            avg_movements[pname] = float("nan")

    return avg_timesteps, avg_movements, n_qubits


def evaluate_exception_rates_for_strategies_3x3(
    n_qubits_min: int = 2,
    n_qubits_max: int = 24,
    n_samples: int = 100,
    width: int = 3,
    height: int = 3,
    p_success: float = 0.998,
    p_repair: float = 0.25,
) -> Dict[str, List[float]]:
    
    routing_strategies: Dict[str, RoutingStrategy] = {
        "Default": DefaultRoutingPlanner(),
        "Reroute": RerouteRoutingPlanner(),
        "Rotation": RotationRoutingPlanner(),
        "HybridRotation": HybridRotationRoutingPlanner(),
    }

    placement: PlacementStrategy = RandomPlacementStrategy()

    n_qubits_list = list(range(n_qubits_min, n_qubits_max + 1))
    exception_rates: Dict[str, List[float]] = {name: [] for name in routing_strategies}

    strategy_dead: Dict[str, bool] = {name: False for name in routing_strategies}

    print("\n=== Exception-Rate Evaluation (3x3 Grid, RandomPlacement) ===")
    print(f"Qubits {n_qubits_min}..{n_qubits_max}, Samples={n_samples}, "
          f"p_success={p_success}, p_repair={p_repair}")

    for n_qubits in n_qubits_list:
        print(f"\n--- n_qubits = {n_qubits} ---")

        for strat_name, routing_strategy in routing_strategies.items():
            if strategy_dead[strat_name]:
                exception_rates[strat_name].append(1.0)
                print(f"  Strategie: {strat_name} → skipped (already dead), rate=1")
                continue

            print(f"  Strategie: {strat_name}")

            fail_count = 0

            for sample_idx in range(n_samples):
                print(f"    Sample {sample_idx+1}/{n_samples} ...", end="", flush=True)

                base_seed = (
                    (hash(strat_name) & 0x7FFFFFFF) * 10_000
                    + n_qubits * 100
                    + sample_idx
                )
                random.seed(base_seed)

                config = SimulationConfig(
                    width=width,
                    height=height,
                    n_qubits=n_qubits,
                    rounds=5,
                    p_success=p_success,
                    p_repair=p_repair,
                    seed=base_seed,
                )

                simulator = RoutingSimulator(
                    placement_strategy=placement,
                    routing_strategy=routing_strategy,
                    config=config,
                )

                try:
                    simulator.run()
                    print(" ok", flush=True)
                except Exception as e:
                    fail_count += 1
                    print(f" FAILED ({e})", flush=True)

            rate = fail_count / n_samples
            exception_rates[strat_name].append(rate)
            print(f"  → Exception-Rate {strat_name} @ n_qubits={n_qubits}: {rate:.3f}")

            if rate >= 1.0:
                strategy_dead[strat_name] = True
                print(f"  → {strat_name} marked as DEAD (will skip future runs)")

    plt.figure(figsize=(10, 6))
    for strat_name, rates in exception_rates.items():
        plt.plot(n_qubits_list, rates, marker="o", label=strat_name)

    plt.xlabel("Anzahl Qubits")
    plt.ylabel("Exception-Rate")
    plt.title(
        "Exception-Rate vs. Qubit-Anzahl\n"
        "3x3-Grid, RandomPlacement, p_success=0.998, p_repair=0.25, 100 Samples"
    )
    plt.xticks(n_qubits_list, rotation=45)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("D:\\Uni\\failrate_qubits.png")
    plt.show()

    return exception_rates


def evaluate_exception_rates_vs_edge_expectation_3x3(
    expectation_values: List[float] = None,
    n_qubits: int = 6,
    n_samples: int = 20,
    width: int = 3,
    height: int = 3,
    rounds: int = 5,
) -> Dict[str, List[float]]:
    if expectation_values is None:
        expectation_values = [round(1.0 - 0.025 * i, 3) for i in range(41)]

    routing_strategies: Dict[str, RoutingStrategy] = {
        "Default": DefaultRoutingPlanner(),
        "Reroute": RerouteRoutingPlanner(),
        "Rotation": RotationRoutingPlanner(),
        "HybridRotation": HybridRotationRoutingPlanner(),
    }

    placement: PlacementStrategy = RandomPlacementStrategy()

    exception_rates: Dict[str, List[float]] = {name: [] for name in routing_strategies}

    strategy_dead: Dict[str, bool] = {name: False for name in routing_strategies}

    print("\n=== Exception-Rate vs. Erwartungswert E (3x3 Grid, RandomPlacement) ===")
    print(f"Grid: {width}x{height}, n_qubits={n_qubits}, Samples={n_samples}, rounds={rounds}")
    print("E-Werte (absteigend ausgewertet):", expectation_values)

    for E in expectation_values:
        print(f"\n--- Erwartungswert E = {E:.2f} ---")

        p_success = E
        p_repair = E

        for strat_name, routing_strategy in routing_strategies.items():
            if strategy_dead[strat_name]:
                exception_rates[strat_name].append(1.0)
                print(f"  Strategie: {strat_name} → skipped (already dead), rate=0")
                continue

            print(f"  Strategie: {strat_name}")

            fail_count = 0

            for sample_idx in range(n_samples):
                print(f"    Sample {sample_idx+1}/{n_samples} ...", end="", flush=True)

                base_seed = 42
                random.seed(base_seed)

                config = SimulationConfig(
                    width=width,
                    height=height,
                    n_qubits=n_qubits,
                    rounds=rounds,
                    p_success=p_success,
                    p_repair=p_repair,
                    seed=base_seed,
                )

                simulator = RoutingSimulator(
                    placement_strategy=placement,
                    routing_strategy=routing_strategy,
                    config=config,
                )

                try:
                    simulator.run()
                    print(" ok", flush=True)
                except Exception as e:
                    fail_count += 1
                    print(f" FAILED ({e})", flush=True)

            rate = fail_count / n_samples
            exception_rates[strat_name].append(rate)
            print(f"  → Exception-Rate {strat_name} @ E={E:.2f}: {rate:.3f}")

            if rate >= 1.0:
                strategy_dead[strat_name] = True
                print(f"  → {strat_name} marked as DEAD (will skip future E with rate=0)")

    E_sorted = sorted(expectation_values) 

    plt.figure(figsize=(10, 6))
    for strat_name, rates in exception_rates.items():
        E_to_rate = {E: r for E, r in zip(expectation_values, rates)}
        rates_sorted = [E_to_rate[E] for E in E_sorted]

        plt.plot(E_sorted, rates_sorted, marker="o", label=strat_name)

    plt.xlabel("Erwartungswert E funktionierender Kanten")
    plt.ylabel("Exception-Rate")
    plt.title(
        "Exception-Rate vs. Erwartungswert E\n"
        f"3x3-Grid, n_qubits={n_qubits}, RandomPlacement, 100 Samples pro E"
    )
    plt.xticks(E_sorted, rotation=45)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("D:\\Uni\\failrate_expectation.png")
    plt.show()

    return exception_rates


def evaluate_runtimes_for_strategies_3x3(
    n_qubits_min: int = 2,
    n_qubits_max: int = 24,
    n_samples: int = 100,
    width: int = 3,
    height: int = 3,
    p_success: float = 0.998,
    p_repair: float = 0.25,
) -> Dict[str, List[float]]:
    routing_strategies: Dict[str, RoutingStrategy] = {
        "Default": DefaultRoutingPlanner(),
        "Reroute": RerouteRoutingPlanner(),
        "Rotation": RotationRoutingPlanner(),
        "HybridRotation": HybridRotationRoutingPlanner(),
    }

    placement: PlacementStrategy = RandomPlacementStrategy()

    n_qubits_list = list(range(n_qubits_min, n_qubits_max + 1))
    runtimes: Dict[str, List[float]] = {name: [] for name in routing_strategies}

    # Track, ob eine Strategie dauerhaft "tot" ist
    strategy_dead: Dict[str, bool] = {name: False for name in routing_strategies}

    print("\n=== Runtime Evaluation (3x3 Grid, RandomPlacement) ===")
    print(f"Qubits {n_qubits_min}..{n_qubits_max}, Samples={n_samples}, "
          f"p_success={p_success}, p_repair={p_repair}")

    for n_qubits in n_qubits_list:
        print(f"\n--- n_qubits = {n_qubits} ---")

        for strat_name, routing_strategy in routing_strategies.items():
            if strategy_dead[strat_name]:
                runtimes[strat_name].append(float("nan"))
                print(f"  Strategie: {strat_name} → skipped (already dead), runtime=NaN")
                continue

            print(f"  Strategie: {strat_name}")

            sample_runtimes: List[float] = []
            fail_count = 0

            for sample_idx in range(n_samples):
                print(f"    Sample {sample_idx+1}/{n_samples} ...", end="", flush=True)

                base_seed = (
                    (hash(strat_name) & 0x7FFFFFFF) * 10_000
                    + n_qubits * 100
                    + sample_idx
                )
                random.seed(base_seed)

                config = SimulationConfig(
                    width=width,
                    height=height,
                    n_qubits=n_qubits,
                    rounds=5,
                    p_success=p_success,
                    p_repair=p_repair,
                    seed=base_seed,
                )

                simulator = RoutingSimulator(
                    placement_strategy=placement,
                    routing_strategy=routing_strategy,
                    config=config,
                )

                start_t = time.perf_counter()
                try:
                    simulator.run()
                    print(" ok", flush=True)
                except Exception as e:
                    fail_count += 1
                    print(f" FAILED ({e})", flush=True)
                finally:
                    duration = time.perf_counter() - start_t
                    sample_runtimes.append(duration)

            avg_runtime = mean(sample_runtimes)
            runtimes[strat_name].append(avg_runtime)
            print(f"  → Avg runtime {strat_name} @ n_qubits={n_qubits}: {avg_runtime:.6f} s")

            if fail_count == n_samples:
                print(
                    f"  WARNUNG: Keine erfolgreichen Runs für {strat_name} "
                    f"@ n_qubits={n_qubits}, markiere Strategie als DEAD."
                )
                strategy_dead[strat_name] = True

    plt.figure(figsize=(10, 6))
    for strat_name, rt in runtimes.items():
        plt.plot(n_qubits_list, rt, marker="o", label=strat_name)

    plt.xlabel("Anzahl Qubits")
    plt.ylabel("Durchschnittliche Laufzeit pro Run [s]")
    plt.title(
        "Laufzeit vs. Qubit-Anzahl\n"
        "3x3-Grid, RandomPlacement, "
        f"p_success={p_success}, p_repair={p_repair}, {n_samples} Samples"
    )
    plt.xticks(n_qubits_list, rotation=45)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("D:\\Uni\\runtime_qubits.png")
    plt.show()

    return runtimes


def evaluate_strategy_with_errorbars(
    routing_strategy: RoutingStrategy,
    n_qubits_list: List[int],
    n_samples: int = 20,
    width: int = 3,
    height: int = 3,
    rounds: int = 5,
    p_success: float = 0.99,
    p_repair: float = 0.25,
) -> Tuple[List[float], List[float], List[float], List[float], List[int]]:
    placement: PlacementStrategy = RandomPlacementStrategy()

    t_mean: List[float] = []
    t_std: List[float] = []
    m_mean: List[float] = []
    m_std: List[float] = []
    n_success: List[int] = []

    strategy_dead = False

    print(f"\n=== Starte Evaluation für {routing_strategy.__class__.__name__} ===")

    for idx_q, n_qubits in enumerate(n_qubits_list, start=1):
        if strategy_dead:
            print(
                f"\nQubits: {n_qubits} ({idx_q}/{len(n_qubits_list)}) "
                f"→ skipped (already dead), setze NaN."
            )
            t_mean.append(float("nan"))
            t_std.append(float("nan"))
            m_mean.append(float("nan"))
            m_std.append(float("nan"))
            n_success.append(0)
            continue

        timesteps_samples: List[int] = []
        movements_samples: List[int] = []

        print(f"\nQubits: {n_qubits} ({idx_q}/{len(n_qubits_list)})")

        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx+1}/{n_samples} ...", end="", flush=True)

            base_seed = 1000 * n_qubits + sample_idx
            random.seed(base_seed)

            config = SimulationConfig(
                width=width,
                height=height,
                n_qubits=n_qubits,
                rounds=rounds,
                p_success=p_success,
                p_repair=p_repair,
                seed=base_seed,
            )

            simulator = RoutingSimulator(
                placement_strategy=placement,
                routing_strategy=routing_strategy,
                config=config,
            )

            try:
                timelines, _ = simulator.run()
            except Exception as e:
                print(f" FAILED ({e})", flush=True)
                continue

            timesteps_samples.append(total_timesteps(timelines))
            movements_samples.append(count_movements(timelines))
            print(" done", flush=True)

        n_success.append(len(timesteps_samples))

        if timesteps_samples:
            t_mean.append(mean(timesteps_samples))
            m_mean.append(mean(movements_samples))

            t_std.append(stdev(timesteps_samples) if len(timesteps_samples) > 1 else 0.0)
            m_std.append(stdev(movements_samples) if len(movements_samples) > 1 else 0.0)
        else:
            print(
                f"  WARNUNG: Keine erfolgreichen Samples für n_qubits={n_qubits}, "
                f"setze Wert auf NaN und markiere Strategie als DEAD."
            )
            t_mean.append(float("nan"))
            t_std.append(float("nan"))
            m_mean.append(float("nan"))
            m_std.append(float("nan"))
            strategy_dead = True

    return t_mean, t_std, m_mean, m_std, n_success


def save_results_csv(
    path: str,
    n_qubits_list: List[int],
    strategy_name: str,
    timesteps_mean: List[float],
    timesteps_std: List[float],
    movements_mean: List[float],
    movements_std: List[float],
    n_success: List[int],
    n_samples: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "strategy",
                    "n_qubits",
                    "timesteps_mean",
                    "timesteps_std",
                    "movements_mean",
                    "movements_std",
                    "n_success",
                    "n_samples",
                ]
            )
        for i, nq in enumerate(n_qubits_list):
            writer.writerow(
                [
                    strategy_name,
                    nq,
                    timesteps_mean[i],
                    timesteps_std[i],
                    movements_mean[i],
                    movements_std[i],
                    n_success[i],
                    n_samples,
                ]
            )


def load_results_csv(path: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    data: Dict[str, Dict[int, Dict[str, float]]] = {}
    if not os.path.exists(path):
        return data

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strat = row["strategy"]
            nq = int(row["n_qubits"])
            data.setdefault(strat, {})
            data[strat][nq] = {
                "timesteps_mean": float(row["timesteps_mean"]),
                "timesteps_std": float(row["timesteps_std"]),
                "movements_mean": float(row["movements_mean"]),
                "movements_std": float(row["movements_std"]),
                "n_success": int(float(row["n_success"])),
                "n_samples": int(float(row["n_samples"])),
            }
    return data


def plot_two_axis_with_errorbars(
    n_qubits_list: List[int],
    results: Dict[str, Dict[int, Dict[str, float]]],
    out_png: str,
    title: str,
) -> None:
    with plt.style.context(["science", "nature"]):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        for strat_name, per_nq in results.items():
            x = np.array(n_qubits_list, dtype=float)

            t_mean = np.array([per_nq.get(nq, {}).get("timesteps_mean", np.nan) for nq in n_qubits_list], dtype=float)
            t_std = np.array([per_nq.get(nq, {}).get("timesteps_std", np.nan) for nq in n_qubits_list], dtype=float)

            m_mean = np.array([per_nq.get(nq, {}).get("movements_mean", np.nan) for nq in n_qubits_list], dtype=float)
            m_std = np.array([per_nq.get(nq, {}).get("movements_std", np.nan) for nq in n_qubits_list], dtype=float)

            ax1.errorbar(
                x,
                t_mean,
                yerr=t_std,
                marker="o",
                linestyle="-",
                capsize=3,
                label=f"{strat_name} (timesteps)",
            )

            ax2.errorbar(
                x,
                m_mean,
                yerr=m_std,
                marker="s",
                linestyle="--",
                capsize=3,
                label=f"{strat_name} (movements)",
            )

        ax1.set_xlabel("Number of Qubits")
        ax1.set_ylabel("Timesteps")
        ax2.set_ylabel("Movements")

        ax1.set_title(title)
        ax1.set_xticks(n_qubits_list)
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="best",
            fontsize=9,
            frameon=True,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")   
        legend.get_frame().set_alpha(1.0) 


        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.show()


def plot_two_axis_no_errorbars(
    n_qubits_list: List[int],
    results: Dict[str, Dict[int, Dict[str, float]]],
    out_png: str,
    title: str,
) -> None:
    with plt.style.context(["science", "nature"]):
        plt.rcParams.update({
            "font.size": 11,       
            "axes.labelsize": 11,   
            "axes.titlesize": 11,   
            "xtick.labelsize": 11,  
            "ytick.labelsize": 11,
            "legend.fontsize": 11, 
        })
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        for strat_name, per_nq in results.items():
            x = np.array(n_qubits_list, dtype=float)

            t_mean = np.array(
                [per_nq.get(nq, {}).get("timesteps_mean", np.nan) for nq in n_qubits_list],
                dtype=float,
            )
            m_mean = np.array(
                [per_nq.get(nq, {}).get("movements_mean", np.nan) for nq in n_qubits_list],
                dtype=float,
            )

            ax1.plot(
                x,
                t_mean,
                marker="o",
                linestyle="-",
                color = "#ed9015" if strat_name == "Rotation Algorithm with Waiting" else "tab:blue",
                label=f"{strat_name} (Timesteps)",
            )

            ax2.plot(
                x,
                m_mean,
                marker="s",
                linestyle="--",
                color = "#ed9015" if strat_name == "Rotation Algorithm with Waiting" else "tab:blue",
                label=f"{strat_name} (Movements)",
            )

        ax1.set_xlabel("Number of Qubits", labelpad=10)
        ax1.set_ylabel("Mean Timesteps", labelpad=12)
        ax2.set_ylabel("Mean Movements", labelpad=12)

        ax1.set_title(title)
        ax1.set_xticks(n_qubits_list)
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="best",
            borderaxespad=1.5,
            borderpad=0.8, 
            frameon=True,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")   
        legend.get_frame().set_alpha(1.0) 

        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.show()



def main():
    width, height = 3, 3
    rounds = 5
    p_success = 0.99
    p_repair = 0.25
    n_samples = 50

    n_qubits_list = list(range(2, 25))  

    csv_path = "results_strategy_3x3.csv"
    plot_path = "strategy_3x3_timesteps_movements.pdf"

    existing = load_results_csv(csv_path)

    strategies = {
        "Default": DefaultRoutingPlanner(),
        "Rotation": RotationRoutingPlanner(),
    }

    for strat_name, strat in strategies.items():
        missing_any = (
            strat_name not in existing
            or any(nq not in existing[strat_name] for nq in n_qubits_list)
        )
        if missing_any:
            already = existing.get(strat_name, {})
            todo_nqs = [nq for nq in n_qubits_list if nq not in already]
            if not todo_nqs:
                continue

            t_mean, t_std, m_mean, m_std, n_success = evaluate_strategy_with_errorbars(
                routing_strategy=strat,
                n_qubits_list=todo_nqs,
                n_samples=n_samples,
                width=width,
                height=height,
                rounds=rounds,
                p_success=p_success,
                p_repair=p_repair,
            )

            save_results_csv(
                path=csv_path,
                n_qubits_list=todo_nqs,
                strategy_name=strat_name,
                timesteps_mean=t_mean,
                timesteps_std=t_std,
                movements_mean=m_mean,
                movements_std=m_std,
                n_success=n_success,
                n_samples=n_samples,
            )

            existing = load_results_csv(csv_path)

    plot_two_axis_no_errorbars(
        n_qubits_list=n_qubits_list,
        results={
            "Path Algorithm with Waiting": existing.get("Default", {}),
            "Rotation Algorithm with Waiting": existing.get("Rotation", {}),
        },
        out_png=plot_path,
        title="",
    )

    print(f"\nCSV gespeichert unter: {os.path.abspath(csv_path)}")
    print(f"Plot gespeichert unter: {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    main()