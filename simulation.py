from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx

from routing.routing_strategy import RoutingStrategy
from placements.placement_strategy import PlacementStrategy
from utils.animation import animate_mapf


@dataclass
class SimulationConfig:
    width: int
    height: int
    n_qubits: int
    rounds: int
    p_success: float
    p_repair: float
    seed: int


class RoutingSimulator:
    """
    Hauptklasse / Facade:
      - baut das Netzwerk über die Placement-Strategie
      - führt Routing über die Routing-Strategie aus
      - liefert ein SimulationResult mit allem drum und dran
    """

    def __init__(
        self,
        placement_strategy: PlacementStrategy,
        routing_strategy: RoutingStrategy,
        config: SimulationConfig,
    ):
        self.placement_strategy = placement_strategy
        self.routing_strategy = routing_strategy
        self.config = config

    def run(self):
        # 1) Netzwerk + Qubits + Paare
        G, qubits, pairs = self.placement_strategy.build_network_and_place(
            width=self.config.width,
            height=self.config.height,
            n_qubits=self.config.n_qubits,
            rounds=self.config.rounds,
            seed=self.config.seed
        )

        # 2) Routing
        timelines, edge_timebands = self.routing_strategy.route(
            G,
            qubits,
            pairs,
            self.config.p_success,
            self.config.p_repair,
        )

        return timelines, edge_timebands
