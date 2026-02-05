from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random
import networkx as nx

from placements.placement_strategy import PlacementStrategy
from routing.common import Qubit, Coord
from utils.network import NetworkBuilder
from routing.rotation_routing import RotationRoutingPlanner  


class ReverseTraversalPlacementStrategy(PlacementStrategy):

    def place_qubits(
        self,
        sn_nodes: List[Tuple[int, int]],
        n_qubits: int,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        rng = random.Random(seed)
        return rng.sample(sn_nodes, n_qubits)

    def build_network_and_place(
        self,
        width: int,
        height: int,
        n_qubits: int,
        rounds: int,
        max_pairs_per_round: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[nx.Graph, List[Qubit], List[Tuple[Qubit, Qubit]]]:

        pair_ids: List[Tuple[int, int]] = self.build_pairs(
            n_qubits=n_qubits,
            rounds=rounds,
            max_pairs_per_round=max_pairs_per_round,
            seed=seed,
        )

        G = NetworkBuilder.build_network(width, height)

        sn_nodes = [u for u, d in G.nodes(data=True) if d.get("type") == "SN"]
        if n_qubits > len(sn_nodes):
            raise ValueError(
                f"n_qubits={n_qubits} exceeds available SN nodes ({len(sn_nodes)})."
            )

        rng = random.Random(seed)
        initial_coords = rng.sample(sn_nodes, n_qubits)
        qubits_initial = [Qubit(i, coord) for i, coord in enumerate(initial_coords)]
        qubit_by_id_initial: Dict[int, Qubit] = {q.id: q for q in qubits_initial}

        reversed_pair_ids = list(reversed(pair_ids))
        warmup_pairs: List[Tuple[Qubit, Qubit]] = [
            (qubit_by_id_initial[a_id], qubit_by_id_initial[b_id])
            for (a_id, b_id) in reversed_pair_ids
        ]

        router = RotationRoutingPlanner()
        timelines, _ = router.route(
            G,
            qubits_initial,
            warmup_pairs,
            p_success=1.0,
            p_repair=1.0,
        )

        final_coords_by_id: Dict[int, Coord] = {
            qid: timeline[-1][0] for qid, timeline in timelines.items()
        }

        final_qubits: List[Qubit] = [
            Qubit(qid, final_coords_by_id[qid]) for qid in range(n_qubits)
        ]
        qubit_by_id_final: Dict[int, Qubit] = {q.id: q for q in final_qubits}

        final_pairs: List[Tuple[Qubit, Qubit]] = [
            (qubit_by_id_final[a_id], qubit_by_id_final[b_id])
            for (a_id, b_id) in pair_ids
        ]

        return G, final_qubits, final_pairs
