# placements/interaction_placement_strategy.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import random

import networkx as nx

from placements.placement_strategy import PlacementStrategy


DECAY = 0.9


class InteractionPlacementStrategy(PlacementStrategy):

    def build_pairs(
        self,
        n_qubits: int,
        rounds: int,
        max_pairs_per_round: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:

        pair_ids = super().build_pairs(
            n_qubits=n_qubits,
            rounds=rounds,
            max_pairs_per_round=max_pairs_per_round,
            seed=seed,
        )
        self._last_pair_ids = pair_ids
        return pair_ids

    def place_qubits(
        self,
        sn_nodes: List[Tuple[int, int]],
        n_qubits: int,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:

        if self._last_pair_ids is None:
            rng = random.Random(seed)
            return rng.sample(sn_nodes, n_qubits)

        pair_ids = self._last_pair_ids
        interaction_weights: Dict[Tuple[int, int], float] = {}

        for idx, (a, b) in enumerate(pair_ids):
            if a == b:
                continue
            i, j = sorted((a, b))
            w = DECAY**idx
            interaction_weights[(i, j)] = interaction_weights.get((i, j), 0.0) + w

        H = nx.Graph()
        H.add_nodes_from(range(n_qubits))

        for (i, j), w in interaction_weights.items():
            H.add_edge(i, j, weight=w)

        pos = nx.spring_layout(H, weight="weight", seed=seed)

        def get_pos(i: int) -> Tuple[float, float]:
            return tuple(pos.get(i, (0.0, 0.0)))

        qubit_order = sorted(range(n_qubits), key=lambda i: get_pos(i))
        sn_nodes_sorted = sorted(sn_nodes, key=lambda xy: (xy[0], xy[1]))

        if n_qubits > len(sn_nodes_sorted):
            raise ValueError(
                f"n_qubits={n_qubits} exceeds available SN nodes ({len(sn_nodes_sorted)})."
            )

        coords_for_id: List[Optional[Tuple[int, int]]] = [None] * n_qubits

        for k, qubit_id in enumerate(qubit_order[:n_qubits]):
            coords_for_id[qubit_id] = sn_nodes_sorted[k]

        return [coords_for_id[i] for i in range(n_qubits)]
