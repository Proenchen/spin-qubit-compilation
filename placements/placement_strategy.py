from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import networkx as nx
import random
from routing.common import Qubit
from utils.network import NetworkBuilder


class PlacementStrategy(ABC):
    """Strategie-Interface für Qubit-Platzierung + Paarerzeugung."""

    def build_pairs(
        self,
        n_qubits: int,
        rounds: int,
        max_pairs_per_round: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Erzeugt eine zufällige Folge von Paaren NUR über Qubit-IDs (ohne Positionen).

        - IDs: 0 .. n_qubits-1
        - pro Runde: disjunkte Paare, d.h. jedes Qubit max. einmal
        - über Runden hinweg: Wiederholungen möglich
        """
        rng = random.Random(seed)
        pair_ids: List[Tuple[int, int]] = []

        for _ in range(rounds):
            ids = list(range(n_qubits))
            rng.shuffle(ids)

            possible_pairs = len(ids) // 2
            use_pairs = (
                possible_pairs
                if max_pairs_per_round is None
                else min(max_pairs_per_round, possible_pairs)
            )

            for k in range(use_pairs):
                a_id = ids[2 * k]
                b_id = ids[2 * k + 1]
                pair_ids.append((a_id, b_id))

        return pair_ids

    def build_network_and_place(
        self,
        width: int,
        height: int,
        n_qubits: int,
        rounds: int,
        max_pairs_per_round: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[nx.Graph, List[Qubit], List[Tuple[Qubit, Qubit]]]:
        """
        1. Erzeugt zuerst die Paarfolge (nur über IDs).
        2. Baut den Graphen.
        3. Platziert die Qubits.
        4. Mapped ID-Paare -> Qubit-Paare.
        """
        # 1) Paarfolge nur über IDs erzeugen
        pair_ids: List[Tuple[int, int]] = self.build_pairs(
            n_qubits=n_qubits,
            rounds=rounds,
            max_pairs_per_round=max_pairs_per_round,
            seed=seed,
        )

        # 2) Graph bauen
        G = NetworkBuilder.build_network(width, height)

        # 3) Nur SN-Knoten sammeln
        sn_nodes = [u for u, d in G.nodes(data=True) if d.get("type") == "SN"]
        if n_qubits > len(sn_nodes):
            raise ValueError(
                f"n_qubits={n_qubits} exceeds available SN nodes ({len(sn_nodes)})."
            )

        # 4) Qubits zufällig platzieren (ohne Wiederholung)
        chosen_coords = self.place_qubits(sn_nodes, n_qubits, seed)
        qubits = [Qubit(i, coord) for i, coord in enumerate(chosen_coords)]

        # Map für schnellen Zugriff
        qubit_by_id = {q.id: q for q in qubits}

        # 5) ID-Paare in Qubit-Paare umwandeln (Reihenfolge bleibt erhalten)
        pairs: List[Tuple[Qubit, Qubit]] = [
            (qubit_by_id[a_id], qubit_by_id[b_id]) for (a_id, b_id) in pair_ids
        ]

        return G, qubits, pairs

    @abstractmethod
    def place_qubits(
        self,
        sn_nodes: List[Tuple[int, int]],
        n_qubits: int,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Wählt aus allen SN-Nodes diejenigen Koordinaten aus,
        auf denen Qubits platziert werden sollen.
        """
        pass