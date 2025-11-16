from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import networkx as nx
import random
from routing.common import Qubit
from utils.network import NetworkBuilder

class PlacementStrategy(ABC):
    """Strategie-Interface für Qubit-Platzierung + Paarerzeugung."""

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
        Erzeugt den Graphen, platziert n_qubits auf 'SN'-Nodes und erzeugt eine zufällige Folge von Paaren.

        - Qubits werden ohne Wiederholung zufällig auf 'SN'-Koordinaten platziert.
        - Paaring wird rundenweise erzeugt; innerhalb einer Runde ist jedes Qubit höchstens einmal beteiligt.
          Über verschiedene Runden hinweg sind Wiederholungen möglich (wie in deinem Beispiel).

        Args:
            width, height: Kachelanzahl für den Graphen.
            n_qubits: Anzahl der zu platzierenden Qubits (<= #SN-Nodes).
            rounds: Anzahl der Runden (Zeit-Slots), in denen gepaart wird.
            max_pairs_per_round: Optionales Limit pro Runde; default: maximal mögliche disjunkte Paare.
            seed: Optionaler Seed für reproduzierbare Zufälligkeit.

        Returns:
            (G, qubits, pairs): Graph, platzierte Qubits und flache Liste aller Paare (in Runden-Reihenfolge).
        """
        rng = random.Random(seed)

        # 1) Graph bauen
        G = NetworkBuilder.build_network(width, height)

        # 2) Nur SN-Knoten sammeln
        sn_nodes = [u for u, d in G.nodes(data=True) if d.get("type") == "SN"]
        if n_qubits > len(sn_nodes):
            raise ValueError(f"n_qubits={n_qubits} exceeds available SN nodes ({len(sn_nodes)}).")

        # 3) Qubits zufällig platzieren (ohne Wiederholung)
        chosen_coords = self.place_qubits(sn_nodes, n_qubits, seed)
        qubits = [Qubit(i, coord) for i, coord in enumerate(chosen_coords)]

        # 4) Zufällige Paarfolge erzeugen (rundenweise, disjunkte Paare pro Runde)
        pairs: List[Tuple[Qubit, Qubit]] = []
        for _ in range(rounds):
            # zufällige Reihenfolge der Qubit-IDs
            ids = list(range(n_qubits))
            rng.shuffle(ids)

            # viele disjunkte Paare wie möglich (oder bis max_pairs_per_round)
            possible_pairs = len(ids) // 2
            use_pairs = possible_pairs if max_pairs_per_round is None else min(max_pairs_per_round, possible_pairs)

            for k in range(use_pairs):
                a = qubits[ids[2 * k]]
                b = qubits[ids[2 * k + 1]]
                pairs.append((a, b))

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