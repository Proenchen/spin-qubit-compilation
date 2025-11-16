# placements/interaction_placement_strategy.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import random

import networkx as nx

from placements.placement_strategy import PlacementStrategy


DECAY = 0.9


class InteractionPlacementStrategy(PlacementStrategy):
    """
    Platziert Qubits basierend auf ihrer Interaktionshäufigkeit:
    - Paare, die oft (und früh) interagieren, erhalten hohe Gewichte.
    - Ein gewichteter Graph der Qubits wird per spring_layout eingebettet.
    - Qubits werden dann so auf SN-Nodes gemappt, dass nahe Punkte
      im Layout auf räumlich nahe SN-Nodes fallen.
    """
    # -------------------------------
    # 1) Paare erzeugen + merken
    # -------------------------------
    def build_pairs(
        self,
        n_qubits: int,
        rounds: int,
        max_pairs_per_round: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Nutzt die Standard-Implementierung aus PlacementStrategy,
        speichert die erzeugte Paarfolge aber zusätzlich ab, damit
        place_qubits sie auswerten kann.
        """
        pair_ids = super().build_pairs(
            n_qubits=n_qubits,
            rounds=rounds,
            max_pairs_per_round=max_pairs_per_round,
            seed=seed,
        )
        self._last_pair_ids = pair_ids
        return pair_ids

    # -------------------------------
    # 2) Platzierungslogik
    # -------------------------------
    def place_qubits(
        self,
        sn_nodes: List[Tuple[int, int]],
        n_qubits: int,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Platziert Qubits so, dass stark (und früh) interagierende Qubits
        auf räumlich nahen SN-Nodes landen.

        Fallback: Falls keine Paarinformationen vorliegen, wird zufällig
        platziert (wie RandomPlacementStrategy).
        """
        # Fallback falls build_pairs nicht vorher aufgerufen wurde
        if self._last_pair_ids is None:
            rng = random.Random(seed)
            return rng.sample(sn_nodes, n_qubits)

        pair_ids = self._last_pair_ids

        # 2.1) Gewichtete Interaktionsmatrix aufbauen
        # keys: (min_id, max_id), value: gewichtete Summe
        interaction_weights: Dict[Tuple[int, int], float] = {}

        # frühe Interaktionen sollen höher gewichtet werden:
        # index 0 -> decay**0 = 1, index 1 -> decay**1, ...
        for idx, (a, b) in enumerate(pair_ids):
            if a == b:
                continue
            i, j = sorted((a, b))
            w = DECAY**idx
            interaction_weights[(i, j)] = interaction_weights.get((i, j), 0.0) + w

        # 2.2) Qubit-Interaktionsgraph aufbauen
        H = nx.Graph()
        H.add_nodes_from(range(n_qubits))

        for (i, j), w in interaction_weights.items():
            # networkx spring_layout nutzt 'weight' als Kanten-Gewicht
            H.add_edge(i, j, weight=w)

        # 2.3) 2D-Layout der Qubit-IDs bestimmen
        # strongly interacting Qubits -> nahe Punkte
        pos = nx.spring_layout(H, weight="weight", seed=seed)

        # 2D-Pos kann für einzelne Nodes evtl. fehlen (isolierte Knoten);
        # dann einfach (0, 0) als Default nehmen
        def get_pos(i: int) -> Tuple[float, float]:
            return tuple(pos.get(i, (0.0, 0.0)))

        # 2.4) Qubits nach Layout-Koordinaten sortieren
        # einfache 1D-Ordering über (x, y)
        qubit_order = sorted(range(n_qubits), key=lambda i: get_pos(i))

        # 2.5) SN-Nodes räumlich sortieren (z.B. lexikographisch nach (x, y))
        sn_nodes_sorted = sorted(sn_nodes, key=lambda xy: (xy[0], xy[1]))

        # sicherstellen, dass genügend SN-Nodes da sind (wurde im Caller schon geprüft)
        if n_qubits > len(sn_nodes_sorted):
            raise ValueError(
                f"n_qubits={n_qubits} exceeds available SN nodes ({len(sn_nodes_sorted)})."
            )

        # 2.6) Mapping: Qubit in qubit_order[k] -> sn_nodes_sorted[k]
        # build_network_and_place nutzt nur die Reihenfolge der coords und
        # weist IDs 0..n_qubits-1 in dieser Reihenfolge zu.
        # Wir wollen aber, dass Qubit-ID i an “seinen” Platz kommt.
        #
        # Deshalb drehen wir die Zuordnung um: Wir bauen ein Array coords_for_id
        # mit Länge n_qubits, in dem an Index i die Position steht, auf der
        # Qubit i landen soll.
        coords_for_id: List[Optional[Tuple[int, int]]] = [None] * n_qubits

        for k, qubit_id in enumerate(qubit_order[:n_qubits]):
            coords_for_id[qubit_id] = sn_nodes_sorted[k]

        # Reihenfolge der zurückgegebenen Koordinaten bestimmt die Zuordnung:
        # Qubit-ID i bekommt coords_for_id[i]
        return [coords_for_id[i] for i in range(n_qubits)]
