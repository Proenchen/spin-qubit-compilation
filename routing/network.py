import random
import networkx as nx

from typing import List, Tuple, Optional
from routing.common import Qubit


class NetworkBuilder:
    """Builds tiled 8-neighborhood graphs with typed nodes (IN/SN)."""

    @staticmethod
    def build_network(width: int, height: int) -> nx.Graph:
        """
        Create a width x height tiling of the 8-node pattern (no center),
        connected with 8-neighborhood edges. Node attribute 'type' ∈ {'IN','SN'}.
        Corners are 'IN'; others are 'SN'.

        Coordinates follow the original convention:
        - Each tile is offset by +2 in x for columns
        - Each tile is offset by -2 in y for rows (so rows go downward)
        """
        if not (isinstance(width, int) and isinstance(height, int) and width >= 1 and height >= 1):
            raise ValueError("width and height must be integers >= 1")

        # Single-tile template (no center)
        template_pos = {
            0: (-1,  1), 1: (0, 1),  2: (1, 1),
            3: (-1,  0),             4: (1, 0),
            5: (-1, -1), 6: (0, -1), 7: (1, -1),
        }
        corners = {0, 2, 5, 7}

        G = nx.Graph()

        # Place tiles in a width x height grid
        for j in range(height):         # rows
            for i in range(width):      # columns
                dx, dy = 2 * i, -2 * j  # keep original spacing/orientation
                for t_id, (x, y) in template_pos.items():
                    coord = (x + dx, y + dy)
                    if coord not in G:
                        G.add_node(coord, type=("IN" if t_id in corners else "SN"))

        # Connect 8-neighborhood across the entire tiled grid
        coords = list(G.nodes())
        coord_set = set(coords)
        for (x, y) in coords:
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    if ddx == 0 and ddy == 0:
                        continue
                    v = (x + ddx, y + ddy)
                    if v in coord_set:
                        G.add_edge((x, y), v)

        return G

    @staticmethod
    def place_qubits_and_make_pairs(
        width: int,
        height: int,
        n_qubits: int,
        *,
        rounds: int = 6,
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
        chosen_coords = rng.sample(sn_nodes, n_qubits)
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
