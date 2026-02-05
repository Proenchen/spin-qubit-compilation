from typing import List, Optional, Tuple
import random

from placements.placement_strategy import PlacementStrategy


class RandomPlacementStrategy(PlacementStrategy):

    def place_qubits(
        self,
        sn_nodes: List[Tuple[int, int]],
        n_qubits: int,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        rng = random.Random(seed)
        return rng.sample(sn_nodes, n_qubits)