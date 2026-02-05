from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set

import networkx as nx

from routing.common import Qubit, TimedNode


class RoutingStrategy(ABC):

    @abstractmethod
    def route(
        self,
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float,
        p_repair: float,
    ) -> Tuple[Dict[int, List[TimedNode]], List[Tuple[int, int, Set[frozenset]]]]:
        pass
