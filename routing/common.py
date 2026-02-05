from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx

Coord = Tuple[int, int]
TimedNode = Tuple[Coord, int]

MAX_TIME = 300

@dataclass(frozen=True)
class Qubit:
    id: int
    pos: Coord

class Reservations:

    def __init__(self, G: nx.Graph, blocked_edges: Optional[Set[frozenset]] = None) -> None:
        self.node_caps: Dict[Coord, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.edge_caps: Dict[frozenset, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.node_type: Dict[Coord, str] = {n: G.nodes[n]["type"] for n in G.nodes()}
        self.blocked_edges: Set[frozenset] = blocked_edges or set()

    def node_capacity(self, node: Coord) -> int:
        return 2 if self.node_type[node] == "IN" else 1

    def can_occupy(self, node: Coord, t: int) -> bool:
        return self.node_caps[node][t] < self.node_capacity(node)

    def occupy_node(self, node: Coord, t: int) -> None:
        self.node_caps[node][t] += 1

    def can_traverse(self, u: Coord, v: Coord, t: int) -> bool:
        e = frozenset({u, v})
        if e in self.blocked_edges:
            return False
        return self.edge_caps[e][t] == 0

    def traverse_edge(self, u: Coord, v: Coord, t: int) -> None:
        self.edge_caps[frozenset({u, v})][t] += 1

    def commit(self, path: List[TimedNode]) -> None:
        self.occupy_node(*path[0])
        for (u, t), (v, t2) in zip(path[:-1], path[1:]):
            self.occupy_node(v, t2)
            if u != v:
                self.traverse_edge(u, v, t)


class AStar:
    """A* search that minimizes (moves first, then time) and respects reservations."""

    @staticmethod
    def search(
        G: nx.Graph,
        start: Coord,
        goal: Coord,
        reservations: Reservations
    ) -> Optional[List[TimedNode]]:
        
        def h(n: Coord) -> int:
            return max(abs(n[0] - goal[0]), abs(n[1] - goal[1]))

        start_state: TimedNode = (start, 0)
        dist: Dict[TimedNode, Tuple[int, int]] = {start_state: (0, 0)}  
        came_from: Dict[TimedNode, TimedNode] = {}

        openQueue: List[Tuple[Tuple[int, int], TimedNode]] = []
        heappush(openQueue, ((h(start), 0), start_state))

        while openQueue:
            (_, _), (node, t) = heappop(openQueue)
            gm, gt = dist[(node, t)]

            if node == goal:
                path: List[TimedNode] = [(node, t)]
                cur = (node, t)
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                return list(reversed(path))

            if t >= MAX_TIME:
                continue

            successors: List[Tuple[Coord, int, int, int]] = []
            successors.append((node, t + 1, 0, 1))
            for nbr in G.neighbors(node):
                successors.append((nbr, t + 1, 1, 1)) 

            for (n2, t2, dm, dt) in successors:
                if not reservations.can_occupy(n2, t2):
                    continue
                if n2 != node and not reservations.can_traverse(node, n2, t):
                    continue

                g2 = (gm + dm, gt + dt)

                old = dist.get((n2, t2))
                if old is None or g2 < old:
                    dist[(n2, t2)] = g2
                    came_from[(n2, t2)] = (node, t)
                    f2 = (g2[0] + h(n2), g2[1] + h(n2))
                    heappush(openQueue, (f2, (n2, t2)))

        return None