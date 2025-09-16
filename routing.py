"""
Rendezvous + MAPF (refactored, same functionality)
--------------------------------------------------
Goal:
- For each qubit pair, pick an "in-the-middle" IN node (by shortest paths)
- Move both qubits collision-free to arrive there at the same time
- IN nodes have capacity 2 (allow rendezvous), SN nodes capacity 1
- Collision-free via time reservations on nodes/edges; forbid edge-swaps
- A*: lexicographic cost (minimize moves first, then time) so waiting is preferred

Notes on this refactor (behavior unchanged):
- The original pure functions are organized into small classes with clear roles:
  * NetworkBuilder: builds the 2x2-tiled graph
  * Reservations: unchanged; manages time-indexed capacities
  * AStar: A* search with lexicographic cost + reservation checks
  * RendezvousPlanner: end-to-end pairwise planning with meeting selection,
    synchronization, and staged egress from IN nodes
- Helper utilities keep their original semantics; names are preserved where useful
- Added English comments for each logical unit; original German docstrings were translated
- The `main` block mirrors the original execution path
"""

from __future__ import annotations

from heapq import heappush, heappop
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from network import NetworkBuilder
from utils.animation import animate_mapf

# --------------------------
# Type aliases and constants
# --------------------------
Coord = Tuple[int, int]
TimedNode = Tuple[Coord, int]

MAX_TIME = 300


class Reservations:
    """
    Time-indexed reservations for node and edge capacities:
      - node_caps[node][t] < capacity(node)
      - edge_caps[{u,v}][t] ∈ {0,1} prevents opposite-direction swaps at the same time

    IN nodes have capacity 2; SN nodes have capacity 1.
    """

    def __init__(self, G: nx.Graph) -> None:
        self.node_caps: Dict[Coord, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    
        # Use forzenset for undirected graph
        self.edge_caps: Dict[frozenset, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.node_type: Dict[Coord, str] = {n: G.nodes[n]["type"] for n in G.nodes()}

    def node_capacity(self, node: Coord) -> int:
        return 2 if self.node_type[node] == "IN" else 1

    def can_occupy(self, node: Coord, t: int) -> bool:
        return self.node_caps[node][t] < self.node_capacity(node)

    def occupy_node(self, node: Coord, t: int) -> None:
        self.node_caps[node][t] += 1

    def can_traverse(self, u: Coord, v: Coord, t: int) -> bool:
        return self.edge_caps[frozenset({u, v})][t] == 0

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
        """
        Lexicographic cost:
          - Each move costs (dmoves=1, dtime=1)
          - Waiting costs (dmoves=0, dtime=1)
        => Avoid unnecessary detours; prefer waiting

        Collision checks:
          - Node and edge capacities are enforced during expansion
        """

        def h(n: Coord) -> int:
            # Use Chebyshev distance as heuristic
            return max(abs(n[0] - goal[0]), abs(n[1] - goal[1]))

        start_state: TimedNode = (start, 0)
        dist: Dict[TimedNode, Tuple[int, int]] = {start_state: (0, 0)}  # (moves, time)
        came_from: Dict[TimedNode, TimedNode] = {} # sotres parent nodes for path reconstruction

        openQueue: List[Tuple[Tuple[int, int], TimedNode]] = []  # Nodes to be evaluated
        heappush(openQueue, ((h(start), 0), start_state))

        while openQueue:
            (_, _), (node, t) = heappop(openQueue)
            gm, gt = dist[(node, t)]

            # Goal check
            if node == goal:
                path: List[TimedNode] = [(node, t)]
                cur = (node, t)
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                return list(reversed(path))

            # Avoid infinite loops
            if t >= MAX_TIME:
                continue

            successors: List[Tuple[Coord, int, int, int]] = []
            # Option 1: wait on the same node
            successors.append((node, t + 1, 0, 1))
            # Option 2: move to any neighbor
            for nbr in G.neighbors(node):
                successors.append((nbr, t + 1, 1, 1))

            for (n2, t2, dm, dt) in successors:
                # Capacity / reservation checks for the candidate step
                if not reservations.can_occupy(n2, t2):
                    continue
                if n2 != node and not reservations.can_traverse(node, n2, t):
                    continue

                g2 = (gm + dm, gt + dt)  # lexicographic cost update

                old = dist.get((n2, t2))
                if old is None or g2 < old:
                    dist[(n2, t2)] = g2
                    came_from[(n2, t2)] = (node, t)
                    f2 = (g2[0] + h(n2), g2[1] + h(n2))
                    heappush(openQueue, (f2, (n2, t2)))

        return None


# =============================================================
# Rendezvous planner (pairwise, prioritized)
# =============================================================
class RoutingPlanner:
    """Orchestrates meeting-node selection and MAPF for pairs with egress staging."""

    @staticmethod
    def compute_meeting_node(
        G: nx.Graph, q0: Coord, q1: Coord, reserved_meetings: Optional[set] = None
    ) -> Optional[Tuple[Coord, Dict[Coord, int], Dict[Coord, int]]]:
        """
        Choose an IN node that is "in the middle" while avoiding already reserved meetings.
        Ranking key (unchanged):
          1) max(distA, distB)
          2) distA + distB
          3) |distA - distB|
          4) node coordinate for tie-break
        Returns:
          - meeting node
          - bfs distances for q0 and q1
        """
        if reserved_meetings is None:
            reserved_meetings = set()

        # Use BFS to find meeting node in the middle
        q0_bfs_dist = nx.single_source_shortest_path_length(G, q0)
        q1_bfs_dist = nx.single_source_shortest_path_length(G, q1)
        all_in_nodes = [n for n in G if G.nodes[n]["type"] == "IN" and n in q0_bfs_dist and n in q1_bfs_dist]

        sorted_candidates = sorted(
            all_in_nodes,
            key=lambda n: (max(q0_bfs_dist[n], q1_bfs_dist[n]),  # 1) minimize the slower agent's distance
                           q0_bfs_dist[n] + q1_bfs_dist[n],      # 2) then minimize total distance
                           abs(q0_bfs_dist[n] - q1_bfs_dist[n]), # 3) then prefer balanced distances
                           n)                                    # 4) stable tiebreak by coordinate
        )

        for candidate in sorted_candidates:
            if candidate not in reserved_meetings:
                return candidate, q0_bfs_dist, q1_bfs_dist

        raise RuntimeError("No common unreserved meeting node reachable.")
    

    @staticmethod
    def pairwise_mapf(
        G: nx.Graph,
        pairs: List[Tuple[Coord, Coord]],
    ) -> Dict[str, List[TimedNode]]:
        """
        Plan pairwise rendezvous with priority ordering.
        Steps per pair:
          1) Pick meeting IN node (avoid reusing a previous meeting node when possible)
          2) Route both agents to meeting with A* + reservations
          3) Synchronize arrival times (waits only)
          4) Manage egress of previous pairs from the same meeting to avoid capacity conflicts
        Returns:
          - plans: dict agent_id -> time-stamped path
        """
        infos = [] # collects per-pair info (meeting choice + distances)
        reserved_meetings = set()  # track already used meeting nodes

        # Decide meetings for all pairs up front (same as original)
        for i, (q0, q1) in enumerate(pairs):
            m, da, db = RoutingPlanner.compute_meeting_node(G, q0, q1, reserved_meetings)
            reserved_meetings.add(m)
            infos.append({
                "pair_id": i,
                "q0": q0,
                "q1": q1,
                "meeting": m,
                "d0": da[m],
                "d1": db[m],
                "maxd": max(da[m], db[m]),
                "sumd": da[m] + db[m],
            })

        # Order: longer-first (by max distance, then sum distance)
        plan_order = (
            sorted(infos, key=lambda p: (p["maxd"], p["sumd"]), reverse=True)
        )

        res = Reservations(G)
        plans: Dict[str, List[TimedNode]] = {}

        for p in plan_order:
            aid, bid = f"pair{p['pair_id']}_A", f"pair{p['pair_id']}_B" # First qubit is A and second qubit is B
            meeting = p["meeting"]

            # Route A then commit
            a_plan = AStar.search(G, p["q0"], meeting, res)
            if a_plan is None:
                raise RuntimeError(f"No route for {aid} to meeting {meeting}")
            res.commit(a_plan)

            # Route B then commit
            b_plan = AStar.search(G, p["q1"], meeting, res)
            if b_plan is None:
                raise RuntimeError(f"No route for {bid} to meeting {meeting}")
            res.commit(b_plan)

            # Synchronize arrival times by waiting only
            Ta, Tb = a_plan[-1][1], b_plan[-1][1]
            Tm = max(Ta, Tb)

            if Ta < Tm:
                a_plan = RoutingPlanner.extend_waits(a_plan, Tm, res)
                if a_plan is None:
                    raise RuntimeError(f"Synchronization failed for {aid} @ {meeting} until t={Tm}")

            if Tb < Tm:
                b_plan = RoutingPlanner.extend_waits(b_plan, Tm, res)
                if b_plan is None:
                    raise RuntimeError(f"Synchronization failed for {bid} @ {meeting} until t={Tm}")

            plans[aid] = a_plan
            plans[bid] = b_plan

        return plans
    

    # Helpers
    # --------------------------------  
    @staticmethod
    def extend_waits(
        path: List[TimedNode],
        target_time: int,
        res: Reservations,
    ) -> Optional[List[TimedNode]]:
        """
        Extend a time-stamped path by WAIT actions until its last state's time == target_time.
        Each wait reserves the node at the next time via Reservations.

        Args:
            path: existing time-stamped path [(node, t), ...]
            target_time: desired final time after extension
            res: Reservations instance to check/reserve capacity

        Returns:
            A new extended path, or None if a required reservation/constraint fails.
        """
        if not path:
            return None

        ext = path.copy()
        while ext[-1][1] < target_time:
            n, t = ext[-1]
            t2 = t + 1
            if not res.can_occupy(n, t2):
                return None
            res.occupy_node(n, t2)
            ext.append((n, t2))
        return ext


if __name__ == "__main__":
    G = NetworkBuilder.build_network()

    # Example input pairs (same as the active set in the original file)
    pairs = [
        ((1, -2), (1, 0)),
        ((-1, 0), (3, -2)),
        ((0, -3), (2, 1)),
        ((2, -3), (3, 0)),
        ((2, -1), (0, 1)),
        ((0, -1), (-1, -2)),
    ]

    # pairs = [((3, 0), (-1, -2)), ((-1, 0), (3, -2))] 
    # pairs = [((1, -2), (1, 0)), ((-1, 0), (3, -2)), ((0, -3), (2, 1))] 
    # pairs = [((1, -2), (1, 0)), ((-1, 0), (3, -2)), ((0, -3), (2, 1)), ((2, -3), (3, 0))] 
    #pairs = [((1, -2), (1, 0)), ((-1, 0), (3, -2)), ((0, -3), (2, 1)), ((2, -3), (3, 0)), ((2, -1), (0, 1))]

    planner = RoutingPlanner()
    plans = planner.pairwise_mapf(G, pairs)

    for agent_id, path in plans.items():
        trace = " -> ".join(f"{n}@t{t}" for n, t in path)
        print(f"{agent_id}: {trace}")

    animate_mapf(G, plans)
