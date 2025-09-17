from __future__ import annotations

from copy import deepcopy 
from dataclasses import dataclass
from heapq import heappush, heappop
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx

from network import NetworkBuilder
from utils.animation import animate_mapf

# --------------------------
# Type aliases and constants
# --------------------------
Coord = Tuple[int, int]
TimedNode = Tuple[Coord, int]

@dataclass(frozen=True)
class Qubit:
    id: int
    pos: Coord

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
        def h(n: Coord) -> int:
            return max(abs(n[0] - goal[0]), abs(n[1] - goal[1]))

        start_state: TimedNode = (start, 0)
        dist: Dict[TimedNode, Tuple[int, int]] = {start_state: (0, 0)}  # (moves, time)
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
            successors.append((node, t + 1, 0, 1))  # wait
            for nbr in G.neighbors(node):
                successors.append((nbr, t + 1, 1, 1))  # move

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


class RoutingPlanner:
    """Orchestrates meeting-node selection and MAPF for pairs with egress staging."""

    @staticmethod
    def meeting_candidates(
        G: nx.Graph, q0: Coord, q1: Coord, reserved_meetings: Optional[Set[Coord]] = None
    ) -> List[Coord]:
        """
        Liefert die sortierte Kandidatenliste von IN-Knoten (gleiche Ranking-Logik wie compute_meeting_node),
        filtert bereits reservierte Meetings raus.
        """
        if reserved_meetings is None:
            reserved_meetings = set()

        q0_bfs_dist = nx.single_source_shortest_path_length(G, q0)
        q1_bfs_dist = nx.single_source_shortest_path_length(G, q1)
        all_in_nodes = [n for n in G
                        if G.nodes[n]["type"] == "IN"
                        and n in q0_bfs_dist and n in q1_bfs_dist]

        sorted_candidates = sorted(
            all_in_nodes,
            key=lambda n: (max(q0_bfs_dist[n], q1_bfs_dist[n]),
                           q0_bfs_dist[n] + q1_bfs_dist[n],
                           abs(q0_bfs_dist[n] - q1_bfs_dist[n]),
                           n)
        )

        return [c for c in sorted_candidates if c not in reserved_meetings]

    @staticmethod
    def pairwise_mapf(
        G: nx.Graph,
        pairs: List[Tuple[Qubit, Qubit]],
        static_blockers: Optional[Dict[Coord, int]] = None,
    ) -> Dict[int, List[TimedNode]]:
        """
        Plant eine Batch von Paaren. Falls Synchronisation am Meeting scheitert,
        wird automatisch auf alternative Meeting-Kandidaten ausgewichen.
        'static_blockers': node -> bereits belegte Slots (z.B. ruhende Qubits aus anderen Batches).
        """
        # 1) Reihenfolge bestimmen anhand der besten Kandidaten (nur für Ranking)
        prelim_infos = []
        tmp_reserved: Set[Coord] = set()  # nur für Ranking, nicht final bindend
        for i, (qa, qb) in enumerate(pairs):
            # beste Kandidatenliste auf Basis aktueller Starts
            cands = RoutingPlanner.meeting_candidates(G, qa.pos, qb.pos, tmp_reserved)
            if not cands:
                raise RuntimeError(f"No meeting candidates available for pair {qa.id}-{qb.id}")
            m0 = cands[0]
            # Distanzen für Ranking bestimmen
            da = nx.single_source_shortest_path_length(G, qa.pos)
            db = nx.single_source_shortest_path_length(G, qb.pos)
            prelim_infos.append({
                "pair_id": i,
                "qa": qa,
                "qb": qb,
                "best_m": m0,
                "maxd": max(da[m0], db[m0]),
                "sumd": da[m0] + db[m0],
            })
            # Für Ranking-Exklusivität schon mal markieren (wie vorher), rein heuristisch
            tmp_reserved.add(m0)

        plan_order = sorted(prelim_infos, key=lambda p: (p["maxd"], p["sumd"]), reverse=True)

        # 2) Echte Planung mit Reservierungen
        res = Reservations(G)

        # 2a) Statische Blocker: belegen ihre Nodes für alle t der Batch
        if static_blockers:
            for node, count in static_blockers.items():
                for t in range(0, MAX_TIME + 1):
                    for _ in range(count):
                        res.occupy_node(node, t)

        plans: Dict[int, List[TimedNode]] = {}
        reserved_meetings: Set[Coord] = set()  # jetzt wirklich verbindlich

        for item in plan_order:
            qa: Qubit = item["qa"]
            qb: Qubit = item["qb"]

            # Kandidatenliste unter Berücksichtigung schon verbindlich reservierter Meetings
            candidates = RoutingPlanner.meeting_candidates(G, qa.pos, qb.pos, reserved_meetings)
            if not candidates:
                raise RuntimeError(f"No meeting candidates left for pair {qa.id}-{qb.id}")

            placed = False
            for meeting in candidates:
                # Auf einem Snapshot planen (Rollback möglich)
                res_try = deepcopy(res)

                # Route A
                a_plan = AStar.search(G, qa.pos, meeting, res_try)
                if a_plan is None:
                    continue
                res_try.commit(a_plan)

                # Route B
                b_plan = AStar.search(G, qb.pos, meeting, res_try)
                if b_plan is None:
                    continue
                res_try.commit(b_plan)

                # Synchronisierung
                Ta, Tb = a_plan[-1][1], b_plan[-1][1]
                Tm = max(Ta, Tb)

                if Ta < Tm:
                    a_plan_ext = RoutingPlanner.extend_waits(a_plan, Tm, res_try)
                    if a_plan_ext is None:
                        # Synchronisation an diesem Meeting gescheitert → anderer Kandidat
                        continue
                    a_plan = a_plan_ext

                if Tb < Tm:
                    b_plan_ext = RoutingPlanner.extend_waits(b_plan, Tm, res_try)
                    if b_plan_ext is None:
                        # Synchronisation an diesem Meeting gescheitert → anderer Kandidat
                        continue
                    b_plan = b_plan_ext

                # Erfolg: Snapshot übernehmen, Meeting fixieren, Pläne mergen
                res = res_try
                reserved_meetings.add(meeting)
                plans[qa.id] = a_plan
                plans[qb.id] = b_plan
                placed = True
                break

            if not placed:
                raise RuntimeError(f"No feasible meeting (with sync) for pair {qa.id}-{qb.id}")

        return plans
    
    @staticmethod
    def plan_in_batches(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
    ) -> Dict[int, List[TimedNode]]:
        """
        Zerlegt die 'pairs' in Batches ohne Qubit-Überschneidung. Sobald ein Paar ein
        bereits in der Batch enthaltenes Qubit verwenden würde, wird die aktuelle
        Batch geplanten und die Positionen werden auf die Endpositionen gesetzt.

        Rückgabe:
          - batch_plans: Liste von Dicts {qubit_id -> [TimedNode]}, eins pro Batch (Zeiten beginnen jeweils bei 0)
          - final_positions: Endpositionen aller Qubits nach allen Batches
        """
        # aktuelle Positionen separat tracken, Qubit ist frozen
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}

        # pairs werden in der gegebenen Reihenfolge verarbeitet
        i = 0
        n = len(pairs)
        batch_plans: List[Dict[int, List[TimedNode]]] = []

        while i < n:
            used_in_batch: Set[int] = set()
            batch: List[Tuple[Qubit, Qubit]] = []

            # Paare einsammeln bis Konflikt auftritt
            j = i
            while j < n:
                qa, qb = pairs[j]
                a_id, b_id = qa.id, qb.id
                if a_id in used_in_batch or b_id in used_in_batch:
                    break  # Konflikt -> Batch schließen (dieses Paar bleibt für nächste Runde)
                # Qubit-Objekte mit *aktueller* Startposition erzeugen:
                qa_now = Qubit(a_id, current_pos[a_id])
                qb_now = Qubit(b_id, current_pos[b_id])
                batch.append((qa_now, qb_now))
                used_in_batch.update([a_id, b_id])
                j += 1

            # Sicherheitsnetz: Falls aus irgendeinem Grund keine Paare gesammelt wurden,
            # nimm zumindest das nächste Paar allein, damit keine Endlosschleife entsteht.
            if not batch:
                qa, qb = pairs[j]
                qa_now = Qubit(qa.id, current_pos[qa.id])
                qb_now = Qubit(qb.id, current_pos[qb.id])
                batch = [(qa_now, qb_now)]
                j += 1

            static_blockers: Dict[Coord, int] = defaultdict(int)
            for q in qubits:
                if q.id not in used_in_batch:
                    static_blockers[current_pos[q.id]] += 1

            # Plane die Batch mit Blockern + Kandidaten-Fallback
            plans = RoutingPlanner.pairwise_mapf(G, batch, static_blockers=static_blockers)
            batch_plans.append(plans)

            # Positionen der in der Batch beteiligten Qubits auf Endpositionen setzen
            for qid in plans.keys():
                end_coord, _ = plans[qid][-1]
                current_pos[qid] = end_coord

            # Der Konflikt-Auslöser wird in der nächsten Runde erneut betrachtet
            i = j

        return RoutingPlanner.stitch_batches(qubits, batch_plans)
    

    # Helpers
    #--------------------------------------------

    @staticmethod
    def extend_waits(
        path: List[TimedNode],
        target_time: int,
        res: Reservations,
    ) -> Optional[List[TimedNode]]:
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
    
    @staticmethod
    def stitch_batches(
        qubits: List[Qubit],
        batch_plans: List[Dict[int, List[TimedNode]]],
    ) -> Dict[int, List[TimedNode]]:
        """
        Nimmt die per-Batch-Pläne (Zeit jeweils ab 0) und erzeugt einen
        globalen Plan {qubit_id -> [(coord, t), ...]} mit durchlaufender Zeitachse.
        Qubits, die in einer Batch nicht aktiv sind, warten an ihrer aktuellen Position.
        """
        # Start-Positionen aus dem allerersten Zustand ableiten
        initial_pos: Dict[int, Coord] = {}
        for q in qubits:
            initial_pos[q.id] = q.pos

        timelines: Dict[int, List[TimedNode]] = {q.id: [(initial_pos[q.id], 0)] for q in qubits}
        current_pos: Dict[int, Coord] = initial_pos.copy()

        t_offset = 0  # globaler Zeitversatz

        for plans in batch_plans:
            # Dauer der Batch (maximale Endzeit in dieser Batch)
            if not plans:
                continue
            batch_T = max(path[-1][1] for path in plans.values())

            # 1) Für alle Qubits, die in dieser Batch aktiv sind: Plan mit Zeitversatz einfügen
            for q in qubits:
                qid = q.id
                last_coord, last_t = timelines[qid][-1]

                if qid in plans:
                    # Zeiten dieser Batch um t_offset verschieben
                    shifted = [(coord, t + t_offset) for (coord, t) in plans[qid]]

                    # Sicherstellen, dass die Timeline bis zum Batch-Start gefüllt ist
                    if last_t < t_offset:
                        # warte am Platz bis t_offset
                        for tt in range(last_t + 1, t_offset + 1):
                            timelines[qid].append((last_coord, tt))
                        last_coord, last_t = timelines[qid][-1]

                    # Doppelten Startknoten vermeiden, falls identisch
                    if timelines[qid][-1] == shifted[0]:
                        timelines[qid].extend(shifted[1:])
                    else:
                        timelines[qid].extend(shifted)

                    current_pos[qid] = shifted[-1][0]
                else:
                    # 2) Nicht beteiligte Qubits warten über die Batch-Dauer
                    # Wir füllen jeden Zeitschritt bis t_offset + batch_T
                    target_t = t_offset + batch_T
                    if last_t < target_t:
                        for tt in range(last_t + 1, target_t + 1):
                            timelines[qid].append((last_coord, tt))
                        current_pos[qid] = last_coord

            t_offset += batch_T

        return timelines


if __name__ == "__main__":
    G = NetworkBuilder.build_network()

    qubits: List[Qubit] = [
        Qubit(0, (1, -2)),
        Qubit(1, (1,  0)),
        Qubit(2, (-1, 0)),
        Qubit(3, (3, -2)),
        Qubit(4, (0, -3)),
        Qubit(5, (2,  1))
    ]

    pairs: List[Tuple[Qubit, Qubit]] = [
        (qubits[0], qubits[1]),  
        (qubits[0], qubits[2]),
        (qubits[2], qubits[3]),  
        (qubits[4], qubits[5]),  
        (qubits[0], qubits[2]),
        (qubits[1], qubits[4]),
        (qubits[3], qubits[5]),
        (qubits[0], qubits[5]),
        (qubits[1], qubits[3]),
        (qubits[4], qubits[5]),
    ]

    planner = RoutingPlanner()
    routing_plan = planner.plan_in_batches(G, qubits, pairs)

    animate_mapf(G, routing_plan)

