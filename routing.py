from __future__ import annotations

from copy import deepcopy 
from dataclasses import dataclass
from heapq import heappush, heappop
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx
import random

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
P_SUCCESS = 1


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
        G: nx.Graph,
        q0: Coord,
        q1: Coord,
        reserved_meetings: Optional[Set[Coord]] = None,
        occupied_nodes: Optional[Set[Coord]] = None
    ) -> List[Coord]:
        if reserved_meetings is None:
            reserved_meetings = set()
        if occupied_nodes is None:
            occupied_nodes = set()

        q0_bfs_dist = nx.single_source_shortest_path_length(G, q0)
        q1_bfs_dist = nx.single_source_shortest_path_length(G, q1)
        all_in_nodes = [
            n for n in G
            if G.nodes[n]["type"] == "IN"
            and n in q0_bfs_dist and n in q1_bfs_dist
        ]

        sorted_candidates = sorted(
            all_in_nodes,
            key=lambda n: (max(q0_bfs_dist[n], q1_bfs_dist[n]),
                           q0_bfs_dist[n] + q1_bfs_dist[n],
                           abs(q0_bfs_dist[n] - q1_bfs_dist[n]),
                           n)
        )

        return [c for c in sorted_candidates if c not in reserved_meetings and c not in occupied_nodes]


    @staticmethod
    def pairwise_mapf(
        G: nx.Graph,
        pairs: List[Tuple[Qubit, Qubit]],
        fixed_meetings: Optional[Dict[frozenset, Coord]] = None,
        occupied_nodes: Optional[Set[Coord]] = None
    ) -> Dict[int, List[TimedNode]]:
        """
        Plan a batch of pairs. If synchronization at the meeting node fails,
        the algorithm automatically falls back to alternative meeting candidates.

        Parameters:
        - pairs: list of qubit pairs to be routed.
        - fixed_meetings: {pair -> node} for pairs that already have a fixed meeting node.
        - occupied_nodes: nodes that must not be chosen as meeting candidates in this attempt.

        Returns:
        - A plan mapping qubit_id -> path (list of (node, time) steps).
        """
        fixed_meetings = fixed_meetings or {}
        occupied_nodes = occupied_nodes or set()

        # 1) Determine ordering of pairs based on their best candidate meeting node
        prelim_infos = []
        tmp_reserved: Set[Coord] = set()  # temporary reserved meetings, only for ordering
        for i, (qa, qb) in enumerate(pairs):
            pair_key = frozenset({qa.id, qb.id})
            if pair_key in fixed_meetings:
                m0 = fixed_meetings[pair_key]
            else:
                # best candidate list based on current start positions (avoid forbidden nodes)
                cands = RoutingPlanner.meeting_candidates(G, qa.pos, qb.pos, tmp_reserved, occupied_nodes)
                if not cands:
                    raise RuntimeError(f"No meeting candidates available for pair {qa.id}-{qb.id}")
                m0 = cands[0]

            # Distances to best candidate (for ranking)
            da = nx.single_source_shortest_path_length(G, qa.pos)
            db = nx.single_source_shortest_path_length(G, qb.pos)
            prelim_infos.append({
                "pair_id": i,
                "qa": qa,
                "qb": qb,
                "best_m": m0,
                "maxd": max(da[m0], db[m0]),   # maximum distance
                "sumd": da[m0] + db[m0],      # sum of distances
            })
            # Reserve this node temporarily for ordering purposes
            tmp_reserved.add(m0)

        # Sort pairs so the hardest ones (largest distance) are planned first
        plan_order = sorted(prelim_infos, key=lambda p: (p["maxd"], p["sumd"]), reverse=True)

        # 2) Actual planning with reservations
        res = Reservations(G)

        moving_starts: Set[Coord] = set()
        for qa, qb in pairs:
            moving_starts.add(qa.pos)
            moving_starts.add(qb.pos)

        # Nodes occupied at t=0 by qubits outside this attempt
        static_nodes: Set[Coord] = set(occupied_nodes) - moving_starts

        # For the duration of this attempt, treat those nodes as unavailable.
        # We fully block capacity to avoid sharing with static agents.
        for node in static_nodes:
            cap = res.node_capacity(node)
            # Reserve across the full search horizon so A* never steps on them.
            for t in range(0, MAX_TIME + 1):
                res.node_caps[node][t] = min(res.node_caps[node][t] + 1, cap)

        plans: Dict[int, List[TimedNode]] = {}
        reserved_meetings: Set[Coord] = set()  # permanently reserved meeting nodes

        for item in plan_order:
            qa: Qubit = item["qa"]
            qb: Qubit = item["qb"]
            pair_key = frozenset({qa.id, qb.id})

            # Candidate list for this pair
            if pair_key in fixed_meetings:
                candidates = [fixed_meetings[pair_key]]  # force fixed meeting node
            else:
                candidates = RoutingPlanner.meeting_candidates(
                    G, qa.pos, qb.pos, reserved_meetings, occupied_nodes
                )

            if not candidates:
                raise RuntimeError(f"No meeting candidates left for pair {qa.id}-{qb.id}")

            placed = False
            for meeting in candidates:
                # (a) Suche beider Pfade, aber committe noch nicht endgültig in 'res'
                res_base = deepcopy(res)

                # A zuerst, damit B Kollisionen mit A meidet
                res_for_b = deepcopy(res_base)
                a_plan = AStar.search(G, qa.pos, meeting, res_for_b)
                if a_plan is None:
                    continue
                # "soft commit" nur in res_for_b, damit B darauf Rücksicht nimmt
                Reservations.commit(res_for_b, a_plan)  # nutzt staticmethod? -> siehe unten
                # Falls Reservations.commit nicht static ist, ersetze mit:
                # for node in [a_plan[0]] + [p for p in a_plan[1:]]:
                #   (wir brauchen hier wirklich commit; einfacher: Hilfsres wie unten)

                b_plan = AStar.search(G, qb.pos, meeting, res_for_b)
                if b_plan is None:
                    continue

                # (b) Synchronisation: früher Ankommende warten am PRE-IN, nicht im IN
                Ta, Tb = a_plan[-1][1], b_plan[-1][1]
                Tm = max(Ta, Tb)

                a_plan2 = RoutingPlanner._retime_to_pre_in_wait(a_plan, meeting, Tm)
                if a_plan2 is None:
                    continue
                b_plan2 = RoutingPlanner._retime_to_pre_in_wait(b_plan, meeting, Tm)
                if b_plan2 is None:
                    continue

                # (c) Validieren & endgültig committen gegen die aktuelle globale 'res'
                res_commit = deepcopy(res)
                if not RoutingPlanner._try_commit(res_commit, a_plan2):
                    continue
                if not RoutingPlanner._try_commit(res_commit, b_plan2):
                    continue

                # (d) Erfolg: übernehme commit, reserviere Meeting, speichere Pläne
                res = res_commit
                reserved_meetings.add(meeting)
                plans[qa.id] = a_plan2
                plans[qb.id] = b_plan2
                placed = True
                break


            if not placed:
                raise RuntimeError(f"No feasible meeting (with sync) for pair {qa.id}-{qb.id}")

        return plans
    
    @staticmethod
    def try_until_success(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float = P_SUCCESS
    ) -> Dict[int, List[TimedNode]]:
        """
        Synchronisierte Attempt-Logik pro Batch mit PRE-IN-Konfliktvermeidung:

        1) Alle Qubits der Attempt-Batch laufen zunächst nur bis PRE-IN und warten
        global bis alle am PRE-IN sind (T_pre).
        2) Danach: Erfolg -> Schritt PRE-IN -> Meeting bei T_pre+1; Misserfolg -> Reset-Sprung
        zurück zur Batch-Startposition bei T_pre+1.
        3) Globaler Meeting-Sync (durch simultanes T_pre+1 gegeben).
        4) Wenn mehrere Paare denselben PRE-IN zur selben Zeit T_pre belegen würden, wird
        in diesem Microbatch nur EIN Paar pro PRE-IN zugelassen; die übrigen Paare
        werden für diesen Attempt zurückgestellt (bleiben sichtbar am Batch-Start).
        5) Wiederholen, bis alle Paare des Batches komplett erfolgreich sind.
        6) Gemeinsamer Egress-Microbatch mit Fallback auf nächstgelegene freie SNs.
        """

        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}

        i = 0
        n = len(pairs)
        batch_plans: List[Dict[int, List[TimedNode]]] = []

        while i < n:
            used_in_batch: Set[int] = set()
            batch_pairs: List[Tuple[Qubit, Qubit]] = []

            # disjunkte Paare sammeln
            j = i
            while j < n:
                qa, qb = pairs[j]
                a_id, b_id = qa.id, qb.id
                if a_id in used_in_batch or b_id in used_in_batch:
                    break
                qa_now = Qubit(a_id, current_pos[a_id])
                qb_now = Qubit(b_id, current_pos[b_id])
                batch_pairs.append((qa_now, qb_now))
                used_in_batch.update([a_id, b_id])
                j += 1

            if not batch_pairs:
                qa, qb = pairs[j]
                qa_now = Qubit(qa.id, current_pos[qa.id])
                qb_now = Qubit(qb.id, current_pos[qb.id])
                batch_pairs = [(qa_now, qb_now)]
                used_in_batch.update([qa.id, qb.id])
                j += 1

            batch_start_pos: Dict[int, Coord] = {qid: current_pos[qid] for qid in used_in_batch}
            id_pairs: List[Tuple[int, int]] = [(qa.id, qb.id) for (qa, qb) in batch_pairs]
            success: Dict[int, bool] = dict.fromkeys(used_in_batch, False)
            fixed_meetings: Dict[frozenset, Coord] = {}

            # Puffer (für Egress)
            last_meeting_of_pair: Dict[frozenset, Coord] = {}
            last_paths_of_pair: Dict[frozenset, Dict[int, List[TimedNode]]] = {}

            def plan_egress_for_whole_batch() -> None:
                """Gemeinsamer Egress-Microbatch für alle Paare des Batches.
                Fallback: falls kein PRE-IN ermittelbar oder belegt, wähle das
                nächstgelegene freie SN via BFS vom Meeting aus."""
                return_starts: Dict[int, Coord] = {}
                raw_targets: Dict[int, Coord] = {}
                approach_paths: Dict[int, List[TimedNode]] = {}
                meet_of_qid: Dict[int, Coord] = {}

                # Entry-SNs aus letzter Anmarschplanung ableiten
                for a_id, b_id in id_pairs:
                    pkey = frozenset({a_id, b_id})
                    if pkey not in last_meeting_of_pair or pkey not in last_paths_of_pair:
                        continue
                    mnode = last_meeting_of_pair[pkey]
                    a_path = last_paths_of_pair[pkey][a_id]
                    b_path = last_paths_of_pair[pkey][b_id]

                    approach_paths[a_id] = a_path
                    approach_paths[b_id] = b_path
                    meet_of_qid[a_id] = mnode
                    meet_of_qid[b_id] = mnode

                    a_entry_sn = RoutingPlanner._entry_sn_from_path(a_path, mnode)
                    b_entry_sn = RoutingPlanner._entry_sn_from_path(b_path, mnode)

                    # Start des Egress ist immer das Meeting
                    return_starts[a_id] = mnode
                    return_starts[b_id] = mnode

                    # Primärziel ist der PRE-IN; wenn None -> später Fallback
                    if a_entry_sn is not None:
                        raw_targets[a_id] = a_entry_sn
                    if b_entry_sn is not None:
                        raw_targets[b_id] = b_entry_sn

                if not return_starts:
                    return  # nichts zu egressen

                # aktuell belegte Knoten
                occupied_now: Set[Coord] = {current_pos[q.id] for q in qubits}
                egress_qids: Set[int] = set(return_starts.keys())

                # blockiere beim Routing alle Knoten von Qubits, die NICHT egressen
                blocked_nodes_for_paths: Set[Coord] = {
                    pos for qid, pos in current_pos.items() if qid not in egress_qids
                }

                # Ziel-Deduplizierung
                claimed: Set[Coord] = set()
                return_targets: Dict[int, Coord] = {}

                def first_alternative_sn(qid: int, avoid: Set[Coord]) -> Optional[Coord]:
                    """Gehe den Anmarschpfad rückwärts; wähle das erste SN, das frei ist."""
                    path = approach_paths[qid]
                    meet = path[-1][0]
                    first_meet_idx = None
                    for ii, (c, _) in enumerate(path):
                        if c == meet:
                            first_meet_idx = ii
                            break
                    if first_meet_idx is None:
                        return None
                    for ii in range(first_meet_idx - 1, -1, -1):
                        node = path[ii][0]
                        if G.nodes[node]["type"] == "SN" and node not in avoid:
                            return node
                    return None

                def cheb(a: Coord, b: Coord) -> int:
                    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

                # schwierigere zuerst
                order = sorted(
                    return_starts.keys(),
                    key=lambda qid: cheb(return_starts[qid], raw_targets.get(qid, return_starts[qid])),
                    reverse=True
                )

                hard_avoid_base: Set[Coord] = set(occupied_now)

                for qid in order:
                    avoid = claimed | hard_avoid_base
                    tgt = raw_targets.get(qid)

                    pick = None
                    # 1) bevorzugt: PRE-IN (falls vorhanden & frei)
                    if tgt is not None and tgt not in avoid:
                        pick = tgt
                    else:
                        # 2) Alternative entlang des Anmarschpfads
                        alt = first_alternative_sn(qid, avoid)
                        if alt is not None:
                            pick = alt
                        else:
                            # 3) Fallback: nächstgelegenes freies SN vom Meeting aus (BFS)
                            meet = meet_of_qid[qid]
                            pick = RoutingPlanner._nearest_free_sn(G, meet, avoid)

                    if pick is not None:
                        return_targets[qid] = pick
                        claimed.add(pick)
                    # Falls None → dieses Qubit egressed nicht in diesem Microbatch (selten)

                # nur Qubits mit Ziel egressen lassen
                starts_exec = {qid: s for qid, s in return_starts.items() if qid in return_targets}
                targets_exec = {qid: return_targets[qid] for qid in starts_exec}
                if not starts_exec:
                    return

                return_plans = RoutingPlanner._mapf_to_targets(
                    G, starts_exec, targets_exec, blocked_nodes=blocked_nodes_for_paths
                )
                batch_plans.append(return_plans)

                # Positionen aktualisieren
                for qid, rpath in return_plans.items():
                    current_pos[qid] = rpath[-1][0]

            # Attempts, bis alle Paare des Batches fertig sind
            while True:
                # fertig?
                if all(success[a] and success[b] for (a, b) in id_pairs):
                    plan_egress_for_whole_batch()
                    break

                # pending Paare
                pending_pairs_ids = [(a, b) for (a, b) in id_pairs if not (success[a] and success[b])]

                # Attempt-Paare aus aktuellen Positionen
                attempt_pairs: List[Tuple[Qubit, Qubit]] = []
                for a_id, b_id in pending_pairs_ids:
                    qa_now = Qubit(a_id, current_pos[a_id])
                    qb_now = Qubit(b_id, current_pos[b_id])
                    attempt_pairs.append((qa_now, qb_now))

                occupied_now_for_meet: Set[Coord] = {current_pos[q.id] for q in qubits}

                # Planung bis zum Meeting (liefert kollisionsfreie Anmarschpfade)
                attempt_full = RoutingPlanner.pairwise_mapf(
                    G,
                    attempt_pairs,
                    fixed_meetings=fixed_meetings,
                    occupied_nodes=occupied_now_for_meet,
                )

                # Für die Timeline einen Microbatch vorbereiten (wir füllen ihn gleich)
                batch_plans.append({})
                micro_idx = len(batch_plans) - 1

                # Meeting je pending Paar + Pufferung
                meeting_of_pair: Dict[frozenset, Coord] = {}
                for a_id, b_id in pending_pairs_ids:
                    pkey = frozenset({a_id, b_id})
                    meet = attempt_full[a_id][-1][0]
                    meeting_of_pair[pkey] = meet
                    last_meeting_of_pair[pkey] = meet
                    last_paths_of_pair[pkey] = {
                        a_id: attempt_full[a_id],
                        b_id: attempt_full[b_id],
                    }

                # -------- Phase 1: PRE-IN global synchronisieren --------
                pre_in_arrival_times: Dict[int, int] = {}
                for a_id, b_id in pending_pairs_ids:
                    pkey = frozenset({a_id, b_id})
                    meet = meeting_of_pair[pkey]
                    for qid in (a_id, b_id):
                        t_pre = RoutingPlanner._first_pre_in_time(attempt_full[qid], meet)
                        if t_pre is None:
                            raise RuntimeError("Fehler: PRE-IN nicht gefunden.")
                        pre_in_arrival_times[qid] = t_pre

                T_pre = max(pre_in_arrival_times.values())

                # PRE-IN-Wartepfade bis T_pre
                pre_in_wait_paths: Dict[int, List[TimedNode]] = {}
                for a_id, b_id in pending_pairs_ids:
                    pkey = frozenset({a_id, b_id})
                    meet = meeting_of_pair[pkey]
                    for qid in (a_id, b_id):
                        cut = RoutingPlanner._retime_until_pre_in_wait(attempt_full[qid], meet, T_pre)
                        if cut is None:
                            raise RuntimeError("Fehler beim PRE-IN-Retiming.")
                        pre_in_wait_paths[qid] = cut

                # -------- NEU: PRE-IN-Konflikte im Microbatch verhindern --------
                # Map: qid -> PRE-IN-Knoten (aus den pre_in_wait_paths extrahieren)
                qid_pre_in: Dict[int, Coord] = {}
                for a_id, b_id in pending_pairs_ids:
                    pkey = frozenset({a_id, b_id})
                    meet = meeting_of_pair[pkey]
                    for qid in (a_id, b_id):
                        path_cut = pre_in_wait_paths[qid]
                        # letzter Knoten in path_cut ist PRE-IN @ T_pre
                        qid_pre_in[qid] = path_cut[-1][0]

                # Baue Konfliktgruppen: PRE-IN-Knoten -> Liste der Qubits, die dort @ T_pre stehen wollen
                pre_in_groups: Dict[Coord, List[int]] = defaultdict(list)
                for qid, pin in qid_pre_in.items():
                    pre_in_groups[pin].append(qid)

                # Wähle pro PRE-IN höchstens EIN Paar (2 qids) aus; die restlichen Paare werden zurückgestellt
                allowed_qids: Set[int] = set()
                blocked_qids: Set[int] = set()

                # Hilfsmap: qid -> zugehörige Paar-IDs (a_id,b_id)
                qid_to_pair: Dict[int, Tuple[int, int]] = {}
                for a_id, b_id in pending_pairs_ids:
                    qid_to_pair[a_id] = (a_id, b_id)
                    qid_to_pair[b_id] = (a_id, b_id)

                # Priorität: Reihenfolge in pending_pairs_ids beibehalten (stabil)
                pair_priority = { (a,b): idx for idx, (a,b) in enumerate(pending_pairs_ids) }

                for pre_in, qids in pre_in_groups.items():
                    if len(qids) == 1:
                        allowed_qids.add(qids[0])
                        continue

                    # Mehrere Qubits wollen denselben PRE-IN: wähle das Paar mit höchster Priorität
                    # und blocke die anderen Paare komplett.
                    cand_pairs = set(qid_to_pair[q] for q in qids)
                    winner_pair = min(cand_pairs, key=lambda p: pair_priority[p])
                    wa, wb = winner_pair
                    allowed_qids.update([wa, wb])

                    # Alle Qubits aus anderen Paaren werden blockiert
                    for p in cand_pairs:
                        if p == winner_pair:
                            continue
                        ba, bb = p
                        blocked_qids.update([ba, bb])

                # Falls ein Qubit weder explizit allowed noch blocked ist (kein Konflikt),
                # erlaube es ganz normal:
                for a_id, b_id in pending_pairs_ids:
                    if a_id not in blocked_qids:
                        allowed_qids.add(a_id)
                    if b_id not in blocked_qids:
                        allowed_qids.add(b_id)
                # ---------------------------------------------------------------

                # -------- Phase 2: Reset/Entry + Meeting-Sync --------
                # Entscheidung pro Qubit (nur relevant für allowed_qids)
                decided_success: Dict[int, bool] = {}
                for a_id, b_id in pending_pairs_ids:
                    # nur für Qubits, die noch nicht success sind
                    if not success[a_id]:
                        decided_success[a_id] = (random.random() < p_success)
                    else:
                        decided_success[a_id] = True
                    if not success[b_id]:
                        decided_success[b_id] = (random.random() < p_success)
                    else:
                        decided_success[b_id] = True

                T_meet = T_pre + 1
                micro_paths: Dict[int, List[TimedNode]] = {}

                for a_id, b_id in pending_pairs_ids:
                    pkey = frozenset({a_id, b_id})
                    meet = meeting_of_pair[pkey]

                    for qid in (a_id, b_id):
                        if qid in blocked_qids:
                            # Dieses Qubit nimmt NICHT am Attempt teil:
                            # Es bleibt bis T_pre am Batch-Start und springt (sichtbar) bei T_pre+1 wieder dorthin.
                            base = [(batch_start_pos[qid], 0)]
                            cur_t = 0
                            while cur_t < T_pre:
                                base.append((batch_start_pos[qid], cur_t + 1))
                                cur_t += 1
                            # Sichtbarer „Reset“-Schritt bei T_pre+1
                            base.append((batch_start_pos[qid], T_pre + 1))
                            micro_paths[qid] = base
                            continue

                        # Normale Teilnahme (wie bisher)
                        base = pre_in_wait_paths[qid].copy()   # endet bei PRE-IN @ T_pre
                        assert base[-1][1] == T_pre

                        if decided_success[qid]:
                            # Schritt PRE-IN -> Meeting bei T_pre+1
                            base.append((meet, T_pre + 1))
                        else:
                            # Reset-Sprung zur Batch-Startposition bei T_pre+1
                            base.append((batch_start_pos[qid], T_pre + 1))

                        # Sync bis T_meet (ist bereits T_pre+1)
                        cur_t = base[-1][1]
                        while cur_t < T_meet:
                            base.append((base[-1][0], cur_t + 1))
                            cur_t += 1

                        micro_paths[qid] = base

                # Microbatch einsetzen
                batch_plans[micro_idx] = micro_paths

                # -------- Status-Update / Fixierung / Positionen --------
                for a_id, b_id in pending_pairs_ids:
                    pkey = frozenset({a_id, b_id})
                    meet = meeting_of_pair[pkey]

                    # a
                    if a_id in blocked_qids:
                        # Nicht teilgenommen → bleibt am Batch-Start
                        current_pos[a_id] = batch_start_pos[a_id]
                    else:
                        if not success[a_id]:
                            if decided_success[a_id]:
                                success[a_id] = True
                                current_pos[a_id] = meet
                            else:
                                current_pos[a_id] = batch_start_pos[a_id]
                        else:
                            current_pos[a_id] = meet  # bleibt beim Meeting (wartet)

                    # b
                    if b_id in blocked_qids:
                        current_pos[b_id] = batch_start_pos[b_id]
                    else:
                        if not success[b_id]:
                            if decided_success[b_id]:
                                success[b_id] = True
                                current_pos[b_id] = meet
                            else:
                                current_pos[b_id] = batch_start_pos[b_id]
                        else:
                            current_pos[b_id] = meet

                    # sobald mind. ein Qubit der Paarung erfolgreich war → Meeting fixieren
                    if pkey not in fixed_meetings and ((a_id not in blocked_qids and decided_success[a_id]) or
                                                    (b_id not in blocked_qids and decided_success[b_id]) or
                                                    success[a_id] or success[b_id]):
                        if success[a_id] or success[b_id]:
                            fixed_meetings[pkey] = meet

            # nächster Batch
            i = j

        return RoutingPlanner.stitch_batches(qubits, batch_plans)



        

    @staticmethod
    def interleaved_routing(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: int = P_SUCCESS
    ) -> Dict[int, List[TimedNode]]:
        """
        Interleaved strategy with co-scheduling and hard meeting locks:
        - Pending pairs (retries) + new disjoint pairs run in the same time step.
        - Fixed meeting nodes are globally exclusive: only the associated pair may use them,
        until the pair is completely successful.
        - Once a pair is completed, its meeting node is permanently locked
        (no other pair may choose it as a meeting later).
        - In the same batch, a fixed meeting node may only appear once.
        - Already occupied nodes (qubit waiting there) + permanently locked meeting nodes
        are excluded from meeting candidate selection.
        """

        # Current position per qubit
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}

        # For later stitching
        batch_plans: List[Dict[int, List[TimedNode]]] = []

        # Pairs only as (id,id)
        all_pairs_ids: List[Tuple[int, int]] = [(qa.id, qb.id) for (qa, qb) in pairs]

        # pair_idx -> {a_id: bool, b_id: bool} (whether the respective qubit was already successful)
        pending_success: Dict[int, Dict[int, bool]] = {}

        # Once at least one qubit of a pair was successful, we fix its meeting node
        fixed_meetings: Dict[frozenset, Coord] = {}

        # Completely completed pairs
        done_pairs: Set[int] = set()

        # Permanently locked meeting nodes (once a pair was completed there)
        finalized_meeting_nodes: Set[Coord] = set()

        # Next pair not yet considered (for linear progression)
        next_idx = 0

        # Local helper function: select maximum number of disjoint pair INDICES in given order
        def select_disjoint_indices(index_list: List[int]) -> List[int]:
            used_q: Set[int] = set()
            picked: List[int] = []
            for pidx in index_list:
                a_id, b_id = all_pairs_ids[pidx]
                if a_id not in used_q and b_id not in used_q:
                    picked.append(pidx)
                    used_q.add(a_id); used_q.add(b_id)
            return picked

        while len(done_pairs) < len(all_pairs_ids):
            # Qubits currently blocked by incomplete pending pairs
            blocked_qids: Set[int] = set()
            for pidx, succ_map in pending_success.items():
                if pidx in done_pairs:
                    continue
                a_id, b_id = all_pairs_ids[pidx]
                if not (succ_map.get(a_id, False) and succ_map.get(b_id, False)):
                    blocked_qids.add(a_id); blocked_qids.add(b_id)

            # ---------- Form batch: retry core + new independent pairs ----------

            # (1) Retry core: pending pairs in order that are not yet finished
            pending_ordered: List[int] = [
                pidx for pidx in range(len(all_pairs_ids))
                if pidx in pending_success and pidx not in done_pairs
            ]
            retry_core_idx: List[int] = select_disjoint_indices(pending_ordered)

            # (2) New candidates: never attempted (not in pending_success), not done.
            #     These are considered independently of "blocked_qids", but must be disjoint
            #     from the already chosen retry core.
            new_candidates_idx: List[int] = []
            scan_idx = next_idx
            while scan_idx < len(all_pairs_ids):
                if scan_idx in done_pairs or scan_idx in pending_success:
                    scan_idx += 1
                    continue
                new_candidates_idx.append(scan_idx)
                scan_idx += 1

            # (3) Augment batch: first retry core, then add new disjoint pairs
            batch_idx_list: List[int] = list(retry_core_idx)
            used_qids_in_batch: Set[int] = set()
            for pidx in batch_idx_list:
                a_id, b_id = all_pairs_ids[pidx]
                used_qids_in_batch.add(a_id); used_qids_in_batch.add(b_id)

            for pidx in new_candidates_idx:
                a_id, b_id = all_pairs_ids[pidx]
                if a_id not in used_qids_in_batch and b_id not in used_qids_in_batch:
                    batch_idx_list.append(pidx)
                    used_qids_in_batch.add(a_id); used_qids_in_batch.add(b_id)

            # (4) Fallback: If nothing was chosen, take the earliest pending or new pair solo
            if not batch_idx_list:
                fallback = None
                # first pending
                for pidx in pending_ordered:
                    fallback = pidx
                    break
                # then new pair
                if fallback is None:
                    for pidx in range(len(all_pairs_ids)):
                        if pidx not in done_pairs and pidx not in pending_success:
                            fallback = pidx
                            break
                if fallback is None:
                    # nothing left (should hardly occur due to while condition)
                    break
                batch_idx_list = [fallback]

            # ---------- Deduplicate fixed meeting nodes in the batch ----------
            # Multiple pairs in the same batch must NOT enforce the same fixed meeting node.
            fixed_node_to_owner_in_batch: Dict[Coord, int] = {}
            pruned_batch: List[int] = []
            for pidx in batch_idx_list:
                a_id, b_id = all_pairs_ids[pidx]
                pkey = frozenset({a_id, b_id})
                if pkey in fixed_meetings:
                    node = fixed_meetings[pkey]
                    if node in fixed_node_to_owner_in_batch:
                        # Collision -> defer this pair
                        continue
                    fixed_node_to_owner_in_batch[node] = pidx
                pruned_batch.append(pidx)
            batch_idx_list = pruned_batch

            # Progress pointer only for newly included (never attempted) pairs
            advanced_new = [pidx for pidx in batch_idx_list if pidx not in pending_success]
            if advanced_new:
                next_idx = max(next_idx, max(advanced_new) + 1)

            # ---------- Planning of the batch ----------

            # Batch start positions (for reset in case of failure)
            used_qids: Set[int] = set()
            for pidx in batch_idx_list:
                a_id, b_id = all_pairs_ids[pidx]
                used_qids.add(a_id); used_qids.add(b_id)
            batch_start_pos: Dict[int, Coord] = {qid: current_pos[qid] for qid in used_qids}

            # Qubit objects for the batch (with current positions)
            batch_pairs: List[Tuple[Qubit, Qubit]] = []
            for pidx in batch_idx_list:
                a_id, b_id = all_pairs_ids[pidx]
                qa_now = Qubit(a_id, current_pos[a_id])
                qb_now = Qubit(b_id, current_pos[b_id])
                batch_pairs.append((qa_now, qb_now))

            # ---------------- Global + permanent locks for meeting nodes ----------------
            # (a) Exclusively locked meeting nodes from already fixed meetings:
            exclusive_nodes: Set[Coord] = set(fixed_meetings.values())

            # (b) Which fixed nodes are used in THIS batch by their owner pairs?
            allowed_nodes_this_batch: Set[Coord] = set()
            for pidx in batch_idx_list:
                a_id, b_id = all_pairs_ids[pidx]
                pkey = frozenset({a_id, b_id})
                if pkey in fixed_meetings:
                    allowed_nodes_this_batch.add(fixed_meetings[pkey])

            # (c) Already occupied nodes from previous batches + permanently locked meeting nodes
            occupied_now: Set[Coord] = {current_pos[q.id] for q in qubits}
            # Permanent locks (completed pairs)
            occupied_now |= finalized_meeting_nodes
            # Exclusive locks (fixed meetings), except when the owner is in the current batch
            occupied_now |= (exclusive_nodes - allowed_nodes_this_batch)

            # Fixed meetings that are allowed in this batch (after de-dup)
            fm_for_this_attempt: Dict[frozenset, Coord] = {}
            for pidx in batch_idx_list:
                a_id, b_id = all_pairs_ids[pidx]
                pkey = frozenset({a_id, b_id})
                if pkey in fixed_meetings:
                    fm_for_this_attempt[pkey] = fixed_meetings[pkey]

            # Plan (MAPF) for the entire batch
            attempt_plans = RoutingPlanner.pairwise_mapf(
                G,
                batch_pairs,
                fixed_meetings=fm_for_this_attempt,
                occupied_nodes=occupied_now,  # <- prevents meetings on occupied/locked INs
            )
            batch_plans.append(attempt_plans)

            # Determine meeting nodes per pair
            meeting_of_pair: Dict[frozenset, Coord] = {}
            for pidx in batch_idx_list:
                a_id, b_id = all_pairs_ids[pidx]
                meet = attempt_plans[a_id][-1][0]
                meeting_of_pair[frozenset({a_id, b_id})] = meet

            # ---------- Simulate success/failure and update status ----------
            for pidx in batch_idx_list:
                a_id, b_id = all_pairs_ids[pidx]
                pkey = frozenset({a_id, b_id})
                mnode = meeting_of_pair[pkey]

                # Initialize pending map if first time
                if pidx not in pending_success:
                    pending_success[pidx] = {a_id: False, b_id: False}

                # a
                if not pending_success[pidx][a_id]:
                    a_success = (random.random() < p_success)
                    if a_success:
                        pending_success[pidx][a_id] = True
                        current_pos[a_id] = mnode
                    else:
                        current_pos[a_id] = batch_start_pos[a_id]
                else:
                    current_pos[a_id] = mnode

                # b
                if not pending_success[pidx][b_id]:
                    b_success = (random.random() < p_success)
                    if b_success:
                        pending_success[pidx][b_id] = True
                        current_pos[b_id] = mnode
                    else:
                        current_pos[b_id] = batch_start_pos[b_id]
                else:
                    current_pos[b_id] = mnode

                # Fix meeting as soon as at least one qubit was successful
                if pkey not in fixed_meetings and (pending_success[pidx][a_id] or pending_success[pidx][b_id]):
                    fixed_meetings[pkey] = mnode

                # Pair finished?
                if pending_success[pidx][a_id] and pending_success[pidx][b_id]:
                    done_pairs.add(pidx)
                    # Permanently lock: this meeting node may never serve as meeting again
                    finalized_meeting_nodes.add(mnode)

        # Combine all batches into global timelines
        return RoutingPlanner.stitch_batches(qubits, batch_plans)



    # Helpers
    #--------------------------------------------
    @staticmethod
    def _nearest_free_sn(
        G: nx.Graph,
        source: Coord,
        avoid: Set[Coord],
    ) -> Optional[Coord]:
        """Finde das nächstgelegene SN (BFS), das nicht in 'avoid' liegt."""
        from collections import deque
        q = deque([source])
        seen = {source}
        while q:
            u = q.popleft()
            if G.nodes[u]["type"] == "SN" and u not in avoid:
                return u
            for v in G.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return None
    
    @staticmethod
    def _retime_until_pre_in_wait(
        path: List[TimedNode],
        meeting: Coord,
        sync_time: int,
    ) -> Optional[List[TimedNode]]:
        """
        Schneidet 'path' so ab, dass er am PRE-IN endet und dort bis 'sync_time' wartet.
        Gibt einen Pfad zurück, der mit PRE-IN @ sync_time endet (Meeting wird NICHT betreten).
        """
        if not path:
            return None

        # Index der ersten Ankunft am Meeting
        first_meet_idx = None
        for i, (c, _) in enumerate(path):
            if c == meeting:
                first_meet_idx = i
                break
        if first_meet_idx is None:
            return None  # path endet nicht im meeting (sollte nicht passieren)

        if first_meet_idx == 0:
            # Start bereits im Meeting → PRE-IN existiert nicht; wir warten einfach im Start, bis sync_time
            start_node, start_t = path[0]
            new_path = [(start_node, start_t)]
            cur_t = start_t
            while cur_t < sync_time:
                new_path.append((start_node, cur_t + 1))
                cur_t += 1
            return new_path

        pre_in, t_pre_in = path[first_meet_idx - 1]
        new_path = path[: first_meet_idx]  # endet am PRE-IN (Zeit t_pre_in)

        # Warte am PRE-IN bis sync_time
        cur_t = t_pre_in
        while cur_t < sync_time:
            new_path.append((pre_in, cur_t + 1))
            cur_t += 1
        return new_path

    @staticmethod
    def _first_pre_in_time(path: List[TimedNode], meeting: Coord) -> Optional[int]:
        """Zeitstempel der ersten Ankunft am PRE-IN (Schritt vor der *ersten* Meeting-Ankunft)."""
        first_meet_idx = None
        for i, (c, t) in enumerate(path):
            if c == meeting:
                first_meet_idx = i
                break
        if first_meet_idx is None:
            return None
        if first_meet_idx == 0:
            return path[0][1]
        return path[first_meet_idx - 1][1]
    
    @staticmethod
    def _try_commit(res: Reservations, path: List[TimedNode]) -> bool:
        """Prüft und commitet 'path' Schritt für Schritt in 'res'.
        Gibt False zurück, falls Kapazitäten/Edge-Exklusivität verletzt würden."""
        if not path:
            return False
        # erster Knoten
        n0, t0 = path[0]
        if not res.can_occupy(n0, t0):
            return False
        res.occupy_node(n0, t0)

        for (u, tu), (v, tv) in zip(path[:-1], path[1:]):
            # Zielknoten verfügbar?
            if not res.can_occupy(v, tv):
                return False
            # Kante frei, falls Bewegung
            if u != v and not res.can_traverse(u, v, tu):
                return False
            res.occupy_node(v, tv)
            if u != v:
                res.traverse_edge(u, v, tu)
        return True

    @staticmethod
    def _retime_to_pre_in_wait(
        path: List[TimedNode],
        meeting: Coord,
        sync_time: int,
    ) -> Optional[List[TimedNode]]:
        """Lässt ein zu frühes Ankommen am PRE-IN warten und betritt den IN erst bei sync_time.
        Erwartet 'path' als gültigen Pfad, der im Meeting endet."""
        if not path:
            return None
        # Zeitpunkt der ersten Ankunft im Meeting
        first_meet_idx = None
        for i, (c, _) in enumerate(path):
            if c == meeting:
                first_meet_idx = i
                break
        if first_meet_idx is None:
            return None  # path endet nicht im meeting (sollte nicht passieren)

        # Falls Pfad schon bei sync_time ankommt (oder später): nichts zu tun
        Ta = path[-1][1]
        if Ta >= sync_time:
            return path

        # PRE-IN bestimmen (Schritt direkt vor erster Meeting-Ankunft)
        if first_meet_idx == 0:
            # Startet bereits im Meeting → kein PRE-IN existiert; fallback: am Meeting warten
            # (Verhalten wie bisher)
            ext = path.copy()
            while ext[-1][1] < sync_time:
                n, t = ext[-1]
                ext.append((n, t + 1))
            return ext

        pre_in = path[first_meet_idx - 1][0]
        t_pre_in = path[first_meet_idx - 1][1]

        # Neu zusammensetzen:
        #  - bis inkl. PRE-IN original lassen
        #  - ab t_pre_in+1 bis sync_time-1 am PRE-IN warten
        #  - in Schritt (sync_time-1 -> sync_time) in den Meeting-Knoten gehen
        new_path = path[: first_meet_idx]  # endet bei PRE-IN zur Zeit t_pre_in

        # am PRE-IN warten
        cur_t = t_pre_in
        while cur_t + 1 < sync_time:
            new_path.append((pre_in, cur_t + 1))
            cur_t += 1

        # finaler Schritt ins Meeting bei sync_time
        new_path.append((meeting, sync_time))
        return new_path

    @staticmethod
    def _entry_sn_from_path(path: List[TimedNode], meeting: Coord) -> Optional[Coord]:
        """
        Given a path that ends at `meeting`, return the node that was visited
        immediately before the *first* arrival at `meeting` (the SN they used
        to enter the IN). If the path starts at the meeting, return None.
        """
        first_meet_idx = None
        for i, (c, _) in enumerate(path):
            if c == meeting:
                first_meet_idx = i
                break
        if first_meet_idx is None or first_meet_idx == 0:
            return None
        return path[first_meet_idx - 1][0]

    @staticmethod
    def _mapf_to_targets(
        G: nx.Graph,
        starts: Dict[int, Coord],
        targets: Dict[int, Coord],
        blocked_nodes: Optional[Set[Coord]] = None,
    ) -> Dict[int, List[TimedNode]]:
        """
        Collision-free multi-agent routing from per-qubit start -> target.
        Uses the same A* and Reservations; no synchronization needed.
        """
        if not starts:
            return {}

        res = Reservations(G)
        blocked_nodes = blocked_nodes or set()

        # Ghost-Startknoten der noch nicht erfolgreichen Qubits
        # für die Dauer des Microbatches voll blockieren.
        for node in blocked_nodes:
            cap = res.node_capacity(node)
            for t in range(0, MAX_TIME + 1):
                res.node_caps[node][t] = cap
        plans: Dict[int, List[TimedNode]] = {}

        # Pre-occupy all start nodes at t=0 to reflect initial occupancy
        for qid, s in starts.items():
            res.occupy_node(s, 0)

        # Plan "harder first": by (Chebyshev) distance descending
        def cheb(a: Coord, b: Coord) -> int:
            return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

        order = sorted(starts.keys(), key=lambda q: cheb(starts[q], targets[q]), reverse=True)

        for qid in order:
            s, t = starts[qid], targets[qid]
            path = AStar.search(G, s, t, res)
            if path is None:
                raise RuntimeError(f"Return routing failed for qubit {qid} from {s} -> {t}")
            res.commit(path)
            plans[qid] = path

        return plans

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
        Visible, consistent resets:
        - If a qubit after batch b starts in batch b+1 at a different node (reset),
        we show in batch b the movement only up to the node *before the first entry into the IN* (PRE_IN).
        - From the next time step onward, it jumps within the same batch to the reset target (next batch start 'nxt')
        and **waits there** until the end of the batch. This makes the reset visible and the waiting time
        always occurs at the actual reset target.
        """

        if not batch_plans:
            return {q.id: [(q.pos, 0)] for q in qubits}

        # --- Preprocessing: start/end/duration per batch ---
        starts: List[Dict[int, Coord]] = []
        durations: List[int] = []
        for plans in batch_plans:
            if not plans:
                starts.append({})
                durations.append(0)
                continue
            b_starts: Dict[int, Coord] = {}
            for qid, path in plans.items():
                b_starts[qid] = path[0][0]
            starts.append(b_starts)
            durations.append(max(p[-1][1] for p in plans.values()))

        # --- Build timelines ---
        initial_pos = {q.id: q.pos for q in qubits}
        timelines: Dict[int, List[TimedNode]] = {q.id: [(initial_pos[q.id], 0)] for q in qubits}

        t_offset = 0
        for b, plans in enumerate(batch_plans):
            batch_T = durations[b]

            # Fill up to batch start
            for q in qubits:
                qid = q.id
                last_coord, last_t = timelines[qid][-1]
                if last_t < t_offset:
                    for tt in range(last_t + 1, t_offset + 1):
                        timelines[qid].append((last_coord, tt))

            if batch_T == 0:
                continue

            for q in qubits:
                qid = q.id
                last_coord, last_t = timelines[qid][-1]

                if qid not in plans:
                    # not involved: waits the entire batch
                    target_t = t_offset + batch_T
                    for tt in range(last_t + 1, target_t + 1):
                        timelines[qid].append((last_coord, tt))
                    continue

                local_path = plans[qid]
                shifted = [(c, t + t_offset) for (c, t) in local_path]
                plan_end = shifted[-1][0]

                if b + 1 >= len(batch_plans):
                    nxt = None
                else:
                    nxt = starts[b + 1].get(qid)

                # Reset if there is a start in the next batch and it ≠ current batch end
                is_reset = (nxt is not None) and (nxt != plan_end)

                if not is_reset:
                    # normal takeover
                    if timelines[qid][-1] == shifted[0]:
                        timelines[qid].extend(shifted[1:])
                    else:
                        timelines[qid].extend(shifted)
                else:
                    # 1) run until PRE_IN
                    # first occurrence of the IN
                    first_in_idx = None
                    for idx, (c, _) in enumerate(local_path):
                        if c == local_path[-1][0]:
                            first_in_idx = idx
                            break
                    if first_in_idx is None or first_in_idx == 0:
                        anchor_coord = local_path[-1][0]  # Fallback
                    else: 
                        # the step before is our PRE_IN
                        anchor_coord = local_path[first_in_idx - 1][0]

                    # index of the last occurrence of anchor in the local path
                    last_anchor_idx = None
                    for idx, (c, _) in enumerate(local_path):
                        if c == anchor_coord:
                            last_anchor_idx = idx
                    if last_anchor_idx is None:
                        part = shifted  # Fallback
                    else:
                        part_local = local_path[: last_anchor_idx + 1]
                        part = [(c, t + t_offset) for (c, t) in part_local]

                    if part:
                        # avoid duplicate start
                        if timelines[qid][-1] == part[0]:
                            timelines[qid].extend(part[1:])
                        else:
                            timelines[qid].extend(part)

                    # 2) from the next time step until batch end, wait at the **reset target 'nxt'**
                    #    (visible jump from PRE_IN -> nxt)
                    current_last_t = timelines[qid][-1][1]
                    target_t = t_offset + batch_T
                    if current_last_t < target_t:
                        for tt in range(current_last_t + 1, target_t + 1):
                            timelines[qid].append((nxt, tt))

            t_offset += batch_T

        return timelines


if __name__ == "__main__":
    G = NetworkBuilder.build_network()
    random.seed(42)

    qubits: List[Qubit] = [
        Qubit(0, (1, -2)),
        Qubit(1, (1,  0)),
        Qubit(2, (-1, 0)),
        Qubit(3, (3, -2)),
        Qubit(4, (0, -3)),
        Qubit(5, (2,  1))
    ]

    pairs: List[Tuple[Qubit, Qubit]] = [
        (qubits[0], qubits[1]),  #1
        (qubits[0], qubits[2]),  #2
        (qubits[2], qubits[3]),  #3
        (qubits[4], qubits[5]),  #3
        (qubits[0], qubits[2]),  #4
        (qubits[1], qubits[4]),  #4
        (qubits[3], qubits[5]),  #4
        (qubits[0], qubits[5]),  #5
        (qubits[1], qubits[3]),  #5
        (qubits[4], qubits[5]),  #6
    ]

    planner = RoutingPlanner()
    routing_plan = planner.try_until_success(G, qubits, pairs)
    
    #routing_plan = planner.interleaved_routing(G, qubits, pairs)

    max_time = max(path[-1][1] for path in routing_plan.values())
    print(f"Total number of time steps: {max_time}")

    for t in range(max_time + 1):
        positions = {qid: next(coord for (coord, tt) in routing_plan[qid] if tt == t)
                     for qid in routing_plan}
        print(f"t={t}: {positions}")

    animate_mapf(G, routing_plan)
