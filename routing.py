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
P_SUCCESS = 0.8


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
    def layered_routing(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float = P_SUCCESS,
    ) -> Dict[int, List[TimedNode]]:
        """
        Layer-basierter Router gemäß Spezifikation:

        1) Paare in Layer zerlegen (greedy, disjunkt pro Layer).
        2) Pro Layer:
           - Schritt 1: Für jedes Paar einen eindeutigen IN finden und Anmarschpfade bis zum IN planen.
                        Startpositionen aller Layer-Qubits sind füreinander gesperrt.
                        Nicht-Layer-Blocker, die auf Pfadknoten sitzen, werden vorab auf freie SN verschoben.
                        Paare ohne validen IN/Anmarsch -> Spillover-Layer direkt hinter diesem Layer.
                        In der Ausführung bewegen sich die Qubits nur bis PRE-IN und warten synchron.
                        Sampling mit P_SUCCESS: nicht erfolgreich -> zurück zum Layer-Start, erneut versuchen,
                        bis alle ihren PRE-IN erfolgreich erreicht haben.
           - Schritt 3: Alle gehen einmal in den IN und sofort zurück auf PRE-IN.
        3) Nächstes Layer (inkl. eingeschobener Spillover-Layer).
        """

        # ---- Zustand / Hilfsstrukturen ----
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}
        qids_all: Set[int] = {q.id for q in qubits}
        batch_plans: List[Dict[int, List[TimedNode]]] = []  # wird am Ende mit stitch_batches zusammengesetzt

        # Schnelle Maps
        def qobj(qid: int) -> Qubit:
            return Qubit(qid, current_pos[qid])

        # ---------- Layerbildung (greedy, disjunkt) ----------
        layers: List[List[Tuple[int, int]]] = []
        used: Set[int] = set()
        cur_layer: List[Tuple[int, int]] = []
        for qa, qb in pairs:
            a, b = qa.id, qb.id
            if a not in used and b not in used:
                cur_layer.append((a, b))
                used.add(a); used.add(b)
            else:
                # starte neues Layer, falls Kollision
                if cur_layer:
                    layers.append(cur_layer)
                cur_layer = [(a, b)]
                used = {a, b}
        if cur_layer:
            layers.append(cur_layer)

        # Hilfsfunktion: eindeutige IN-Auswahl + Pfadplanung (nur Kandidaten sammeln)
        def plan_layer_candidates(
            layer_pairs: List[Tuple[int, int]],
        ) -> Tuple[Dict[int, List[TimedNode]], Dict[frozenset, Coord], List[Tuple[int, int]]]:
            """
            Plant so viele Paare wie möglich im aktuellen Layer, mit:
            - eindeutigen INs pro Paar
            - eindeutigen PRE-INs (SN) für *alle* Layer-Qubits
            Ein Paar wird nur dann in 'spill_pairs' gelegt, wenn *keiner* seiner Meeting-Kandidaten
            (unter allen bereits gewählten Paaren) funktioniert.
            """
            attempt_full: Dict[int, List[TimedNode]] = {}
            meeting_of_pair: Dict[frozenset, Coord] = {}
            spill_pairs: List[Tuple[int, int]] = []

            # Sperrmenge: Startknoten aller Qubits im Layer (gegenseitig tabu)
            layer_start_nodes: Set[Coord] = {current_pos[a] for a, _ in layer_pairs} | {current_pos[b] for _, b in layer_pairs}
            # Belegte Knoten jetzt (alle)
            occupied_now: Set[Coord] = {current_pos[q] for q in qids_all}

            # pro Layer: IN darf nicht doppelt belegt werden
            reserved_in_nodes: Set[Coord] = set()

            # Greedy-Reihenfolge: "schwierig zuerst"
            # (heuristisch: wenig Kandidaten -> zuerst; sonst große max-Distanz)
            pair_infos = []
            for a_id, b_id in layer_pairs:
                qa_now, qb_now = qobj(a_id), qobj(b_id)
                cands = RoutingPlanner.meeting_candidates(
                    G,
                    qa_now.pos, qb_now.pos,
                    reserved_meetings=reserved_in_nodes,  # leer zu Beginn
                    occupied_nodes=(occupied_now | (layer_start_nodes - {qa_now.pos, qb_now.pos})),
                )
                # grobe Distanzen für Order-Heuristik
                da = nx.single_source_shortest_path_length(G, qa_now.pos)
                db = nx.single_source_shortest_path_length(G, qb_now.pos)
                maxd = min((max(da[n], db[n]) for n in cands), default=10**9)
                pair_infos.append((a_id, b_id, len(cands), maxd, cands))

            # sortiere: wenig Kandidaten zuerst, bei Gleichheit größere maxd zuerst
            pair_infos.sort(key=lambda x: (x[2], -x[3]))

            active_pairs: List[Tuple[int, int]] = []
            active_fixed_meetings: Dict[frozenset, Coord] = {}
            active_plans: Dict[int, List[TimedNode]] = {}

            for a_id, b_id, _, _, initial_cands in pair_infos:
                qa_now, qb_now = qobj(a_id), qobj(b_id)

                placed = False
                # Kandidaten dynamisch neu berechnen (weil wir inzwischen INs reserviert haben)
                cands = RoutingPlanner.meeting_candidates(
                    G,
                    qa_now.pos, qb_now.pos,
                    reserved_meetings=set(active_fixed_meetings.values()),
                    occupied_nodes=(occupied_now | (layer_start_nodes - {qa_now.pos, qb_now.pos})),
                )

                for mnode in cands:
                    try_pairs = active_pairs + [(a_id, b_id)]
                    try_objs = [(qobj(x), qobj(y)) for (x, y) in try_pairs]
                    try_fixed = dict(active_fixed_meetings)
                    try_fixed[frozenset({a_id, b_id})] = mnode

                    # Sperren: blockiere alle Layer-Startknoten außer denen der try-Menge
                    allowed = {current_pos[x] for (x, _) in try_pairs} | {current_pos[y] for (_, y) in try_pairs}
                    hard_blocked = (layer_start_nodes - allowed)

                    # Versuche gemeinsame, konsistente Planung für alle bisher aktiven + dieses Paar
                    try:
                        tmp_plans = RoutingPlanner.pairwise_mapf(
                            G,
                            try_objs,
                            fixed_meetings=try_fixed,
                            occupied_nodes=occupied_now | hard_blocked,
                        )
                    except RuntimeError:
                        continue

                    # PRE-INs für *alle* bisher geplanten Paare neu bestimmen
                    preins_map = RoutingPlanner._preins_for_plans(tmp_plans, try_fixed)
                    if preins_map is None:
                        # irgendein PRE-IN nicht bestimmbar -> nächster Kandidat
                        continue

                    # Eindeutigkeit prüfen (global über alle Layer-Qubits dieses Versuchs)
                    all_preins = list(preins_map.values())
                    if len(set(all_preins)) != len(all_preins):
                        # Kollision bei PRE-IN -> nächster Kandidat
                        continue

                    # akzeptieren: übernehme neue globale Planung
                    active_pairs = try_pairs
                    active_fixed_meetings = try_fixed
                    active_plans = tmp_plans
                    reserved_in_nodes.add(mnode)
                    placed = True
                    break

                if not placed:
                    spill_pairs.append((a_id, b_id))

            attempt_full = active_plans
            meeting_of_pair = {frozenset({a, b}): active_fixed_meetings[frozenset({a, b})] for (a, b) in active_pairs}
            return attempt_full, meeting_of_pair, spill_pairs

    

        # ---------- Haupt-Loop über Layer inkl. Spillover ----------
        idx = 0
        while idx < len(layers):
            layer_pairs = layers[idx]

            # 1) Kandidaten & Pfade (bis Meeting) unter Layer-Sperren bestimmen
            attempt_full, meeting_of_pair, spill_pairs = plan_layer_candidates(layer_pairs)

            # 1a) Blocker räumen (Nicht-Layer-Qubits auf Pfadknoten)
            remaining_block_coords = RoutingPlanner._evacuate_blocking_non_layer(
                G, qubits, current_pos,
                layer_pairs,
                attempt_full,
                meeting_of_pair,
                batch_plans,
            )

            # Falls Evakuierung nicht alle Blocker wegbekam:
            # verschiebe *nur* die betroffenen Paare ins Spillover-Layer (direkt hinter dieses Layer)
            if remaining_block_coords:
                def path_nodes_of_pair(a_id: int, b_id: int) -> Set[Coord]:
                    return {c for (c, _) in attempt_full.get(a_id, [])} | \
                           {c for (c, _) in attempt_full.get(b_id, [])}

                newly_spilled: List[Tuple[int, int]] = []
                for (a_id, b_id) in layer_pairs:
                    pkey = frozenset({a_id, b_id})
                    if pkey not in meeting_of_pair:
                        continue
                    pn = path_nodes_of_pair(a_id, b_id)
                    if pn & remaining_block_coords:
                        newly_spilled.append((a_id, b_id))
                        # entferne Paar aus diesem Layer-Plan
                        attempt_full.pop(a_id, None)
                        attempt_full.pop(b_id, None)
                        meeting_of_pair.pop(pkey, None)

                if newly_spilled:
                    # füge direkt hinter diesem Layer ein Spillover-Layer ein
                    layers[idx+1:idx+1] = [newly_spilled]

            # Wenn nach Evakuierung/Spillover kein Paar mehr in diesem Layer planbar ist:
            if not any(frozenset({a,b}) in meeting_of_pair for (a,b) in layer_pairs):
                # Sichtbares Warte-Microbatch und weiter
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                idx += 1
                continue



            # 1b) pre-IN schneiden
            pre_in_arrival: Dict[int, int] = {}
            pre_in_paths: Dict[int, List[TimedNode]] = {}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue  # dieses Paar wurde gesplittet
                meet = meeting_of_pair[pkey]
                for qid in (a_id, b_id):
                    cut = RoutingPlanner._retime_until_pre_in_wait(attempt_full[qid], meet, 0)  # erstmal nur bis PRE-IN schneiden
                    if cut is None:
                        # sollte praktisch nicht passieren, aber falls doch -> Spill
                        spill_pairs.append((a_id, b_id))
                        break
                    # Zeitstempel am Ende = PRE-IN-Ankunftszeit
                    t_pre = cut[-1][1]
                    pre_in_arrival[qid] = t_pre
                    pre_in_paths[qid] = cut

            # Paare, die nicht planbar waren, in Spillover-Layer direkt hinter diesem Layer
            if spill_pairs:
                layers[idx+1:idx+1] = [spill_pairs]

            # Wenn in diesem Layer nichts planbar war: sichtbare Warte-Microbatch und weiter
            if not pre_in_paths:
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                # Positionen bleiben
                idx += 1
                continue

            # 2) Synchronisiere auf T_pre (max über alle) und wiederhole Versuche bis Erfolg
            T_pre_sync = 0
            # Erstes Microbatch: alle bewegen sich bis PRE-IN und warten, bis alle dort sind
            T_pre_sync = max(
                RoutingPlanner._first_pre_in_time(attempt_full[qid], meeting_of_pair[frozenset({a,b})])
                if frozenset({a,b}) in meeting_of_pair and qid in (a,b) else 0
                for (a,b) in layer_pairs for qid in (a,b)
                if frozenset({a,b}) in meeting_of_pair
            )
            # Schneiden & auf T_pre_sync auffüllen
            micro_paths_approach: Dict[int, List[TimedNode]] = {}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]
                for qid in (a_id, b_id):
                    cut = RoutingPlanner._retime_until_pre_in_wait(attempt_full[qid], meet, T_pre_sync)
                    micro_paths_approach[qid] = cut
            # Qubits außerhalb des Layers warten sichtbar
            layer_qids = {q for ab in layer_pairs if frozenset(ab) in meeting_of_pair for q in ab}
            others = qids_all - layer_qids
            for qid in others:
                micro_paths_approach[qid] = [(current_pos[qid], 0), (current_pos[qid], T_pre_sync+1)]
            batch_plans.append(micro_paths_approach)
            for qid in layer_qids:
                current_pos[qid] = micro_paths_approach[qid][-1][0]

            # Wiederholtes Sampling, bis alle Layer-Qubits ihren PRE-IN erfolgreich erreicht haben
            success: Dict[int, bool] = dict.fromkeys(others, True)
            success.update(dict.fromkeys(layer_qids, False))
            # Merke die Startknoten des Layers
            layer_starts: Dict[int, Coord] = {q: pre_in_paths[q][0][0] for q in layer_qids}

            while not all(success[q] for q in layer_qids):
                micro: Dict[int, List[TimedNode]] = {}
                path_lengths: Dict[int, int] = {}
                targets_after: Dict[int, Coord] = {}

                for (a_id, b_id) in layer_pairs:
                    pkey = frozenset({a_id, b_id})
                    if pkey not in meeting_of_pair:
                        continue
                    meet = meeting_of_pair[pkey]

                    for qid in (a_id, b_id):
                        if success[qid]:
                            # schon am PRE-IN: erstmal Platzhalter, Länge setzen wir später
                            micro[qid] = [(current_pos[qid], 0)]
                            path_lengths[qid] = 0
                            targets_after[qid] = current_pos[qid]
                            continue

                        start = current_pos[qid]
                        pre_node = pre_in_paths[qid][-1][0]
                        layer_start = layer_starts[qid]

                        ok = (random.random() < p_success)

                        if ok:
                            # ERFOLG: von aktueller Position zum PRE_IN *laufen*
                            try:
                                node_path = nx.shortest_path(G, source=start, target=pre_node)
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                node_path = [start, pre_node]

                            timed: List[TimedNode] = []
                            t = 0
                            timed.append((node_path[0], t))
                            for k in range(1, len(node_path)):
                                t += 1
                                timed.append((node_path[k], t))

                            micro[qid] = timed
                            path_lengths[qid] = t
                            targets_after[qid] = pre_node

                            success[qid] = True
                            current_pos[qid] = pre_node

                        else:
                            # FEHLSCHLAG: harter Reset — *Teleport* in 1 Schritt zurück zum layer_start
                            timed: List[TimedNode] = [(start, 0), (layer_start, 1)]
                            micro[qid] = timed
                            path_lengths[qid] = 1
                            targets_after[qid] = layer_start

                            current_pos[qid] = layer_start


                # Nicht-Layer-Qubits warten, Dauer wird gleich aufgefüllt
                for qid in (qids_all - layer_qids):
                    micro[qid] = [(current_pos[qid], 0)]
                    path_lengths[qid] = 0
                    targets_after[qid] = current_pos[qid]

                # Alle Pfade auf gleiche Micro-Dauer auffüllen (Warten am Endknoten)
                Lmax = max(path_lengths.values()) if path_lengths else 1
                for qid, path in micro.items():
                    # path beginnt bei t=0; verlängere bis Lmax
                    last_node, last_t = path[-1]
                    for t in range(last_t + 1, Lmax + 1):
                        path.append((last_node, t))

                batch_plans.append(micro)


            # 3) kurzer IN-Hop (rein und sofort raus auf PRE-IN)
            micro_in: Dict[int, List[TimedNode]] = {}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id,b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]
                for qid in (a_id, b_id):
                    pre_node = current_pos[qid]  # wir stehen aktuell am PRE-IN
                    micro_in[qid] = [(pre_node, 0), (meet, 1), (pre_node, 2)]
                    current_pos[qid] = pre_node
            for qid in (qids_all - {q for ab in layer_pairs if frozenset(ab) in meeting_of_pair for q in ab}):
                micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]
            batch_plans.append(micro_in)

            idx += 1

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
                    # 1) run until PRE_IN (wie gehabt)
                    first_in_idx = None
                    for idx2, (c, _) in enumerate(local_path):
                        if c == local_path[-1][0]:
                            first_in_idx = idx2
                            break
                    if first_in_idx is None or first_in_idx == 0:
                        anchor_coord = local_path[-1][0]  # Fallback
                    else:
                        anchor_coord = local_path[first_in_idx - 1][0]

                    last_anchor_idx = None
                    for idx2, (c, _) in enumerate(local_path):
                        if c == anchor_coord:
                            last_anchor_idx = idx2
                    if last_anchor_idx is None:
                        part = shifted  # Fallback
                    else:
                        part_local = local_path[: last_anchor_idx + 1]
                        part = [(c, t + t_offset) for (c, t) in part_local]

                    if part:
                        if timelines[qid][-1] == part[0]:
                            timelines[qid].extend(part[1:])
                        else:
                            timelines[qid].extend(part)

                    # 2) Sichtbare Bewegung **vom PRE_IN zum 'nxt'** innerhalb der Restzeit
                    current_last_t = timelines[qid][-1][1]
                    target_t = t_offset + batch_T
                    rem_steps = target_t - current_last_t
                    if rem_steps <= 0 or nxt is None:
                        continue

                    src = anchor_coord
                    if src == nxt:
                        # gleicher Knoten: einfach warten
                        for tt in range(current_last_t + 1, target_t + 1):
                            timelines[qid].append((nxt, tt))
                        continue

                    # Kürzesten Pfad im Graphen bestimmen (nur Knotenfolge)
                    try:
                        move_nodes = nx.shortest_path(G, source=src, target=nxt)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # Fallback: wenn keine Route, warte (altes Verhalten)
                        for tt in range(current_last_t + 1, target_t + 1):
                            timelines[qid].append((nxt, tt))
                        continue

                    steps_needed = max(0, len(move_nodes) - 1)

                    if steps_needed == 0:
                        # sollte oben schon behandelt sein, aber zur Sicherheit
                        for tt in range(current_last_t + 1, target_t + 1):
                            timelines[qid].append((nxt, tt))
                    else:
                        # Wir müssen GENAU rem_steps Zeitschritte füllen.
                        # Komprimieren (wenn Route länger) oder expandieren (wenn kürzer).
                        # Wähle Index auf der Route proportional zur Zeit.
                        for i in range(1, rem_steps + 1):
                            # i läuft 1..rem_steps und endet exakt bei nxt @ target_t
                            # Mappe i auf Pfadindex 0..steps_needed
                            idx_on_path = round(i * steps_needed / rem_steps)
                            node_i = move_nodes[idx_on_path]
                            timelines[qid].append((node_i, current_last_t + i))

            t_offset += batch_T

        return timelines

    @staticmethod
    def _collect_layer_forbidden_nodes(
        attempt_full: Dict[int, List[TimedNode]],
        meeting_of_pair: Dict[frozenset, Coord],
    ) -> Set[Coord]:
        """Alle vom Layer genutzten Knoten (Pfadknoten inkl. PRE-IN + INs)."""
        F: Set[Coord] = set()
        for path in attempt_full.values():
            for c, _ in path:
                F.add(c)
        for m in meeting_of_pair.values():
            F.add(m)
        return F

    @staticmethod
    def _evacuate_blocking_non_layer(
        G: nx.Graph,
        qubits: List[Qubit],
        current_pos: Dict[int, Coord],
        layer_pairs: List[Tuple[int, int]],
        attempt_full: Dict[int, List[TimedNode]],
        meeting_of_pair: Dict[frozenset, Coord],
        batch_plans: List[Dict[int, List[TimedNode]]],
    ) -> Set[Coord]:
        """
        Bewegt NUR Non-Layer-Qubits, deren *aktuelle Position* Layer-Pfade/INs blockiert.
        Hält Evakuierungsrouten strikt außerhalb der Layer-Knoten.
        Gibt die Menge verbleibender blockierender Positionen (Coords) zurück (leer = alles frei).
        """
        if not attempt_full:
            return set()

        qids_all: Set[int] = {q.id for q in qubits}
        layer_qids: Set[int] = {q for ab in layer_pairs for q in ab}
        others: List[int] = [qid for qid in qids_all if qid not in layer_qids]

        # Knoten, die der Layer benötigt (Pfadknoten + INs) + Startknoten der Layer-Qubits
        F_layer = RoutingPlanner._collect_layer_forbidden_nodes(attempt_full, meeting_of_pair)
        layer_starts: Set[Coord] = {current_pos[qid] for qid in layer_qids}
        F_all = set(F_layer) | set(layer_starts)

        # Blocker = Non-Layer-Qubits, die derzeit im Layer-Gebiet stehen
        blockers: List[int] = [qid for qid in others if current_pos[qid] in F_layer]
        if not blockers:
            return set()  # nichts im Weg

        occupied_now: Set[Coord] = {current_pos[qid] for qid in qids_all}
        avoid: Set[Coord] = set(occupied_now) | F_all

        # Zielwahl: nächste freie SN außerhalb des Layer-Gebiets (und ohne Doppelziele)
        targets: Dict[int, Coord] = {}
        for qid in blockers:
            tgt = RoutingPlanner._nearest_free_sn(G, current_pos[qid], avoid)
            if tgt is not None:
                targets[qid] = tgt
                avoid.add(tgt)

        if not targets:
            # keiner konnte ein Ziel finden → alle ursprünglichen Blocker-Positionen gelten als verbleibend
            return {current_pos[qid] for qid in blockers}

        starts = {qid: current_pos[qid] for qid in targets}
        # Evakuierungswege dürfen KEINEN Layer-Knoten betreten (auch nicht Layer-Startknoten)
        blocked_nodes = F_all

        # Versuche gemeinsame Evakuierung (kollisionsfrei, außerhalb des Layers)
        try:
            plans = RoutingPlanner._mapf_to_targets(G, starts, targets, blocked_nodes=blocked_nodes)
            if plans:
                batch_plans.append(plans)
                for qid, p in plans.items():
                    current_pos[qid] = p[-1][0]
        except RuntimeError:
            # Greedy agent-weise Evakuierung, jeweils mit aktueller Sperrmenge
            micro: Dict[int, List[TimedNode]] = {}
            for qid, tgt in targets.items():
                try:
                    one = RoutingPlanner._mapf_to_targets(
                        G, {qid: current_pos[qid]}, {qid: tgt}, blocked_nodes=blocked_nodes
                    )
                    micro.update(one)
                    current_pos[qid] = one[qid][-1][0]
                    # blockiere erreichte Ziele für nachfolgende
                    blocked_nodes = set(blocked_nodes) | {current_pos[qid]}
                except RuntimeError:
                    # Dieser Blocker bleibt zunächst stehen
                    pass
            if micro:
                batch_plans.append(micro)

        # Prüfe erneut: welche Non-Layer stehen noch auf Layer-Knoten?
        remaining_blockers: Set[Coord] = set()
        for qid in others:
            if current_pos[qid] in F_layer:
                remaining_blockers.add(current_pos[qid])

        return remaining_blockers
    
    @staticmethod
    def _preins_for_plans(
        plans: Dict[int, List[TimedNode]],
        fixed_meetings: Dict[frozenset, Coord],
    ) -> Optional[Dict[int, Coord]]:
        """
        Liefert für alle in fixed_meetings enthaltenen Qubits den PRE-IN-Knoten.
        Gibt None zurück, falls ein Pfad fehlt oder irgendein PRE-IN nicht bestimmbar ist.
        """
        preins: Dict[int, Coord] = {}
        for pair_key, meet in fixed_meetings.items():
            qids = list(pair_key)
            if len(qids) != 2:
                return None
            for qid in qids:
                path = plans.get(qid)
                if not path:
                    return None
                pin = RoutingPlanner._entry_sn_from_path(path, meet)
                if pin is None:
                    return None
                preins[qid] = pin
        return preins




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
    routing_plan = planner.layered_routing(G, qubits, pairs)

    animate_mapf(G, routing_plan)
