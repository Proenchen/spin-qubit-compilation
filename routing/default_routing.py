from __future__ import annotations

from copy import deepcopy 
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx
import random

from routing.common import AStar, Coord, MAX_TIME, Reservations, TimedNode, Qubit

P_SUCCESS = 0.98
P_REPAIR = 0.5


class DefaultRoutingPlanner:
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
        occupied_nodes: Optional[Set[Coord]] = None,
        blocked_edges: Optional[Set[frozenset]] = None,
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
                cands = DefaultRoutingPlanner.meeting_candidates(G, qa.pos, qb.pos, tmp_reserved, occupied_nodes)
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
        res = Reservations(G, blocked_edges=blocked_edges or set())

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
                candidates = DefaultRoutingPlanner.meeting_candidates(
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

                a_plan2 = DefaultRoutingPlanner._retime_to_pre_in_wait(a_plan, meeting, Tm)
                if a_plan2 is None:
                    continue
                b_plan2 = DefaultRoutingPlanner._retime_to_pre_in_wait(b_plan, meeting, Tm)
                if b_plan2 is None:
                    continue

                # (c) Validieren & endgültig committen gegen die aktuelle globale 'res'
                res_commit = deepcopy(res)
                if not DefaultRoutingPlanner._try_commit(res_commit, a_plan2):
                    continue
                if not DefaultRoutingPlanner._try_commit(res_commit, b_plan2):
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
    def route(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float = P_SUCCESS,
        p_repair: float = P_REPAIR,
    ):
        """
        Layer-basiertes Routing mit persistentem Kanten-Fehlermodell.

        Änderungen ggü. vorher:
        - Vor der Ausführung jedes Layers wird über alle Kanten gesampelt:
            * jede intakte Kante wird mit W'keit (1 - p_success) defekt
            * jede defekte Kante wird mit W'keit p_repair repariert
            Defekte Kanten bleiben über Layer hinweg gesperrt.
        - Pfadfindung (MAPF/A*) meidet defekte Kanten.
        - Nach dem Sampling werden bereits geplante Pfade geprüft; unbenutzbare Paare
            wandern in ein Spillover-Layer direkt hinter das aktuelle Layer.
        - Qubit-Bewegungen selbst sind deterministisch erfolgreich (kein Schritt-Sampling).
        - Für die Animation wird zusätzlich eine Zeitband-Liste der defekten Kanten erzeugt,
            sodass defekte Kanten in den entsprechenden Zeitfenstern rot dargestellt werden.

        Rückgabe:
        timelines, edge_timebands
            timelines: Dict[qid, List[(coord, t)]]
            edge_timebands: List[(t_start, t_end, Set[frozenset({u,v})])]
        """

        # ---- Zustand / Hilfsstrukturen ----
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}
        qids_all: Set[int] = {q.id for q in qubits}

        # Persistenter Zustand defekter Kanten über alle Layer
        defective_edges: Set[frozenset] = set()

        # Wir sammeln Microbatches (sichtbare Teil-„Episoden“) und für jeden Batch
        # einen Snapshot der defekten Kanten (für die rote Overlay-Animation).
        batch_plans: List[Dict[int, List[TimedNode]]] = []
        batch_defects: List[Set[frozenset]] = []

        def add_defect_snapshots(n_new_batches: int):
            """Fügt für die letzten n_new_batches jeweils den aktuellen Defekt-Snapshot hinzu."""
            for _ in range(n_new_batches):
                batch_defects.append(set(defective_edges))

        # Schnelle Map auf Qubit-Objekt in aktueller Position
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
                if cur_layer:
                    layers.append(cur_layer)
                cur_layer = [(a, b)]
                used = {a, b}
        if cur_layer:
            layers.append(cur_layer)

        # ---------- Hilfsfunktion: eindeutige INs + Pfade für Paare dieses Layers ----------
        def plan_layer_candidates(
            layer_pairs: List[Tuple[int, int]],
        ) -> Tuple[Dict[int, List[TimedNode]], Dict[frozenset, Coord], List[Tuple[int, int]]]:
            """
            Plant so viele Paare wie möglich im aktuellen Layer mit:
            - eindeutigen INs pro Paar
            - eindeutigen PRE-INs (SN) für alle Layer-Qubits
            Ein Paar kommt in 'spill_pairs', wenn keiner seiner Meeting-Kandidaten funktioniert.
            Achtung: Verwendet *aktuell bekannte* defekte Kanten und meidet diese bereits in der Planung.
            """
            attempt_full: Dict[int, List[TimedNode]] = {}
            meeting_of_pair: Dict[frozenset, Coord] = {}
            spill_pairs: List[Tuple[int, int]] = []

            layer_start_nodes: Set[Coord] = {current_pos[a] for a, _ in layer_pairs} | {current_pos[b] for _, b in layer_pairs}
            occupied_now: Set[Coord] = {current_pos[q] for q in qids_all}
            reserved_in_nodes: Set[Coord] = set()

            # Heuristische Reihenfolge: wenige Kandidaten zuerst, dann größere maxd
            pair_infos = []
            for a_id, b_id in layer_pairs:
                qa_now, qb_now = qobj(a_id), qobj(b_id)
                cands = DefaultRoutingPlanner.meeting_candidates(
                    G,
                    qa_now.pos, qb_now.pos,
                    reserved_meetings=reserved_in_nodes,
                    occupied_nodes=(occupied_now | (layer_start_nodes - {qa_now.pos, qb_now.pos})),
                )
                # Distanzen (nur Heuristik)
                da = nx.single_source_shortest_path_length(G, qa_now.pos)
                db = nx.single_source_shortest_path_length(G, qb_now.pos)
                maxd = min((max(da[n], db[n]) for n in cands), default=10**9)
                pair_infos.append((a_id, b_id, len(cands), maxd, cands))

            pair_infos.sort(key=lambda x: (x[2], -x[3]))

            active_pairs: List[Tuple[int, int]] = []
            active_fixed_meetings: Dict[frozenset, Coord] = {}
            active_plans: Dict[int, List[TimedNode]] = {}

            for a_id, b_id, _, _, _ in pair_infos:
                qa_now, qb_now = qobj(a_id), qobj(b_id)

                placed = False
                # Kandidaten dynamisch neu berechnen (wegen inzwischen reservierter INs)
                cands = DefaultRoutingPlanner.meeting_candidates(
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

                    allowed = {current_pos[x] for (x, _) in try_pairs} | {current_pos[y] for (_, y) in try_pairs}
                    hard_blocked = (layer_start_nodes - allowed)

                    try:
                        tmp_plans = DefaultRoutingPlanner.pairwise_mapf(
                            G,
                            try_objs,
                            fixed_meetings=try_fixed,
                            occupied_nodes=occupied_now | hard_blocked,
                            blocked_edges=defective_edges,  # <— defekte Kanten strikt meiden
                        )
                    except RuntimeError:
                        continue

                    # PRE-INs bestimmen und eindeutige PRE-INs fordern
                    preins_map = DefaultRoutingPlanner._preins_for_plans(tmp_plans, try_fixed)
                    if preins_map is None:
                        continue
                    if len(set(preins_map.values())) != len(preins_map):
                        continue

                    # akzeptieren
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

            # 1) Kandidaten & Pfade (bis Meeting) ... (dein bestehender Code)
            attempt_full, meeting_of_pair, spill_pairs = plan_layer_candidates(layer_pairs)

            # NEU: Paare, die in diesem Layer nicht platzierbar waren, in ein Spillover-Layer schieben
            if spill_pairs:
                layers[idx+1:idx+1] = [spill_pairs]


            # 1a) Blocker räumen (Nicht-Layer-Qubits auf Layer-Pfad-/IN-Knoten)
            #     Diese Funktion sollte intern _mapf_to_targets mit blocked_edges=defective_edges verwenden.
            prev_batches = len(batch_plans)
            remaining_block_coords = DefaultRoutingPlanner._evacuate_blocking_non_layer(
                G, qubits, current_pos,
                layer_pairs,
                attempt_full,
                meeting_of_pair,
                batch_plans
            )
            # Für alle während der Evakuierung erzeugten Microbatches den Defektzustand mitloggen
            add_defect_snapshots(len(batch_plans) - prev_batches)

            # Blocker nicht vollständig räumbar → betroffene Paare in Spillover-Layer verschieben
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
                        attempt_full.pop(a_id, None)
                        attempt_full.pop(b_id, None)
                        meeting_of_pair.pop(pkey, None)

                if newly_spilled:
                    layers[idx+1:idx+1] = [newly_spilled]

            # Wenn kein Paar mehr in diesem Layer übrig ist → sichtbarer Wartebatch
            if not any(frozenset({a, b}) in meeting_of_pair for (a, b) in layer_pairs):
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                add_defect_snapshots(1)
                idx += 1
                continue

            # 1b) PRE-IN schneiden (noch vor Sampling, aber Pfadfindung hat defekte Kanten bereits gemieden)
            pre_in_paths: Dict[int, List[TimedNode]] = {}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]
                for qid in (a_id, b_id):
                    cut = DefaultRoutingPlanner._retime_until_pre_in_wait(attempt_full[qid], meet, 0)
                    if cut is None:
                        # sollte selten sein → ab ins Spillover
                        attempt_full.pop(a_id, None)
                        attempt_full.pop(b_id, None)
                        meeting_of_pair.pop(pkey, None)
                        break
                    pre_in_paths[qid] = cut

            # Falls durch das Schneiden alles weggefallen ist
            if not pre_in_paths:
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                add_defect_snapshots(1)
                idx += 1
                continue

            # --- NEU: Kanten-Fehlermodell anwenden (P_fail=1-p_success, P_repair)
            DefaultRoutingPlanner._sample_edge_failures(
                G,
                defective_edges,
                p_fail=(1.0 - p_success),
                p_repair=p_repair,
            )

            # Nach dem Sampling prüfen, ob PRE-IN-Pfade + IN-Hop noch gültig sind; sonst in Spillover
            def pair_invalid(a_id: int, b_id: int, meet: Coord) -> bool:
                pa = pre_in_paths.get(a_id)
                pb = pre_in_paths.get(b_id)
                if pa is None or pb is None:
                    return True
                if DefaultRoutingPlanner._path_uses_defective_edge(pa, defective_edges):
                    return True
                if DefaultRoutingPlanner._path_uses_defective_edge(pb, defective_edges):
                    return True
                pre_a = pa[-1][0]
                pre_b = pb[-1][0]
                hop_edges = {frozenset({pre_a, meet}), frozenset({pre_b, meet})}
                return any(e in defective_edges for e in hop_edges)

            newly_spilled_after_sampling: List[Tuple[int, int]] = []
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]
                if pair_invalid(a_id, b_id, meet):
                    newly_spilled_after_sampling.append((a_id, b_id))

            if newly_spilled_after_sampling:
                for (a_id, b_id) in newly_spilled_after_sampling:
                    attempt_full.pop(a_id, None)
                    attempt_full.pop(b_id, None)
                    meeting_of_pair.pop(frozenset({a_id, b_id}), None)
                    pre_in_paths.pop(a_id, None)
                    pre_in_paths.pop(b_id, None)
                layers[idx+1:idx+1] = [newly_spilled_after_sampling]

            # Wenn nach Sampling/Spill nichts mehr übrig ist → Wartebatch
            if not pre_in_paths:
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                add_defect_snapshots(1)
                idx += 1
                continue

            # 2) Synchronisiere auf T_pre (max über alle) und führe deterministisch aus
            T_pre_sync = max(
                DefaultRoutingPlanner._first_pre_in_time(attempt_full[qid], meeting_of_pair[frozenset({a, b})])
                if frozenset({a, b}) in meeting_of_pair and qid in (a, b) else 0
                for (a, b) in layer_pairs for qid in (a, b)
                if frozenset({a, b}) in meeting_of_pair
            )

            micro_paths_approach: Dict[int, List[TimedNode]] = {}
            layer_qids = {q for ab in layer_pairs if frozenset(ab) in meeting_of_pair for q in ab}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]
                for qid in (a_id, b_id):
                    cut = DefaultRoutingPlanner._retime_until_pre_in_wait(attempt_full[qid], meet, T_pre_sync)
                    micro_paths_approach[qid] = cut

            # Qubits außerhalb des Layers warten sichtbar bis T_pre_sync+1
            others = qids_all - layer_qids
            for qid in others:
                micro_paths_approach[qid] = [(current_pos[qid], 0), (current_pos[qid], T_pre_sync + 1)]

            batch_plans.append(micro_paths_approach)
            add_defect_snapshots(1)

            # Positionen aktualisieren (alle Layer-Qubits stehen nun am PRE-IN)
            for qid in layer_qids:
                current_pos[qid] = micro_paths_approach[qid][-1][0]

            # 3) kurzer IN-Hop (rein und sofort raus auf PRE-IN), nur wenn Hop-Kanten nicht defekt
            micro_in: Dict[int, List[TimedNode]] = {}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]

                # Prüfe Hop-Kanten erneut (sollte bereits oben geprüft sein)
                pre_a = current_pos[a_id]
                pre_b = current_pos[b_id]
                if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                    # Wenn Hop nicht möglich ist, verzichtbar: beide warten (oder ins nächste Layer verschieben).
                    # Hier: sichtbar warten 2 Schritte.
                    micro_in[a_id] = [(pre_a, 0), (pre_a, 2)]
                    micro_in[b_id] = [(pre_b, 0), (pre_b, 2)]
                    continue

                # Regulärer kurzer Hop: PRE-IN -> IN -> PRE-IN
                micro_in[a_id] = [(pre_a, 0), (meet, 1), (pre_a, 2)]
                micro_in[b_id] = [(pre_b, 0), (meet, 1), (pre_b, 2)]

            # Nicht beteiligte Qubits warten sichtbar
            for qid in (qids_all - {q for ab in layer_pairs if frozenset(ab) in meeting_of_pair for q in ab}):
                micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]

            batch_plans.append(micro_in)
            add_defect_snapshots(1)

            idx += 1

        # Timelines + Zeitbänder für defekte Kanten erzeugen
        return DefaultRoutingPlanner.stitch_batches(qubits, batch_plans, batch_defects)




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
        blocked_edges: Optional[Set[frozenset]] = None,
    ) -> Dict[int, List[TimedNode]]:
        """
        Collision-free multi-agent routing from per-qubit start -> target.
        Uses the same A* and Reservations; no synchronization needed.
        """
        if not starts:
            return {}

        res = Reservations(G, blocked_edges=blocked_edges or set())
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
        batch_defects: Optional[List[Set[frozenset]]] = None,
    ) -> Tuple[Dict[int, List[TimedNode]], List[Tuple[int, int, Set[frozenset]]]]:
        """
        Fügt Microbatches zu durchgehenden Timelines zusammen und erzeugt Zeitbänder
        für defekte Kanten. Annahme: Es gibt KEINE Resets zwischen Batches
        (d.h. der Endknoten eines Qubits in Batch b ist der Startknoten in Batch b+1).
        """
        if not batch_plans:
            timelines = {q.id: [(q.pos, 0)] for q in qubits}
            return timelines, []

        # Nur noch Batch-Dauern bestimmen
        durations: List[int] = []
        for plans in batch_plans:
            if not plans:
                durations.append(0)
                continue
            durations.append(max(p[-1][1] for p in plans.values()))

        # Timelines initialisieren
        initial_pos = {q.id: q.pos for q in qubits}
        timelines: Dict[int, List[TimedNode]] = {q.id: [(initial_pos[q.id], 0)] for q in qubits}

        edge_timebands: List[Tuple[int, int, Set[frozenset]]] = []

        t_offset = 0
        for b, plans in enumerate(batch_plans):
            batch_T = durations[b]

            # Lücken bis Batch-Start auffüllen (Warten)
            for q in qubits:
                qid = q.id
                last_coord, last_t = timelines[qid][-1]
                if last_t < t_offset:
                    for tt in range(last_t + 1, t_offset + 1):
                        timelines[qid].append((last_coord, tt))

            if batch_T == 0:
                # Kein Zeitband für leere Batches
                continue

            # Pfade dieses Batches einfügen (ohne Reset-/Interpolationslogik)
            for q in qubits:
                qid = q.id
                last_coord, last_t = timelines[qid][-1]

                if qid not in plans:
                    # Nicht beteiligt: warte über den gesamten Batch
                    target_t = t_offset + batch_T
                    for tt in range(last_t + 1, target_t + 1):
                        timelines[qid].append((last_coord, tt))
                    continue

                local_path = plans[qid]  # Zeiten 0..batch_T
                shifted = [(c, t + t_offset) for (c, t) in local_path]

                # Dublette am Übergang vermeiden
                if timelines[qid][-1] == shifted[0]:
                    timelines[qid].extend(shifted[1:])
                else:
                    timelines[qid].extend(shifted)

            # Zeitband der defekten Kanten für diesen Batch
            defects = set(batch_defects[b]) if (batch_defects is not None and b < len(batch_defects)) else set()
            edge_timebands.append((t_offset, t_offset + batch_T, defects))

            t_offset += batch_T

        return timelines, edge_timebands



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
        Deterministisch, ohne probabilistische Resets.
        Routen bleiben strikt außerhalb der vom Layer benötigten Knoten.

        Rückgabe: Menge verbleibender blockierender Positionen (leer = alles frei).
        """
        if not attempt_full:
            return set()

        qids_all: Set[int] = {q.id for q in qubits}
        layer_qids: Set[int] = {q for ab in layer_pairs for q in ab}
        others: List[int] = [qid for qid in qids_all if qid not in layer_qids]

        # Knoten, die der Layer benötigt (Pfadknoten + INs) + Startknoten der Layer-Qubits
        F_layer = DefaultRoutingPlanner._collect_layer_forbidden_nodes(attempt_full, meeting_of_pair)
        layer_starts: Set[Coord] = {current_pos[qid] for qid in layer_qids}
        F_all = set(F_layer) | set(layer_starts)

        # Blocker = Non-Layer-Qubits, die derzeit im Layer-Gebiet stehen
        blockers: List[int] = [qid for qid in others if current_pos[qid] in F_layer]
        if not blockers:
            return set()  # nichts im Weg

        occupied_now: Set[Coord] = {current_pos[qid] for qid in qids_all}
        avoid: Set[Coord] = set(occupied_now) | F_all

        # Ziele: nächste freie SN außerhalb des Layer-Gebiets (ohne Doppelziele)
        targets: Dict[int, Coord] = {}
        for qid in blockers:
            tgt = DefaultRoutingPlanner._nearest_free_sn(G, current_pos[qid], avoid)
            if tgt is not None:
                targets[qid] = tgt
                avoid.add(tgt)

        if not targets:
            # keiner fand ein Ziel → alle ursprünglichen Blocker-Positionen bleiben
            return {current_pos[qid] for qid in blockers}

        starts = {qid: current_pos[qid] for qid in targets}
        blocked_nodes = F_all  # Evakuierungswege dürfen KEINEN Layer-Knoten betreten

        micro: Dict[int, List[TimedNode]] = {}

        # 1) Versuche gemeinsame kollisionsfreie Planung
        try:
            plans = DefaultRoutingPlanner._mapf_to_targets(
                G, starts, targets, blocked_nodes=blocked_nodes
            )
            # Ausführen (deterministisch): füge Pfade in einen Microbatch ein
            for qid, path in plans.items():
                micro[qid] = path
                current_pos[qid] = path[-1][0]
        except RuntimeError:
            # 2) Fallback: agent-weise, jeweils mit aktueller Sperrmenge erweitern
            #    Erfolgreiche Moves blockieren ihre Zielknoten für nachfolgende.
            blocked_now = set(blocked_nodes)
            for qid, tgt in targets.items():
                try:
                    one = DefaultRoutingPlanner._mapf_to_targets(
                        G, {qid: current_pos[qid]}, {qid: tgt}, blocked_nodes=blocked_now
                    )
                    path = one[qid]
                    micro[qid] = path
                    current_pos[qid] = path[-1][0]
                    blocked_now.add(current_pos[qid])
                except RuntimeError:
                    # Keine Route für diesen Agenten (bleibt stehen)
                    pass

        if micro:
            batch_plans.append(micro)

        # Erneut prüfen: welche Non-Layer stehen noch auf Layer-Knoten?
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
                pin = DefaultRoutingPlanner._entry_sn_from_path(path, meet)
                if pin is None:
                    return None
                preins[qid] = pin
        return preins
    
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
    def _sample_edge_failures(
        G: nx.Graph,
        defective_edges: Set[frozenset],
        p_fail: float,
        p_repair: float,
    ) -> None:
        # Aktualisiert das Set in place
        for u, v in G.edges():
            e = frozenset({u, v})
            if e in defective_edges:
                # Reparaturversuch
                if random.random() < p_repair:
                    defective_edges.discard(e)
            else:
                # Neuer Fehler
                if random.random() < p_fail:
                    defective_edges.add(e)

    @staticmethod
    def _path_uses_defective_edge(path: List[TimedNode], defective_edges: Set[frozenset]) -> bool:
        for (u, _), (v, _) in zip(path[:-1], path[1:]):
            if u != v and frozenset({u, v}) in defective_edges:
                return True
        return False
    