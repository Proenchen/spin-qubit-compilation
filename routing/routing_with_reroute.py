from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional

import networkx as nx
from routing.common import Coord, TimedNode, Qubit
from routing.default_routing import LayerFirstRoutingPlanner, P_REPAIR, P_SUCCESS


class RoutingPlannerWithRerouting(LayerFirstRoutingPlanner):
    """
    Wie DefaultRoutingPlanner, jedoch mit Rerouting innerhalb eines Layers:
    Wenn nach dem Sampling eine Kante auf einem Layer-Pfad defekt ist, wird versucht,
    eine Umfahrung zum selben Meeting-Node zu finden (ohne die defekte Kante zu benutzen).
    Gelingt dies kollisionsfrei und mit eindeutigen PRE-INs, wird das Paar im selben
    Layer ausgeführt; ansonsten kommt es ins Spillover-Layer.
    """

    @staticmethod
    def route(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float = P_SUCCESS,
        p_repair: float = P_REPAIR,
    ):
        # ---- Zustand / Hilfsstrukturen (wie Basis) ----
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}
        qids_all: Set[int] = {q.id for q in qubits}
        defective_edges: Set[frozenset] = set()

        batch_plans: List[Dict[int, List[TimedNode]]] = []
        batch_defects: List[Set[frozenset]] = []

        def add_defect_snapshots(n_new_batches: int):
            for _ in range(n_new_batches):
                batch_defects.append(set(defective_edges))

        def qobj(qid: int) -> Qubit:
            return Qubit(qid, current_pos[qid])

        # ---------- Layerbildung (identisch) ----------
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

        # ---------- Hilfsfunktionen für (Re)Planning ----------
        def plan_layer_candidates(
            layer_pairs: List[Tuple[int, int]],
        ) -> Tuple[Dict[int, List[TimedNode]], Dict[frozenset, Coord], List[Tuple[int, int]]]:
            # identisch zur Basis, aber nutzt self (= DefaultRoutingPlanner) Methoden
            attempt_full: Dict[int, List[TimedNode]] = {}
            meeting_of_pair: Dict[frozenset, Coord] = {}
            spill_pairs: List[Tuple[int, int]] = []

            layer_start_nodes: Set[Coord] = {current_pos[a] for a, _ in layer_pairs} | {current_pos[b] for _, b in layer_pairs}
            occupied_now: Set[Coord] = {current_pos[q] for q in qids_all}
            reserved_in_nodes: Set[Coord] = set()

            pair_infos = []
            for a_id, b_id in layer_pairs:
                qa_now, qb_now = qobj(a_id), qobj(b_id)
                cands = RoutingPlannerWithRerouting.meeting_candidates(
                    G,
                    qa_now.pos, qb_now.pos,
                    reserved_meetings=reserved_in_nodes,
                    occupied_nodes=(occupied_now | (layer_start_nodes - {qa_now.pos, qb_now.pos})),
                )
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
                cands = RoutingPlannerWithRerouting.meeting_candidates(
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
                        tmp_plans = RoutingPlannerWithRerouting.pairwise_mapf(
                            G,
                            try_objs,
                            fixed_meetings=try_fixed,
                            occupied_nodes=occupied_now | hard_blocked,
                            blocked_edges=defective_edges,  # defekte Kanten bereits meiden
                        )
                    except RuntimeError:
                        continue

                    preins_map = RoutingPlannerWithRerouting._preins_for_plans(tmp_plans, try_fixed)
                    if preins_map is None:
                        continue
                    if len(set(preins_map.values())) != len(preins_map):
                        continue

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

        def try_reroute_with_fixed_meetings(
            layer_pairs: List[Tuple[int, int]],
            fixed_meetings: Dict[frozenset, Coord],
        ) -> Optional[Tuple[Dict[int, List[TimedNode]], Dict[int, List[TimedNode]]]]:
            """
            Versucht für ALLE aktuell aktiven Paare dieses Layers eine neue, kollisionsfreie
            Routenplanung bis zum jeweiligen Meeting (fixiert!) zu erzeugen, die defekte Kanten meidet.
            Gibt (attempt_full, pre_in_paths) zurück oder None bei Scheitern.
            """
            if not fixed_meetings:
                return None

            occupied_now: Set[Coord] = {current_pos[q] for q in qids_all}
            layer_start_nodes: Set[Coord] = {current_pos[a] for a, _ in layer_pairs} | {current_pos[b] for _, b in layer_pairs}

            # Nur die Paare berücksichtigen, die tatsächlich ein fixes Meeting haben
            pairs_with_meeting = [(a, b) for (a, b) in layer_pairs if frozenset({a, b}) in fixed_meetings]
            if not pairs_with_meeting:
                return None

            try_objs = [(qobj(a), qobj(b)) for (a, b) in pairs_with_meeting]

            # Welche Startknoten sind "erlaubt" (d.h. zu diesen Paaren gehörend)?
            allowed: Set[Coord] = set()
            for (a, b) in pairs_with_meeting:
                allowed.add(current_pos[a])
                allowed.add(current_pos[b])

            # Alle anderen Layer-Startknoten als "hart blockiert" behandeln
            hard_blocked = layer_start_nodes - allowed

            try:
                new_plans = RoutingPlannerWithRerouting.pairwise_mapf(
                    G,
                    try_objs,
                    fixed_meetings=fixed_meetings,
                    occupied_nodes=occupied_now | hard_blocked,
                    blocked_edges=defective_edges,  # defekte Kanten strikt meiden
                )
            except RuntimeError:
                return None

            # PRE-INs prüfen & eindeutige PRE-INs fordern
            preins_map = RoutingPlannerWithRerouting._preins_for_plans(new_plans, fixed_meetings)
            if preins_map is None or len(set(preins_map.values())) != len(preins_map):
                return None

            # PRE-IN-Schnitte (noch ohne finale sync-time)
            pre_in_paths: Dict[int, List[TimedNode]] = {}
            for pkey, meet in fixed_meetings.items():
                a_id, b_id = tuple(pkey)
                for qid in (a_id, b_id):
                    cut = RoutingPlannerWithRerouting._retime_until_pre_in_wait(new_plans[qid], meet, 0)
                    if cut is None:
                        return None
                    pre_in_paths[qid] = cut

            return new_plans, pre_in_paths


        # ---------- Haupt-Loop über Layer mit Rerouting ----------
        idx = 0
        while idx < len(layers):
            layer_pairs = layers[idx]

            # 1) Kandidaten & Pfade (bis Meeting) – identisch zur Basis
            attempt_full, meeting_of_pair, spill_pairs = plan_layer_candidates(layer_pairs)

            if spill_pairs:
                layers[idx+1:idx+1] = [spill_pairs]

            # 1a) Evakuierungen wie Basis – nur planen, Sampling davor
            evac_plans, blockers_to_pairs, remaining_block_coords = RoutingPlannerWithRerouting._plan_evacuations(
                G, qubits, current_pos, layer_pairs, attempt_full, meeting_of_pair
            )

            # Sampling der defekten Kanten JETZT
            RoutingPlannerWithRerouting._sample_edge_failures(
                G,
                defective_edges,
                p_fail=(1.0 - p_success),
                p_repair=p_repair,
            )

            # Evakuierungspläne auf Defekte prüfen
            invalid_blockers: Set[int] = set()
            for qid, path in evac_plans.items():
                if RoutingPlannerWithRerouting._path_uses_defective_edge(path, defective_edges):
                    invalid_blockers.add(qid)

            newly_spilled_by_evac: Set[Tuple[int, int]] = set()
            for qid in invalid_blockers:
                for pkey in blockers_to_pairs.get(qid, set()):
                    a_id, b_id = tuple(pkey)
                    if pkey in meeting_of_pair:
                        newly_spilled_by_evac.add(tuple(sorted((a_id, b_id))))

            # gültige Evakuierungen ausführen
            if evac_plans:
                micro: Dict[int, List[TimedNode]] = {}
                for qid, path in evac_plans.items():
                    if qid in invalid_blockers:
                        continue
                    micro[qid] = path

                if micro:
                    batch_plans.append(micro)
                    for qid, path in micro.items():
                        current_pos[qid] = path[-1][0]
                    add_defect_snapshots(1)

            if remaining_block_coords:
                remaining_block_coords |= {current_pos[qid] for qid in evac_plans.keys() if qid in invalid_blockers}

            if newly_spilled_by_evac:
                for (a_id, b_id) in newly_spilled_by_evac:
                    attempt_full.pop(a_id, None)
                    attempt_full.pop(b_id, None)
                    meeting_of_pair.pop(frozenset({a_id, b_id}), None)
                layers[idx+1:idx+1] = [list(newly_spilled_by_evac)]

            # Blocker nicht vollständig räumbar → Paare spillen (wie Basis)
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

            # Wenn nach Evakuierung nichts mehr übrig ist → Wartebatch
            if not any(frozenset({a, b}) in meeting_of_pair for (a, b) in layer_pairs):
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                add_defect_snapshots(1)
                idx += 1
                continue

            # 1b) PRE-IN schneiden (vorläufig)
            pre_in_paths: Dict[int, List[TimedNode]] = {}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]
                for qid in (a_id, b_id):
                    cut = RoutingPlannerWithRerouting._retime_until_pre_in_wait(attempt_full[qid], meet, 0)
                    if cut is None:
                        attempt_full.pop(a_id, None)
                        attempt_full.pop(b_id, None)
                        meeting_of_pair.pop(pkey, None)
                        break
                    pre_in_paths[qid] = cut

            if not pre_in_paths:
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                add_defect_snapshots(1)
                idx += 1
                continue

            # --- Prüfung nach Sampling; bei Ungültigkeit erst REROUTING versuchen ---
            def pair_invalid(a_id: int, b_id: int, meet: Coord) -> bool:
                pa = pre_in_paths.get(a_id)
                pb = pre_in_paths.get(b_id)
                if pa is None or pb is None:
                    return True
                if RoutingPlannerWithRerouting._path_uses_defective_edge(pa, defective_edges):
                    return True
                if RoutingPlannerWithRerouting._path_uses_defective_edge(pb, defective_edges):
                    return True
                pre_a = pa[-1][0]
                pre_b = pb[-1][0]
                hop_edges = {frozenset({pre_a, meet}), frozenset({pre_b, meet})}
                return any(e in defective_edges for e in hop_edges)

            invalid_pairs: List[Tuple[int, int]] = []
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                if pair_invalid(a_id, b_id, meeting_of_pair[pkey]):
                    invalid_pairs.append((a_id, b_id))

            if invalid_pairs:
                # === REROUTING-VERSUCH mit fixen Meetings (Umfahrungen) ===
                # Wir versuchen, ALLE aktiven Paare (inkl. invaliden) neu zu planen.
                new_attempt = try_reroute_with_fixed_meetings(layer_pairs, meeting_of_pair)
                if new_attempt is not None:
                    attempt_full, pre_in_paths = new_attempt

                    # Nach Reroute erneut prüfen, welche Paare ggf. weiterhin ungültig sind
                    still_invalid: List[Tuple[int, int]] = []
                    for (a_id, b_id) in layer_pairs:
                        pkey = frozenset({a_id, b_id})
                        if pkey not in meeting_of_pair:
                            continue
                        if pair_invalid(a_id, b_id, meeting_of_pair[pkey]):
                            still_invalid.append((a_id, b_id))

                    # Nur die weiterhin unlösbaren Paare spillen
                    if still_invalid:
                        for (a_id, b_id) in still_invalid:
                            attempt_full.pop(a_id, None)
                            attempt_full.pop(b_id, None)
                            pre_in_paths.pop(a_id, None)
                            pre_in_paths.pop(b_id, None)
                            meeting_of_pair.pop(frozenset({a_id, b_id}), None)
                        layers[idx+1:idx+1] = [still_invalid]
                else:
                    # Reroute gescheitert → alle invaliden Paare spillen (Basisverhalten)
                    for (a_id, b_id) in invalid_pairs:
                        attempt_full.pop(a_id, None)
                        attempt_full.pop(b_id, None)
                        pre_in_paths.pop(a_id, None)
                        pre_in_paths.pop(b_id, None)
                        meeting_of_pair.pop(frozenset({a_id, b_id}), None)
                    layers[idx+1:idx+1] = [invalid_pairs]

            # Falls nach (Re)Routing nichts mehr übrig: Wartebatch
            if not pre_in_paths:
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                add_defect_snapshots(1)
                idx += 1
                continue

            # 2) Synchronisieren wie Basis
            T_pre_sync = max(
                RoutingPlannerWithRerouting._first_pre_in_time(attempt_full[qid], meeting_of_pair[frozenset({a, b})])
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
                    cut = RoutingPlannerWithRerouting._retime_until_pre_in_wait(attempt_full[qid], meet, T_pre_sync)
                    micro_paths_approach[qid] = cut

            others = qids_all - layer_qids
            for qid in others:
                micro_paths_approach[qid] = [(current_pos[qid], 0), (current_pos[qid], T_pre_sync + 1)]

            batch_plans.append(micro_paths_approach)
            add_defect_snapshots(1)

            for qid in layer_qids:
                current_pos[qid] = micro_paths_approach[qid][-1][0]

            # 3) IN-Hop (erneut Hop-Kanten gegen Defekt prüfen)
            micro_in: Dict[int, List[TimedNode]] = {}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]

                pre_a = current_pos[a_id]
                pre_b = current_pos[b_id]
                if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                    micro_in[a_id] = [(pre_a, 0), (pre_a, 2)]
                    micro_in[b_id] = [(pre_b, 0), (pre_b, 2)]
                    continue

                micro_in[a_id] = [(pre_a, 0), (meet, 1), (pre_a, 2)]
                micro_in[b_id] = [(pre_b, 0), (meet, 1), (pre_b, 2)]

            for qid in (qids_all - {q for ab in layer_pairs if frozenset(ab) in meeting_of_pair for q in ab}):
                micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]

            batch_plans.append(micro_in)
            add_defect_snapshots(1)

            idx += 1

        return RoutingPlannerWithRerouting.stitch_batches(qubits, batch_plans, batch_defects)

