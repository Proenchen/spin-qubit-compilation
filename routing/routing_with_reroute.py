from __future__ import annotations

from typing import Dict, List, Tuple, Set

import networkx as nx
import random

from routing.common import Coord, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner, P_REPAIR, P_SUCCESS


class RoutingPlannerWithRerouting(DefaultRoutingPlanner):

    @staticmethod
    def _preins_from_paths_for_pairs(
        pre_in_paths: Dict[int, List[TimedNode]],
        meeting_of_pair: Dict[frozenset, Coord],
    ) -> Set[Coord]:
        """Liefert die Menge der bereits belegten PRE-IN-Knoten aus den (gültigen) pre_in_paths."""
        used: Set[Coord] = set()
        for pkey in meeting_of_pair:
            qids = list(pkey)
            for qid in qids:
                path = pre_in_paths.get(qid)
                if path:
                    used.add(path[-1][0])  # letzter Knoten = PRE-IN @ sync-time
        return used

    @staticmethod
    def _reserved_meetings_for_pairs(meeting_of_pair: Dict[frozenset, Coord]) -> Set[Coord]:
        return set(meeting_of_pair.values())

    @staticmethod
    def routing_with_reroute(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float = P_SUCCESS,
        p_repair: float = P_REPAIR,
    ):
        """
        Wie default_routing – mit dem Unterschied:
        Wenn nach dem Sampling eine für den Pfad benötigte Kante defekt ist,
        wird zunächst eine lokale Umfahrung (Re-Routing) versucht. Erst wenn
        das nicht möglich ist, erfolgt der Spillover.
        """

        # ---- Zustand / Hilfsstrukturen (wie in default_routing) ----
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

        # --- lokale Kopie der Layer-Planungshilfsfunktion (identisch zur default_routing) ---
        def plan_layer_candidates(
            layer_pairs: List[Tuple[int, int]],
        ) -> Tuple[Dict[int, List[TimedNode]], Dict[frozenset, Coord], List[Tuple[int, int]]]:
            attempt_full: Dict[int, List[TimedNode]] = {}
            meeting_of_pair: Dict[frozenset, Coord] = {}
            spill_pairs: List[Tuple[int, int]] = []

            layer_start_nodes: Set[Coord] = {current_pos[a] for a, _ in layer_pairs} | {current_pos[b] for _, b in layer_pairs}
            occupied_now: Set[Coord] = {current_pos[q] for q in qids_all}
            reserved_in_nodes: Set[Coord] = set()

            pair_infos = []
            for a_id, b_id in layer_pairs:
                qa_now, qb_now = qobj(a_id), qobj(b_id)
                cands = DefaultRoutingPlanner.meeting_candidates(
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
                            blocked_edges=defective_edges,  # defekte Kanten strikt meiden
                        )
                    except RuntimeError:
                        continue

                    preins_map = DefaultRoutingPlanner._preins_for_plans(tmp_plans, try_fixed)
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

        # ---------- Haupt-Loop über Layer (wie default_routing, mit Reroute-Schritt) ----------
        idx = 0
        while idx < len(layers):
            layer_pairs = layers[idx]

            attempt_full, meeting_of_pair, spill_pairs = plan_layer_candidates(layer_pairs)
            if spill_pairs:
                layers[idx+1:idx+1] = [spill_pairs]

            prev_batches = len(batch_plans)
            remaining_block_coords = DefaultRoutingPlanner._evacuate_blocking_non_layer(
                G, qubits, current_pos,
                layer_pairs,
                attempt_full,
                meeting_of_pair,
                batch_plans
            )
            add_defect_snapshots(len(batch_plans) - prev_batches)

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

            if not any(frozenset({a, b}) in meeting_of_pair for (a, b) in layer_pairs):
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                add_defect_snapshots(1)
                idx += 1
                continue

            # PRE-IN Schnitt vor Sampling
            pre_in_paths: Dict[int, List[TimedNode]] = {}
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]
                for qid in (a_id, b_id):
                    cut = DefaultRoutingPlanner._retime_until_pre_in_wait(attempt_full[qid], meet, 0)
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

            # --- Sampling der Kantenfehler ---
            DefaultRoutingPlanner._sample_edge_failures(
                G, defective_edges, p_fail=(1.0 - p_success), p_repair=p_repair
            )

            # Prüfen, welche Paare invalid geworden sind
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

            invalid_pairs: List[Tuple[int, int]] = []
            for (a_id, b_id) in layer_pairs:
                pkey = frozenset({a_id, b_id})
                if pkey not in meeting_of_pair:
                    continue
                meet = meeting_of_pair[pkey]
                if pair_invalid(a_id, b_id, meet):
                    invalid_pairs.append((a_id, b_id))

            # --------- NEU: REROUTE-VERSUCH für ungültige Paare ---------
            if invalid_pairs:
                # Reserven der weiterhin gültigen Paare sichern
                reserved_meetings_ok: Set[Coord] = DefaultRoutingPlanner._reserved_meetings_for_pairs(
                    {p: m for p, m in meeting_of_pair.items() if tuple(sorted(p)) not in [tuple(sorted(ip)) for ip in invalid_pairs]}
                )
                reserved_preins_ok: Set[Coord] = DefaultRoutingPlanner._preins_from_paths_for_pairs(
                    {qid: pre_in_paths[qid] for (a, b) in layer_pairs for qid in (a, b)
                     if frozenset({a, b}) in meeting_of_pair and (a, b) not in invalid_pairs and qid in pre_in_paths},
                    {p: m for p, m in meeting_of_pair.items() if tuple(sorted(p)) not in [tuple(sorted(ip)) for ip in invalid_pairs]}
                )

                occupied_now: Set[Coord] = {current_pos[q] for q in qids_all}
                layer_start_nodes: Set[Coord] = {current_pos[a] for a, _ in layer_pairs} | {current_pos[b] for _, b in layer_pairs}

                salvaged: Set[Tuple[int, int]] = set()
                for (a_id, b_id) in invalid_pairs:
                    qa_now, qb_now = qobj(a_id), qobj(b_id)

                    # Kandidaten unter Meidung bereits reservierter Meetings
                    cands = DefaultRoutingPlanner.meeting_candidates(
                        G,
                        qa_now.pos, qb_now.pos,
                        reserved_meetings=reserved_meetings_ok,
                        occupied_nodes=(occupied_now | (layer_start_nodes - {qa_now.pos, qb_now.pos})),
                    )
                    rerouted = False
                    for mnode in cands:
                        try_fixed = {k: v for k, v in meeting_of_pair.items() if k != frozenset({a_id, b_id})}
                        try_fixed[frozenset({a_id, b_id})] = mnode

                        # Blockiere alle Layer-Startknoten außer den beiden Starts dieses Paars
                        allowed = {current_pos[a_id], current_pos[b_id]}
                        hard_blocked = (layer_start_nodes - allowed)

                        # Zusätzlich: meide defekte Kanten; meide bereits belegte PRE-INs
                        try:
                            tmp_plans = DefaultRoutingPlanner.pairwise_mapf(
                                G,
                                [(qa_now, qb_now)],
                                fixed_meetings={frozenset({a_id, b_id}): mnode},
                                occupied_nodes=occupied_now | hard_blocked | reserved_preins_ok,
                                blocked_edges=defective_edges,
                            )
                        except RuntimeError:
                            continue

                        # PRE-INs ermitteln und prüfen, dass sie nicht mit bestehenden kollidieren
                        preins_map = DefaultRoutingPlanner._preins_for_plans(
                            tmp_plans, {frozenset({a_id, b_id}): mnode}
                        )
                        if preins_map is None:
                            continue
                        if any(pin in reserved_preins_ok for pin in preins_map.values()):
                            continue

                        # Erfolgreich: ersetze Pfade & Meeting
                        attempt_full.update(tmp_plans)
                        meeting_of_pair[frozenset({a_id, b_id})] = mnode
                        # PRE-IN Schnitte neu für dieses Paar (sync_time=0)
                        for qid in (a_id, b_id):
                            pre_in_paths[qid] = DefaultRoutingPlanner._retime_until_pre_in_wait(
                                attempt_full[qid], mnode, 0
                            )
                        # Reserves aktualisieren
                        reserved_meetings_ok.add(mnode)
                        reserved_preins_ok |= set(preins_map.values())
                        rerouted = True
                        salvaged.add((a_id, b_id))
                        break

                    if not rerouted:
                        # bleibt invalid → später Spillover
                        pass

                # Übriggebliebene (nicht gerettete) invalid_pairs → Spillover
                newly_spilled_after_sampling: List[Tuple[int, int]] = [
                    (a_id, b_id) for (a_id, b_id) in invalid_pairs if (a_id, b_id) not in salvaged
                ]
                if newly_spilled_after_sampling:
                    for (a_id, b_id) in newly_spilled_after_sampling:
                        attempt_full.pop(a_id, None)
                        attempt_full.pop(b_id, None)
                        meeting_of_pair.pop(frozenset({a_id, b_id}), None)
                        pre_in_paths.pop(a_id, None)
                        pre_in_paths.pop(b_id, None)
                    layers[idx+1:idx+1] = [newly_spilled_after_sampling]

            # Wenn jetzt nichts mehr übrig ist → Wartebatch
            if not pre_in_paths:
                wait_batch: Dict[int, List[TimedNode]] = {}
                for q in qubits:
                    qid = q.id
                    wait_batch[qid] = [(current_pos[qid], 0), (current_pos[qid], 1)]
                batch_plans.append(wait_batch)
                add_defect_snapshots(1)
                idx += 1
                continue

            # Synchronisation & Ausführung (wie default_routing)
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

            others = qids_all - layer_qids
            for qid in others:
                micro_paths_approach[qid] = [(current_pos[qid], 0), (current_pos[qid], T_pre_sync + 1)]

            batch_plans.append(micro_paths_approach)
            add_defect_snapshots(1)

            for qid in layer_qids:
                current_pos[qid] = micro_paths_approach[qid][-1][0]

            # kurzer IN-Hop
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

        return DefaultRoutingPlanner.stitch_batches(qubits, batch_plans, batch_defects)
