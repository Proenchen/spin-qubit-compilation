from __future__ import annotations

import random
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from routing.common import AStar, Coord, MAX_TIME, Reservations, TimedNode, Qubit
from routing.routing_strategy import RoutingStrategy

MAX_REPLANS = 50
MAX_GLOBAL_ITERS = 50


class DefaultRoutingPlanner(RoutingStrategy):
    def route(
        self,
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float,
        p_repair: float,
    ):
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}
        all_qids: Set[int] = {q.id for q in qubits}

        defective_edges: Set[frozenset] = set()
        batch_plans: List[Dict[int, List[TimedNode]]] = []
        batch_defects: List[Set[frozenset]] = []

        total_ins: Set[Coord] = {n for n in G if G.nodes[n].get("type") == "IN"}
        tried_meetings: Dict[frozenset, Set[Coord]] = {}

        layers = self._build_layers(pairs)

        idx = 0
        replan_counts: Dict[int, int] = {}
        global_iter = 0

        while idx < len(layers):
            global_iter += 1
            if global_iter > MAX_GLOBAL_ITERS:
                raise RuntimeError(
                    f"Abbruch (Safeguard): mehr als {MAX_GLOBAL_ITERS} Iterationen "
                    f"im Routing-Hauptloop (mögliche Dauerschleife, aktueller Layer = {idx})."
                )

            tried_meetings.clear()

            layer_pairs = layers[idx]
            layer_qids: Set[int] = {x for ab in layer_pairs for x in ab}
            non_layer_qids: Set[int] = all_qids - layer_qids
            layer_starts: Set[Coord] = {current_pos[q] for q in layer_qids}
            occupied_now: Set[Coord] = {current_pos[q] for q in all_qids}

            replan_current_layer = False

            (
                to_meeting_plans,
                fixed_meetings,
                _,
                unplaceable_pairs_step1,
                exhausted_pairs_step1,
            ) = DefaultRoutingPlanner._plan_layer_only(
                G=G,
                current_pos=current_pos,
                layer_pairs=layer_pairs,
                layer_starts=layer_starts,
                defective_edges=defective_edges,
                banned_meetings=tried_meetings,
                all_ins=total_ins,
            )

            if exhausted_pairs_step1:
                layers[idx + 1 : idx + 1] = [exhausted_pairs_step1]

            if unplaceable_pairs_step1:
                layers[idx + 1 : idx + 1] = [unplaceable_pairs_step1]
                if not fixed_meetings:
                    wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                    batch_plans.append(wait)
                    DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                    idx += 1
                    continue

            if not fixed_meetings and not unplaceable_pairs_step1 and not exhausted_pairs_step1:
                raise RuntimeError(
                    f"Layer {idx} unlösbar: keine Meeting-INs fixiert und kein Spillover möglich. "
                    f"Paare: {layer_pairs}"
                )

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            F_layer: Set[Coord] = DefaultRoutingPlanner._collect_layer_nodes(to_meeting_plans, fixed_meetings)
            F_all = set(F_layer) | set(layer_starts)

            blockers_now: List[int] = [qid for qid in non_layer_qids if current_pos[qid] in F_all]

            blocker_to_pair: Dict[int, Tuple[int, int]] = {}
            evac_plans: Dict[int, List[TimedNode]] = {}

            if blockers_now:
                node_to_pairs: Dict[Coord, List[Tuple[int, int]]] = {}
                for (a, b) in layer_pairs:
                    key = frozenset({a, b})
                    if key not in fixed_meetings:
                        continue
                    nodes = DefaultRoutingPlanner._path_nodes_of_pair(to_meeting_plans, a, b)
                    for n in nodes:
                        node_to_pairs.setdefault(n, []).append((a, b))

                for qid in blockers_now:
                    pos = current_pos[qid]
                    pairs_touching = node_to_pairs.get(pos, [])
                    if pairs_touching:
                        blocker_to_pair[qid] = pairs_touching[0]

                avoid_for_targets = set(occupied_now) | F_all
                targets: Dict[int, Coord] = {}
                for qid in blockers_now:
                    tgt = DefaultRoutingPlanner._nearest_free_sn(G, current_pos[qid], avoid_for_targets)
                    if tgt is not None and tgt not in F_layer:
                        targets[qid] = tgt
                        avoid_for_targets.add(tgt)

                cannot_place = [qid for qid in blockers_now if qid not in targets]
                newly_affected: List[Tuple[int, int]] = []
                for qid in cannot_place:
                    ab = blocker_to_pair.get(qid)
                    if ab:
                        newly_affected.append(ab)

                if newly_affected:
                    unique_pairs: List[Tuple[int, int]] = []
                    seen_pairs: Set[frozenset] = set()
                    for ab in newly_affected:
                        pkey = frozenset(ab)
                        if pkey in seen_pairs:
                            continue
                        seen_pairs.add(pkey)
                        unique_pairs.append(ab)
                        to_meeting_plans.pop(ab[0], None)
                        to_meeting_plans.pop(ab[1], None)
                        fixed_meetings.pop(pkey, None)
                    replan_current_layer = True

                evacuating = {qid: current_pos[qid] for qid in blockers_now if qid in targets}
                if evacuating:
                    try:
                        evac_plans = DefaultRoutingPlanner._mapf_to_targets(
                            G=G,
                            starts=evacuating,
                            targets={qid: targets[qid] for qid in evacuating},
                            blocked_nodes=F_all,
                            blocked_edges=defective_edges,
                        )
                    except RuntimeError:
                        evac_plans = {}
                        blocked_now = set(F_all)
                        for qid in evacuating:
                            try:
                                one = DefaultRoutingPlanner._mapf_to_targets(
                                    G=G,
                                    starts={qid: current_pos[qid]},
                                    targets={qid: targets[qid]},
                                    blocked_nodes=blocked_now,
                                    blocked_edges=defective_edges,
                                )
                                evac_plans[qid] = one[qid]
                                blocked_now.add(one[qid][-1][0])
                            except RuntimeError:
                                ab = blocker_to_pair.get(qid)
                                if ab:
                                    pkey = frozenset(ab)
                                    to_meeting_plans.pop(ab[0], None)
                                    to_meeting_plans.pop(ab[1], None)
                                    fixed_meetings.pop(pkey, None)
                                    replan_current_layer = True

                if evac_plans:
                    waiting_qids = non_layer_qids - set(evacuating.keys())
                    evac_plans = DefaultRoutingPlanner._resolve_evacuate_collisions_with_waiters(
                        G=G,
                        evac_plans=evac_plans,
                        targets={qid: targets[qid] for qid in evacuating},
                        current_pos=current_pos,
                        waiting_qids=waiting_qids,
                        blocked_nodes=F_all,
                        defective_edges=defective_edges,
                        blocker_to_pair=blocker_to_pair,
                    )

            if replan_current_layer:
                replan_counts[idx] = replan_counts.get(idx, 0) + 1
                if replan_counts[idx] > MAX_REPLANS:
                    raise RuntimeError(
                        f"Kein gültiges Routing für Layer {idx} nach {replan_counts[idx]} Neuplanungen."
                    )
                continue

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            DefaultRoutingPlanner._sample_edge_failures(
                G, defective_edges, p_fail=(1.0 - p_success), p_repair=p_repair
            )

            if evac_plans:
                to_spill_for_nonlayer: Set[Tuple[int, int]] = set()
                for qid, path in evac_plans.items():
                    if DefaultRoutingPlanner._path_uses_defective_edge(path, defective_edges):
                        ab = blocker_to_pair.get(qid)
                        if ab:
                            to_spill_for_nonlayer.add(ab)

                if to_spill_for_nonlayer:
                    for (a, b) in to_spill_for_nonlayer:
                        key = frozenset({a, b})
                        to_meeting_plans.pop(a, None)
                        to_meeting_plans.pop(b, None)
                        fixed_meetings.pop(key, None)
                    layers[idx + 1 : idx + 1] = [list(to_spill_for_nonlayer)]
                    evac_plans.clear()

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            pre_in_paths: Dict[int, List[TimedNode]] = {}
            T_pre_sync = 0

            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]

                for qid in (a, b):
                    cut = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[qid], meet, 0)
                    if cut is None:
                        to_meeting_plans.pop(a, None)
                        to_meeting_plans.pop(b, None)
                        fixed_meetings.pop(key, None)
                        replan_current_layer = True
                        break

                    pre_in_paths[qid] = cut
                    if cut:
                        T_pre_sync = max(T_pre_sync, cut[-1][1])

            if replan_current_layer:
                continue

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            to_spill_layer_defects: List[Tuple[int, int]] = []
            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue

                meet = fixed_meetings[key]
                pa = pre_in_paths.get(a)
                pb = pre_in_paths.get(b)

                if pa is None or pb is None:
                    to_spill_layer_defects.append((a, b))
                    continue

                if DefaultRoutingPlanner._path_uses_defective_edge(pa, defective_edges):
                    to_spill_layer_defects.append((a, b))
                    continue

                if DefaultRoutingPlanner._path_uses_defective_edge(pb, defective_edges):
                    to_spill_layer_defects.append((a, b))
                    continue

                pre_a = pa[-1][0]
                pre_b = pb[-1][0]
                if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                    to_spill_layer_defects.append((a, b))

            if to_spill_layer_defects:
                for (a, b) in to_spill_layer_defects:
                    key = frozenset({a, b})
                    to_meeting_plans.pop(a, None)
                    to_meeting_plans.pop(b, None)
                    fixed_meetings.pop(key, None)
                layers[idx + 1 : idx + 1] = [to_spill_layer_defects]

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            if evac_plans:
                evac_plans = {
                    qid: path
                    for qid, path in evac_plans.items()
                    if not DefaultRoutingPlanner._path_uses_defective_edge(path, defective_edges)
                }

                micro_evacuate: Dict[int, List[TimedNode]] = {qid: path for qid, path in evac_plans.items()}
                dur = max((p[-1][1] for p in micro_evacuate.values()), default=0)

                for qid in (all_qids - set(micro_evacuate.keys())):
                    micro_evacuate[qid] = [(current_pos[qid], 0), (current_pos[qid], dur)]

                batch_plans.append(micro_evacuate)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

                for qid in evac_plans:
                    current_pos[qid] = evac_plans[qid][-1][0]

            micro_to_pre: Dict[int, List[TimedNode]] = {}
            exec_layer_qids: Set[int] = set()

            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]

                for qid in (a, b):
                    cut = DefaultRoutingPlanner._retime_until_pre_in_wait(
                        to_meeting_plans[qid], meet, T_pre_sync
                    )
                    micro_to_pre[qid] = cut
                    exec_layer_qids.add(qid)

            others = all_qids - exec_layer_qids
            dur_pre = max((p[-1][1] for p in micro_to_pre.values()), default=0)
            for qid in others:
                micro_to_pre[qid] = [(current_pos[qid], 0), (current_pos[qid], dur_pre)]

            batch_plans.append(micro_to_pre)
            DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

            for qid in exec_layer_qids:
                current_pos[qid] = micro_to_pre[qid][-1][0]

            micro_in: Dict[int, List[TimedNode]] = {}
            active_layer_qids = {q for ab in layer_pairs if frozenset(ab) in fixed_meetings for q in ab}

            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue

                meet = fixed_meetings[key]
                pre_a = current_pos[a]
                pre_b = current_pos[b]

                if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                    micro_in[a] = [(pre_a, 0), (pre_a, 2)]
                    micro_in[b] = [(pre_b, 0), (pre_b, 2)]
                else:
                    micro_in[a] = [(pre_a, 0), (meet, 1), (pre_a, 2)]
                    micro_in[b] = [(pre_b, 0), (meet, 1), (pre_b, 2)]

            for qid in (all_qids - active_layer_qids):
                micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]

            batch_plans.append(micro_in)
            DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

            idx += 1

        return DefaultRoutingPlanner.stitch_batches(qubits, batch_plans, batch_defects)

    @staticmethod
    def _build_layers(pairs: List[Tuple[Qubit, Qubit]]) -> List[List[Tuple[int, int]]]:
        layers: List[List[Tuple[int, int]]] = []
        used: Set[int] = set()
        cur: List[Tuple[int, int]] = []

        for qa, qb in pairs:
            a, b = qa.id, qb.id
            if a not in used and b not in used:
                cur.append((a, b))
                used |= {a, b}
            else:
                if cur:
                    layers.append(cur)
                cur = [(a, b)]
                used = {a, b}

        if cur:
            layers.append(cur)

        return layers

    @staticmethod
    def _snapshot_defects(
        batch_defects: List[Set[frozenset]],
        defective_edges: Set[frozenset],
        n: int,
    ) -> None:
        for _ in range(n):
            batch_defects.append(set(defective_edges))

    @staticmethod
    def _forbidden_for_layer_qid(
        current_pos: Dict[int, Coord],
        layer_starts: Set[Coord],
        qid: int,
    ) -> Set[Coord]:
        return set(layer_starts) - {current_pos[qid]}

    @staticmethod
    def _chebyshev(a: Coord, b: Coord) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def _build_blocker_chain_along_path(
        path: List[TimedNode],
        wait_pos: Dict[int, Coord],
        blocker_to_pair: Dict[int, Tuple[int, int]],
        root_pair: Optional[Tuple[int, int]],
    ) -> List[int]:
        chain: List[int] = []
        seen: Set[int] = set()

        for (_, _), (node_next, _) in zip(path[:-1], path[1:]):
            for bqid, bpos in wait_pos.items():
                if bqid in seen:
                    continue
                if node_next == bpos:
                    chain.append(bqid)
                    seen.add(bqid)
                    if root_pair is not None:
                        blocker_to_pair.setdefault(bqid, root_pair)

        return chain

    @staticmethod
    def _block_non_sn_nodes(
        res: Reservations,
        G: nx.Graph,
        allowed: Set[Coord],
    ) -> None:
        for node in G.nodes:
            if node in allowed:
                continue
            if G.nodes[node].get("type") != "SN":
                cap = res.node_capacity(node)
                for t in range(0, MAX_TIME + 1):
                    res.node_caps[node][t] = cap

    @staticmethod
    def _plan_layer_only(
        G: nx.Graph,
        current_pos: Dict[int, Coord],
        layer_pairs: List[Tuple[int, int]],
        layer_starts: Set[Coord],
        defective_edges: Set[frozenset],
        banned_meetings: Optional[Dict[frozenset, Set[Coord]]] = None,
        all_ins: Optional[Set[Coord]] = None,
    ) -> Tuple[
        Dict[int, List[TimedNode]],
        Dict[frozenset, Coord],
        bool,
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ]:
        res = Reservations(G, blocked_edges=defective_edges)
        plans: Dict[int, List[TimedNode]] = {}
        fixed_meetings: Dict[frozenset, Coord] = {}
        unplaceable: List[Tuple[int, int]] = []

        banned_meetings = banned_meetings or {}
        all_ins = all_ins or set()
        exhausted_pairs: List[Tuple[int, int]] = []

        cand_per_pair: Dict[Tuple[int, int], List[Coord]] = {}
        for (a, b) in layer_pairs:
            qa, qb = current_pos[a], current_pos[b]
            cand_per_pair[(a, b)] = DefaultRoutingPlanner._best_meeting_candidates(
                G, qa, qb, reserved=set(), forbidden_nodes=set()
            )

        order = sorted(layer_pairs, key=lambda ab: len(cand_per_pair.get(ab, [])))
        reserved_in: Set[Coord] = set()

        for (a, b) in order:
            qa, qb = current_pos[a], current_pos[b]
            placed = False

            cur_preins_map = DefaultRoutingPlanner._preins_for_plans(plans, fixed_meetings)
            existing_preins: Set[Coord] = set(cur_preins_map.values()) if cur_preins_map else set()

            cands_all = DefaultRoutingPlanner._best_meeting_candidates(
                G, qa, qb, reserved=reserved_in, forbidden_nodes=set()
            )
            banned = banned_meetings.get(frozenset({a, b}), set())
            cands = [c for c in cands_all if c not in banned]

            existing_path_nodes: Set[Coord] = set()
            for p in plans.values():
                for c, _ in p:
                    existing_path_nodes.add(c)

            if not cands and all_ins and len(banned) >= len(all_ins):
                exhausted_pairs.append((a, b))

            for meet in cands:
                res_try = deepcopy(res)

                allowed_a = {qa, meet}
                allowed_b = {qb, meet}
                DefaultRoutingPlanner._block_non_sn_nodes(res_try, G, allowed=allowed_a | allowed_b)

                if existing_preins:
                    DefaultRoutingPlanner._block_nodes(res_try, existing_preins)

                DefaultRoutingPlanner._block_nodes(
                    res_try,
                    DefaultRoutingPlanner._forbidden_for_layer_qid(current_pos, layer_starts, a),
                )
                pa = AStar.search(G, qa, meet, res_try)
                if pa is None:
                    continue
                Reservations.commit(res_try, pa)

                pre_a = DefaultRoutingPlanner._entry_sn_from_path(pa, meet)
                if pre_a is None:
                    continue
                if pre_a in existing_path_nodes:
                    continue

                DefaultRoutingPlanner._block_nodes(res_try, {pre_a})

                if existing_preins:
                    DefaultRoutingPlanner._block_nodes(res_try, existing_preins)

                DefaultRoutingPlanner._block_nodes(
                    res_try,
                    DefaultRoutingPlanner._forbidden_for_layer_qid(current_pos, layer_starts, b),
                )
                pb = AStar.search(G, qb, meet, res_try)
                if pb is None:
                    continue

                pre_b = DefaultRoutingPlanner._entry_sn_from_path(pb, meet)
                if pre_b is None:
                    continue
                if pre_b in existing_path_nodes:
                    continue

                tmp_plans = dict(plans)
                tmp_plans[a] = pa
                tmp_plans[b] = pb

                tmp_fixed = dict(fixed_meetings)
                tmp_fixed[frozenset({a, b})] = meet

                preins = DefaultRoutingPlanner._preins_for_plans(tmp_plans, tmp_fixed)
                if preins is None:
                    continue

                layer_ids = {x for ab in layer_pairs for x in ab}
                prein_vals = [preins[qid] for qid in preins if qid in layer_ids]
                if len(set(prein_vals)) != len(prein_vals):
                    continue

                Reservations.commit(res_try, pb)

                res = res_try
                plans[a] = pa
                plans[b] = pb
                fixed_meetings[frozenset({a, b})] = meet
                reserved_in.add(meet)
                placed = True
                break

            if not placed:
                unplaceable.append((a, b))

        preins_ok = True
        preins = DefaultRoutingPlanner._preins_for_plans(plans, fixed_meetings)
        if preins is None:
            preins_ok = False
        else:
            layer_ids = {x for ab in layer_pairs for x in ab}
            lv = [preins[qid] for qid in preins if qid in layer_ids]
            preins_ok = (len(set(lv)) == len(lv))

        return plans, fixed_meetings, preins_ok, unplaceable, exhausted_pairs

    @staticmethod
    def _path_nodes_of_pair(plans: Dict[int, List[TimedNode]], a: int, b: int) -> Set[Coord]:
        s: Set[Coord] = set()
        for qid in (a, b):
            for c, _ in plans.get(qid, []):
                s.add(c)
        return s

    @staticmethod
    def _collect_layer_nodes(
        plans: Dict[int, List[TimedNode]],
        fixed_meetings: Dict[frozenset, Coord],
    ) -> Set[Coord]:
        S: Set[Coord] = set()
        for p in plans.values():
            for c, _ in p:
                S.add(c)
        for m in fixed_meetings.values():
            S.add(m)
        return S

    @staticmethod
    def _best_meeting_candidates(
        G: nx.Graph,
        q0: Coord,
        q1: Coord,
        reserved: Optional[Set[Coord]] = None,
        forbidden_nodes: Optional[Set[Coord]] = None,
    ) -> List[Coord]:
        reserved = reserved or set()
        forbidden_nodes = forbidden_nodes or set()

        d0 = nx.single_source_shortest_path_length(G, q0)
        d1 = nx.single_source_shortest_path_length(G, q1)
        ins = [n for n in G if G.nodes[n]["type"] == "IN" and n in d0 and n in d1]

        cands = [n for n in ins if n not in reserved and n not in forbidden_nodes]
        cands.sort(key=lambda n: (d0[n] + d1[n], max(d0[n], d1[n]), abs(d0[n] - d1[n]), n))
        return cands

    @staticmethod
    def _entry_sn_from_path(path: List[TimedNode], meeting: Coord) -> Optional[Coord]:
        first_meet_idx = None
        for i, (c, _) in enumerate(path):
            if c == meeting:
                first_meet_idx = i
                break
        if first_meet_idx is None or first_meet_idx == 0:
            return None
        return path[first_meet_idx - 1][0]

    @staticmethod
    def _preins_for_plans(
        plans: Dict[int, List[TimedNode]],
        fixed_meetings: Dict[frozenset, Coord],
    ) -> Optional[Dict[int, Coord]]:
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
        if not path:
            return None

        first_meet_idx = None
        for i, (c, _) in enumerate(path):
            if c == meeting:
                first_meet_idx = i
                break

        if first_meet_idx is None:
            return None

        if first_meet_idx == 0:
            start_node, start_t = path[0]
            new_path = [(start_node, start_t)]
            cur = start_t
            while cur < sync_time:
                new_path.append((start_node, cur + 1))
                cur += 1
            return new_path

        pre_in, t_pre = path[first_meet_idx - 1]
        new_path = path[:first_meet_idx]
        cur = t_pre
        while cur < sync_time:
            new_path.append((pre_in, cur + 1))
            cur += 1
        return new_path

    @staticmethod
    def _mapf_to_targets(
        G: nx.Graph,
        starts: Dict[int, Coord],
        targets: Dict[int, Coord],
        blocked_nodes: Optional[Set[Coord]] = None,
        blocked_edges: Optional[Set[frozenset]] = None,
    ) -> Dict[int, List[TimedNode]]:
        if not starts:
            return {}

        res = Reservations(G, blocked_edges=blocked_edges or set())
        blocked_nodes = blocked_nodes or set()

        allowed = set(starts.values()) | set(targets.values())
        DefaultRoutingPlanner._block_non_sn_nodes(res, G, allowed=allowed)

        for node in blocked_nodes:
            cap = res.node_capacity(node)
            for t in range(0, MAX_TIME + 1):
                res.node_caps[node][t] = cap

        plans: Dict[int, List[TimedNode]] = {}

        for _, s in starts.items():
            res.occupy_node(s, 0)

        order = sorted(
            starts.keys(),
            key=lambda q: DefaultRoutingPlanner._chebyshev(starts[q], targets[q]),
            reverse=True,
        )

        for qid in order:
            s, t = starts[qid], targets[qid]
            path = AStar.search(G, s, t, res)
            if path is None:
                raise RuntimeError(f"Return routing failed for qubit {qid} from {s} -> {t}")
            res.commit(path)
            plans[qid] = path

        return plans

    @staticmethod
    def _sample_edge_failures(
        G: nx.Graph,
        defective_edges: Set[frozenset],
        p_fail: float,
        p_repair: float,
    ) -> None:
        for u, v in G.edges():
            e = frozenset({u, v})
            if e in defective_edges:
                if random.random() < p_repair:
                    defective_edges.discard(e)
            else:
                if random.random() < p_fail:
                    defective_edges.add(e)

    @staticmethod
    def _path_uses_defective_edge(path: List[TimedNode], defective_edges: Set[frozenset]) -> bool:
        for (u, _), (v, _) in zip(path[:-1], path[1:]):
            if u != v and frozenset({u, v}) in defective_edges:
                return True
        return False

    @staticmethod
    def _nearest_free_sn(G: nx.Graph, source: Coord, avoid: Set[Coord]) -> Optional[Coord]:
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
    def _block_nodes(res: Reservations, nodes: Set[Coord]) -> None:
        for node in nodes:
            cap = res.node_capacity(node)
            for t in range(0, MAX_TIME + 1):
                res.node_caps[node][t] = cap

    @staticmethod
    def _resolve_evacuate_collisions_with_waiters(
        G: nx.Graph,
        evac_plans: Dict[int, List[TimedNode]],
        targets: Dict[int, Coord],
        current_pos: Dict[int, Coord],
        waiting_qids: Set[int],
        blocked_nodes: Set[Coord],
        defective_edges: Set[frozenset],
        blocker_to_pair: Dict[int, Tuple[int, int]],
    ) -> Dict[int, List[TimedNode]]:
        if not evac_plans:
            return {}

        plans = dict(evac_plans)
        wait_pos: Dict[int, Coord] = {qid: current_pos[qid] for qid in waiting_qids}
        targets_local: Dict[int, Coord] = dict(targets)

        changed = True
        while changed:
            changed = False

            for mover_qid, mover_path in list(plans.items()):
                root_pair = blocker_to_pair.get(mover_qid)
                chain_blockers = DefaultRoutingPlanner._build_blocker_chain_along_path(
                    mover_path,
                    wait_pos=wait_pos,
                    blocker_to_pair=blocker_to_pair,
                    root_pair=root_pair,
                )
                if not chain_blockers:
                    continue

                chain_agents: List[int] = [mover_qid] + chain_blockers
                agents_to_plan: Set[int] = set(plans.keys()) | set(chain_blockers)

                starts: Dict[int, Coord] = {qid: current_pos[qid] for qid in agents_to_plan}
                new_targets_full: Dict[int, Coord] = {
                    qid: targets_local[qid] for qid in agents_to_plan if qid in targets_local
                }

                valid_chain = True
                for i, agent in enumerate(chain_agents):
                    if i < len(chain_agents) - 1:
                        nxt = chain_agents[i + 1]
                        new_targets_full[agent] = current_pos[nxt]
                    else:
                        if mover_qid in targets_local:
                            new_targets_full[agent] = targets_local[mover_qid]
                        else:
                            valid_chain = False
                            break

                if valid_chain and agents_to_plan.issubset(new_targets_full.keys()):
                    try:
                        replanned = DefaultRoutingPlanner._mapf_to_targets(
                            G=G,
                            starts=starts,
                            targets=new_targets_full,
                            blocked_nodes=blocked_nodes,
                            blocked_edges=defective_edges,
                        )
                        for qid, path in replanned.items():
                            plans[qid] = path
                            targets_local[qid] = new_targets_full[qid]

                        for bqid in chain_blockers:
                            wait_pos.pop(bqid, None)

                        changed = True
                        break

                    except RuntimeError:
                        b1 = chain_blockers[0]
                        try:
                            partial = DefaultRoutingPlanner._mapf_to_targets(
                                G=G,
                                starts={mover_qid: current_pos[mover_qid], b1: current_pos[b1]},
                                targets={
                                    mover_qid: current_pos[b1],
                                    b1: targets_local.get(mover_qid, current_pos[b1]),
                                },
                                blocked_nodes=blocked_nodes,
                                blocked_edges=defective_edges,
                            )
                            plans[mover_qid] = partial[mover_qid]
                            plans[b1] = partial[b1]

                            targets_local[mover_qid] = current_pos[b1]
                            targets_local[b1] = targets_local.get(mover_qid, current_pos[b1])

                            wait_pos.pop(b1, None)
                            changed = True
                            break
                        except RuntimeError:
                            continue

        return plans

    @staticmethod
    def stitch_batches(
        qubits: List[Qubit],
        batch_plans: List[Dict[int, List[TimedNode]]],
        batch_defects: Optional[List[Set[frozenset]]] = None,
    ) -> Tuple[Dict[int, List[TimedNode]], List[Tuple[int, int, Set[frozenset]]]]:
        if not batch_plans:
            timelines = {q.id: [(q.pos, 0)] for q in qubits}
            return timelines, []

        durations: List[int] = []
        for plans in batch_plans:
            if not plans:
                durations.append(0)
                continue
            durations.append(max(p[-1][1] for p in plans.values()))

        initial_pos = {q.id: q.pos for q in qubits}
        timelines: Dict[int, List[TimedNode]] = {q.id: [(initial_pos[q.id], 0)] for q in qubits}
        edge_timebands: List[Tuple[int, int, Set[frozenset]]] = []

        t_offset = 0
        for b, plans in enumerate(batch_plans):
            batch_T = durations[b]

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
                    target_t = t_offset + batch_T
                    for tt in range(last_t + 1, target_t + 1):
                        timelines[qid].append((last_coord, tt))
                    continue

                local_path = plans[qid]
                shifted = [(c, t + t_offset) for (c, t) in local_path]

                if timelines[qid][-1] == shifted[0]:
                    timelines[qid].extend(shifted[1:])
                else:
                    timelines[qid].extend(shifted)

            defects = set(batch_defects[b]) if (batch_defects is not None and b < len(batch_defects)) else set()
            edge_timebands.append((t_offset, t_offset + batch_T, defects))

            t_offset += batch_T

        return timelines, edge_timebands
