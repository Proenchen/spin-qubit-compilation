from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from routing.common import Coord, MAX_TIME, Reservations, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner
from routing.routing_strategy import RoutingStrategy

MAX_REPLANS = 50
MAX_GLOBAL_ITERS = 50


class RerouteRoutingPlanner(DefaultRoutingPlanner, RoutingStrategy):
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

        layers = RerouteRoutingPlanner._build_layers_from_pairs(pairs)

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
                    batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                    RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                    idx += 1
                    continue

            if not fixed_meetings and not unplaceable_pairs_step1 and not exhausted_pairs_step1:
                raise RuntimeError(
                    f"Layer {idx} unlösbar: keine Meeting-INs fixiert und kein Spillover möglich. "
                    f"Paare: {layer_pairs}"
                )

            if not fixed_meetings:
                batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            F_layer: Set[Coord] = DefaultRoutingPlanner._collect_layer_nodes(to_meeting_plans, fixed_meetings)
            F_all = set(F_layer) | set(layer_starts)

            (
                replan_current_layer,
                to_meeting_plans,
                fixed_meetings,
                evac_plans,
                evac_targets,
                blocker_to_pair,
                blocked_nodes_evacs,
            ) = RerouteRoutingPlanner._plan_non_layer_evacuation(
                G=G,
                current_pos=current_pos,
                all_qids=all_qids,
                non_layer_qids=non_layer_qids,
                layer_pairs=layer_pairs,
                to_meeting_plans=to_meeting_plans,
                fixed_meetings=fixed_meetings,
                defective_edges=defective_edges,
                occupied_now=occupied_now,
                F_all=F_all,
                F_layer=F_layer,
                tried_meetings=tried_meetings,
            )

            if replan_current_layer:
                replan_counts[idx] = replan_counts.get(idx, 0) + 1
                if replan_counts[idx] > MAX_REPLANS:
                    raise RuntimeError(
                        f"Kein gültiges Routing für Layer {idx} nach {replan_counts[idx]} Neuplanungen."
                    )
                continue

            if not fixed_meetings:
                batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            DefaultRoutingPlanner._sample_edge_failures(
                G, defective_edges, p_fail=(1.0 - p_success), p_repair=p_repair
            )

            (
                to_meeting_plans,
                fixed_meetings,
                evac_plans,
            ) = RerouteRoutingPlanner._reroute_or_spill_after_defects(
                G=G,
                current_pos=current_pos,
                layer_pairs=layer_pairs,
                layer_starts=layer_starts,
                defective_edges=defective_edges,
                tried_meetings=tried_meetings,
                to_meeting_plans=to_meeting_plans,
                fixed_meetings=fixed_meetings,
                evac_plans=evac_plans,
                evac_targets=evac_targets,
                blocked_nodes_evacs=blocked_nodes_evacs,
                blocker_to_pair=blocker_to_pair,
                layers=layers,
                idx=idx,
            )

            if not fixed_meetings:
                batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            pre_in_paths, T_pre_sync = RerouteRoutingPlanner._compute_pre_in_paths(
                layer_pairs=layer_pairs,
                to_meeting_plans=to_meeting_plans,
                fixed_meetings=fixed_meetings,
                tried_meetings=tried_meetings,
            )

            if not fixed_meetings:
                batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            RerouteRoutingPlanner._execute_layer_batches(
                all_qids=all_qids,
                layer_pairs=layer_pairs,
                current_pos=current_pos,
                defective_edges=defective_edges,
                batch_plans=batch_plans,
                batch_defects=batch_defects,
                evac_plans=evac_plans,
                to_meeting_plans=to_meeting_plans,
                fixed_meetings=fixed_meetings,
                T_pre_sync=T_pre_sync,
                tried_meetings=tried_meetings,
                layers=layers,
                idx=idx,
            )

            idx += 1

        return DefaultRoutingPlanner.stitch_batches(qubits, batch_plans, batch_defects)

    @staticmethod
    def _build_layers_from_pairs(pairs: List[Tuple[Qubit, Qubit]]) -> List[List[Tuple[int, int]]]:
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
    def _wait_plan(all_qids: Set[int], current_pos: Dict[int, Coord], ticks: int) -> Dict[int, List[TimedNode]]:
        wait: Dict[int, List[TimedNode]] = {}
        for qid in all_qids:
            wait[qid] = [(current_pos[qid], 0), (current_pos[qid], ticks)]
        return wait

    @staticmethod
    def _mark_meeting_failed(
        tried_meetings: Dict[frozenset, Set[Coord]],
        pair: Tuple[int, int],
        meet: Coord,
    ) -> None:
        key = frozenset(pair)
        tried_meetings.setdefault(key, set()).add(meet)

    @staticmethod
    def _snapshot_defects(
        batch_defects: List[Set[frozenset]],
        defective_edges: Set[frozenset],
        n: int,
    ) -> None:
        for _ in range(n):
            batch_defects.append(set(defective_edges))

    @staticmethod
    def _plan_non_layer_evacuation(
        G: nx.Graph,
        current_pos: Dict[int, Coord],
        all_qids: Set[int],
        non_layer_qids: Set[int],
        layer_pairs: List[Tuple[int, int]],
        to_meeting_plans: Dict[int, List[TimedNode]],
        fixed_meetings: Dict[frozenset, Coord],
        defective_edges: Set[frozenset],
        occupied_now: Set[Coord],
        F_all: Set[Coord],
        F_layer: Set[Coord],
        tried_meetings: Dict[frozenset, Set[Coord]],
    ) -> Tuple[
        bool,
        Dict[int, List[TimedNode]],
        Dict[frozenset, Coord],
        Dict[int, List[TimedNode]],
        Dict[int, Coord],
        Dict[int, Tuple[int, int]],
        Set[Coord],
    ]:
        blockers_now: List[int] = [qid for qid in non_layer_qids if current_pos[qid] in F_all]
        if not blockers_now:
            return False, to_meeting_plans, fixed_meetings, {}, {}, {}, set()

        node_to_pairs: Dict[Coord, List[Tuple[int, int]]] = {}
        for (a, b) in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue
            nodes = DefaultRoutingPlanner._path_nodes_of_pair(to_meeting_plans, a, b)
            for n in nodes:
                node_to_pairs.setdefault(n, []).append((a, b))

        blocker_to_pair: Dict[int, Tuple[int, int]] = {}
        for qid in blockers_now:
            pos = current_pos[qid]
            pairs_touching = node_to_pairs.get(pos, [])
            if pairs_touching:
                blocker_to_pair[qid] = pairs_touching[0]

        avoid_for_targets = set(occupied_now) | F_all
        evac_targets: Dict[int, Coord] = {}
        for qid in blockers_now:
            tgt = DefaultRoutingPlanner._nearest_free_sn(G, current_pos[qid], avoid_for_targets)
            if tgt is not None and tgt not in F_layer:
                evac_targets[qid] = tgt
                avoid_for_targets.add(tgt)

        replan_current_layer = False
        cannot_place = [qid for qid in blockers_now if qid not in evac_targets]
        if cannot_place:
            seen_pairs: Set[frozenset] = set()
            for qid in cannot_place:
                ab = blocker_to_pair.get(qid)
                if not ab:
                    continue
                pkey = frozenset(ab)
                if pkey in seen_pairs:
                    continue
                seen_pairs.add(pkey)
                meet = fixed_meetings.get(pkey)
                if meet is not None:
                    RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, ab, meet)
                to_meeting_plans.pop(ab[0], None)
                to_meeting_plans.pop(ab[1], None)
                fixed_meetings.pop(pkey, None)
            replan_current_layer = True

        waiter_qids = non_layer_qids - set(evac_targets.keys())
        waiter_nodes = {current_pos[q] for q in waiter_qids}
        blocked_nodes_evacs: Set[Coord] = set(F_all) | set(waiter_nodes)

        evac_plans: Dict[int, List[TimedNode]] = {}
        if evac_targets and not replan_current_layer:
            try:
                evac_plans = DefaultRoutingPlanner._mapf_to_targets(
                    G=G,
                    starts={qid: current_pos[qid] for qid in evac_targets},
                    targets=evac_targets,
                    blocked_nodes=blocked_nodes_evacs,
                    blocked_edges=defective_edges,
                )
            except RuntimeError:
                evac_plans = {}
                for qid, tgt in evac_targets.items():
                    try:
                        one = DefaultRoutingPlanner._mapf_to_targets(
                            G=G,
                            starts={qid: current_pos[qid]},
                            targets={qid: tgt},
                            blocked_nodes=blocked_nodes_evacs,
                            blocked_edges=defective_edges,
                        )
                        evac_plans[qid] = one[qid]
                    except RuntimeError:
                        ab = blocker_to_pair.get(qid)
                        if ab:
                            pkey = frozenset(ab)
                            meet = fixed_meetings.get(pkey)
                            if meet is not None:
                                RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, ab, meet)
                            to_meeting_plans.pop(ab[0], None)
                            to_meeting_plans.pop(ab[1], None)
                            fixed_meetings.pop(pkey, None)
                            replan_current_layer = True

        if evac_plans and not replan_current_layer:
            waiting_qids = non_layer_qids - set(evac_targets.keys())
            evac_plans = RerouteRoutingPlanner._resolve_evacuate_collisions_with_waiters(
                G=G,
                evac_plans=evac_plans,
                targets=evac_targets,
                current_pos=current_pos,
                waiting_qids=waiting_qids,
                blocked_nodes=blocked_nodes_evacs,
                defective_edges=defective_edges,
                blocker_to_pair=blocker_to_pair,
            )

        return (
            replan_current_layer,
            to_meeting_plans,
            fixed_meetings,
            evac_plans,
            evac_targets,
            blocker_to_pair,
            blocked_nodes_evacs,
        )

    @staticmethod
    def _reroute_or_spill_after_defects(
        G: nx.Graph,
        current_pos: Dict[int, Coord],
        layer_pairs: List[Tuple[int, int]],
        layer_starts: Set[Coord],
        defective_edges: Set[frozenset],
        tried_meetings: Dict[frozenset, Set[Coord]],
        to_meeting_plans: Dict[int, List[TimedNode]],
        fixed_meetings: Dict[frozenset, Coord],
        evac_plans: Dict[int, List[TimedNode]],
        evac_targets: Dict[int, Coord],
        blocked_nodes_evacs: Set[Coord],
        blocker_to_pair: Dict[int, Tuple[int, int]],
        layers: List[List[Tuple[int, int]]],
        idx: int,
    ) -> Tuple[
        Dict[int, List[TimedNode]],
        Dict[frozenset, Coord],
        Dict[int, List[TimedNode]],
    ]:
        if evac_plans:
            broken_movers = {
                qid
                for qid, path in evac_plans.items()
                if DefaultRoutingPlanner._path_uses_defective_edge(path, defective_edges)
            }
            if broken_movers:
                try:
                    evac_plans = DefaultRoutingPlanner._mapf_to_targets(
                        G=G,
                        starts={qid: current_pos[qid] for qid in evac_plans.keys()},
                        targets={qid: evac_targets[qid] for qid in evac_plans.keys()},
                        blocked_nodes=blocked_nodes_evacs,
                        blocked_edges=defective_edges,
                    )
                except RuntimeError:
                    to_spill_for_nonlayer: Set[Tuple[int, int]] = set()
                    for qid in broken_movers:
                        ab = blocker_to_pair.get(qid)
                        if ab:
                            to_spill_for_nonlayer.add(ab)

                    if to_spill_for_nonlayer:
                        for (a, b) in to_spill_for_nonlayer:
                            key = frozenset({a, b})
                            meet = fixed_meetings.get(key)
                            if meet is not None:
                                RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                            to_meeting_plans.pop(a, None)
                            to_meeting_plans.pop(b, None)
                            fixed_meetings.pop(key, None)

                        layers[idx + 1 : idx + 1] = [list(to_spill_for_nonlayer)]

                    evac_plans.clear()

        if not fixed_meetings:
            return to_meeting_plans, fixed_meetings, evac_plans

        need_reroute_pairs = RerouteRoutingPlanner._pairs_needing_reroute(
            layer_pairs=layer_pairs,
            fixed_meetings=fixed_meetings,
            to_meeting_plans=to_meeting_plans,
            defective_edges=defective_edges,
        )

        if need_reroute_pairs:
            ok_local, local_plans = RerouteRoutingPlanner._try_local_triangle_bypass_for_pairs(
                G=G,
                current_pos=current_pos,
                layer_pairs=layer_pairs,
                fixed_meetings=fixed_meetings,
                keep_pairs=need_reroute_pairs,
                defective_edges=defective_edges,
                layer_starts=layer_starts,
                existing_layer_plans=to_meeting_plans,
                existing_evac_plans=evac_plans,
            )
            to_meeting_plans.update(local_plans)

            not_ok = [ab for ab in need_reroute_pairs if ab not in ok_local]
            if not_ok:
                for (a, b) in not_ok:
                    key = frozenset({a, b})
                    meet = fixed_meetings.get(key)
                    if meet is not None:
                        RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                    to_meeting_plans.pop(a, None)
                    to_meeting_plans.pop(b, None)
                    fixed_meetings.pop(key, None)
                layers[idx + 1 : idx + 1] = [not_ok]

        return to_meeting_plans, fixed_meetings, evac_plans

    @staticmethod
    def _pairs_needing_reroute(
        layer_pairs: List[Tuple[int, int]],
        fixed_meetings: Dict[frozenset, Coord],
        to_meeting_plans: Dict[int, List[TimedNode]],
        defective_edges: Set[frozenset],
    ) -> Set[Tuple[int, int]]:
        need: Set[Tuple[int, int]] = set()

        for (a, b) in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue
            meet = fixed_meetings[key]

            cut_a = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[a], meet, 0)
            cut_b = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[b], meet, 0)
            if cut_a is None or cut_b is None:
                need.add((a, b))
                continue

            if DefaultRoutingPlanner._path_uses_defective_edge(cut_a, defective_edges):
                need.add((a, b))
                continue
            if DefaultRoutingPlanner._path_uses_defective_edge(cut_b, defective_edges):
                need.add((a, b))
                continue

            pre_a = cut_a[-1][0]
            pre_b = cut_b[-1][0]
            if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                need.add((a, b))

        return need

    @staticmethod
    def _compute_pre_in_paths(
        layer_pairs: List[Tuple[int, int]],
        to_meeting_plans: Dict[int, List[TimedNode]],
        fixed_meetings: Dict[frozenset, Coord],
        tried_meetings: Dict[frozenset, Set[Coord]],
    ) -> Tuple[Dict[int, List[TimedNode]], int]:
        pre_in_paths: Dict[int, List[TimedNode]] = {}
        T_pre_sync = 0

        for (a, b) in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue

            meet = fixed_meetings[key]
            pa = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[a], meet, 0)
            pb = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[b], meet, 0)
            if pa is None or pb is None:
                RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                to_meeting_plans.pop(a, None)
                to_meeting_plans.pop(b, None)
                fixed_meetings.pop(key, None)
                continue

            pre_in_paths[a] = pa
            pre_in_paths[b] = pb
            if pa:
                T_pre_sync = max(T_pre_sync, pa[-1][1])
            if pb:
                T_pre_sync = max(T_pre_sync, pb[-1][1])

        return pre_in_paths, T_pre_sync

    @staticmethod
    def _execute_layer_batches(
        all_qids: Set[int],
        layer_pairs: List[Tuple[int, int]],
        current_pos: Dict[int, Coord],
        defective_edges: Set[frozenset],
        batch_plans: List[Dict[int, List[TimedNode]]],
        batch_defects: List[Set[frozenset]],
        evac_plans: Dict[int, List[TimedNode]],
        to_meeting_plans: Dict[int, List[TimedNode]],
        fixed_meetings: Dict[frozenset, Coord],
        T_pre_sync: int,
        tried_meetings: Dict[frozenset, Set[Coord]],
        layers: List[List[Tuple[int, int]]],
        idx: int,
    ) -> None:
        if evac_plans and any(
            DefaultRoutingPlanner._path_uses_defective_edge(p, defective_edges) for p in evac_plans.values()
        ):
            evac_plans.clear()

        if evac_plans:
            micro_evacuate: Dict[int, List[TimedNode]] = dict(evac_plans.items())
            dur = max((p[-1][1] for p in micro_evacuate.values()), default=0)

            for qid in (all_qids - set(micro_evacuate.keys())):
                micro_evacuate[qid] = [(current_pos[qid], 0), (current_pos[qid], dur)]

            batch_plans.append(micro_evacuate)
            RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

            for qid in evac_plans:
                current_pos[qid] = evac_plans[qid][-1][0]

        micro_to_pre: Dict[int, List[TimedNode]] = {}
        exec_layer_qids: Set[int] = set()

        for (a, b) in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue

            meet = fixed_meetings[key]
            cut_a = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[a], meet, T_pre_sync)
            cut_b = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[b], meet, T_pre_sync)

            if cut_a is None or cut_b is None:
                RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                fixed_meetings.pop(key, None)
                to_meeting_plans.pop(a, None)
                to_meeting_plans.pop(b, None)
                continue

            micro_to_pre[a] = cut_a
            micro_to_pre[b] = cut_b
            exec_layer_qids.update({a, b})

        dur_pre = max((p[-1][1] for p in micro_to_pre.values()), default=0)
        for qid in (all_qids - exec_layer_qids):
            micro_to_pre[qid] = [(current_pos[qid], 0), (current_pos[qid], dur_pre)]

        batch_plans.append(micro_to_pre)
        RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

        for qid in exec_layer_qids:
            current_pos[qid] = micro_to_pre[qid][-1][0]

        micro_in: Dict[int, List[TimedNode]] = {}
        spill_after_micro: List[Tuple[int, int]] = []

        for (a, b) in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue

            meet = fixed_meetings[key]
            pre_a = current_pos[a]
            pre_b = current_pos[b]

            if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                spill_after_micro.append((a, b))
                micro_in[a] = [(pre_a, 0), (pre_a, 2)]
                micro_in[b] = [(pre_b, 0), (pre_b, 2)]
            else:
                micro_in[a] = [(pre_a, 0), (meet, 1), (pre_a, 2)]
                micro_in[b] = [(pre_b, 0), (meet, 1), (pre_b, 2)]

        active_layer_qids = {q for ab in layer_pairs if frozenset(ab) in fixed_meetings for q in ab}
        for qid in (all_qids - active_layer_qids):
            micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]

        batch_plans.append(micro_in)
        RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

        if spill_after_micro:
            for (a, b) in spill_after_micro:
                key = frozenset({a, b})
                meet = fixed_meetings.get(key)
                if meet is not None:
                    RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                fixed_meetings.pop(key, None)

            layers[idx + 1 : idx + 1] = [spill_after_micro]

        for a, b in [tuple(sorted(k)) for k in fixed_meetings.keys()]:
            tried_meetings.pop(frozenset({a, b}), None)

    @staticmethod
    def _reserve_existing_plans(
        res: Reservations,
        plans: Dict[int, List[TimedNode]],
        skip_qids: Optional[Set[int]] = None,
    ) -> None:
        skip_qids = skip_qids or set()
        for qid, path in plans.items():
            if qid in skip_qids:
                continue
            Reservations.commit(res, path)

    @staticmethod
    def _common_triangle_candidates(
        G: nx.Graph,
        u: Coord,
        v: Coord,
        banned_nodes: Optional[Set[Coord]] = None,
    ) -> List[Coord]:
        banned_nodes = banned_nodes or set()
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        cand = list(Nu & Nv)
        cand = [w for w in cand if w != u and w != v and w not in banned_nodes]
        cand.sort(key=lambda c: (c[0], c[1]))
        return cand

    @staticmethod
    def _patch_path_with_triangle_bypass(
        G: nx.Graph,
        path: List[TimedNode],
        defective_edges: Set[frozenset],
        blocked_nodes_static: Optional[Set[Coord]] = None,
    ) -> List[TimedNode]:
        if not path or len(path) < 2:
            return path

        blocked_nodes_static = blocked_nodes_static or set()

        new_path: List[TimedNode] = [path[0]]
        total_extra = 0

        for i in range(1, len(path)):
            u, tu = new_path[-1]
            v, tv_orig = path[i]
            tv = tv_orig + total_extra

            if frozenset({u, v}) not in defective_edges:
                new_path.append((v, tv))
                continue

            candidates = RerouteRoutingPlanner._common_triangle_candidates(
                G, u, v, banned_nodes=blocked_nodes_static
            )

            picked_w: Optional[Coord] = None
            for w in candidates:
                if frozenset({u, w}) in defective_edges:
                    continue
                if frozenset({w, v}) in defective_edges:
                    continue
                picked_w = w
                break

            if picked_w is None:
                new_path.append((v, tv))
                continue

            new_path.append((picked_w, tu + 1))
            new_path.append((v, tu + 2))
            total_extra += 1

        return new_path

    @staticmethod
    def _try_local_triangle_bypass_for_pairs(
        G: nx.Graph,
        current_pos: Dict[int, Coord],
        layer_pairs: List[Tuple[int, int]],
        fixed_meetings: Dict[frozenset, Coord],
        keep_pairs: Set[Tuple[int, int]],
        defective_edges: Set[frozenset],
        layer_starts: Set[Coord],
        existing_layer_plans: Optional[Dict[int, List[TimedNode]]],
        existing_evac_plans: Optional[Dict[int, List[TimedNode]]],
    ) -> Tuple[Set[Tuple[int, int]], Dict[int, List[TimedNode]]]:
        ok_pairs: Set[Tuple[int, int]] = set()
        new_plans: Dict[int, List[TimedNode]] = {}

        res_base = Reservations(G, blocked_edges=defective_edges)

        skip_qids: Set[int] = {q for ab in keep_pairs for q in ab}
        if existing_layer_plans:
            RerouteRoutingPlanner._reserve_existing_plans(res_base, existing_layer_plans, skip_qids=skip_qids)
        if existing_evac_plans:
            RerouteRoutingPlanner._reserve_existing_plans(res_base, existing_evac_plans, skip_qids=None)

        moving_qids: Set[int] = set()
        if existing_layer_plans:
            moving_qids |= set(existing_layer_plans.keys())
        if existing_evac_plans:
            moving_qids |= set(existing_evac_plans.keys())

        stationary_nodes: Set[Coord] = {
            current_pos[qid]
            for qid in current_pos.keys()
            if (qid not in moving_qids) and (qid not in skip_qids)
        }
        for node in stationary_nodes:
            cap = res_base.node_capacity(node)
            for t in range(0, MAX_TIME + 1):
                res_base.node_caps[node][t] = cap
        for node in stationary_nodes:
            Reservations.commit(res_base, [(node, 0), (node, MAX_TIME)])

        placed_preins: Set[Coord] = set()
        if existing_layer_plans:
            for pair_key, meet in fixed_meetings.items():
                a, b = list(pair_key)
                if (a, b) in keep_pairs or (b, a) in keep_pairs:
                    continue
                for qid in (a, b):
                    pth = existing_layer_plans.get(qid)
                    if not pth:
                        continue
                    pin = DefaultRoutingPlanner._entry_sn_from_path(pth, meet)
                    if pin is not None:
                        placed_preins.add(pin)
            for pin in placed_preins:
                cap = res_base.node_capacity(pin)
                for t in range(0, MAX_TIME + 1):
                    res_base.node_caps[pin][t] = cap

        evac_target_nodes: Set[Coord] = set()
        if existing_evac_plans:
            for p in existing_evac_plans.values():
                if p:
                    evac_target_nodes.add(p[-1][0])
        for node in evac_target_nodes:
            cap = res_base.node_capacity(node)
            for t in range(0, MAX_TIME + 1):
                res_base.node_caps[node][t] = cap

        blocked_nodes_static: Set[Coord] = set(stationary_nodes)

        def md(a: Coord, b: Coord) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        order = sorted(
            keep_pairs,
            key=lambda ab: md(current_pos[ab[0]], fixed_meetings[frozenset(ab)])
            + md(current_pos[ab[1]], fixed_meetings[frozenset(ab)]),
        )

        for (a, b) in order:
            meet = fixed_meetings[frozenset({a, b})]
            pa = existing_layer_plans.get(a) if existing_layer_plans else None
            pb = existing_layer_plans.get(b) if existing_layer_plans else None
            if pa is None or pb is None:
                continue

            uses_defect_a = DefaultRoutingPlanner._path_uses_defective_edge(pa, defective_edges)
            uses_defect_b = DefaultRoutingPlanner._path_uses_defective_edge(pb, defective_edges)
            if not (uses_defect_a or uses_defect_b):
                continue

            pa_patched = (
                RerouteRoutingPlanner._patch_path_with_triangle_bypass(
                    G, pa, defective_edges, blocked_nodes_static=blocked_nodes_static
                )
                if uses_defect_a
                else pa
            )
            pb_patched = (
                RerouteRoutingPlanner._patch_path_with_triangle_bypass(
                    G, pb, defective_edges, blocked_nodes_static=blocked_nodes_static
                )
                if uses_defect_b
                else pb
            )

            pre_a = DefaultRoutingPlanner._entry_sn_from_path(pa_patched, meet)
            pre_b = DefaultRoutingPlanner._entry_sn_from_path(pb_patched, meet)
            if pre_a is None or pre_b is None:
                continue
            if pre_a == pre_b:
                continue
            if pre_a in placed_preins or pre_b in placed_preins:
                continue

            res_try = deepcopy(res_base)
            try:
                Reservations.commit(res_try, pa_patched)
                Reservations.commit(res_try, pb_patched)
            except Exception:
                continue

            if DefaultRoutingPlanner._path_uses_defective_edge(pa_patched, defective_edges):
                continue
            if DefaultRoutingPlanner._path_uses_defective_edge(pb_patched, defective_edges):
                continue

            Reservations.commit(res_base, pa_patched)
            Reservations.commit(res_base, pb_patched)

            placed_preins.update({pre_a, pre_b})
            new_plans[a] = pa_patched
            new_plans[b] = pb_patched
            ok_pairs.add((a, b))

        return ok_pairs, new_plans
