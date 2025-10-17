from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
from copy import deepcopy
import networkx as nx

from routing.common import AStar, Coord, MAX_TIME, Reservations, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner

P_SUCCESS = 1
P_REPAIR = 0.25
MAX_REPLANS = 50


class RerouteRoutingPlanner(DefaultRoutingPlanner):
    """
    Variante: Nach Schritt 3 wird bei *jedem* festgestellten Defekt zuerst
    im selben Layer eine kollisionsfreie Neuplanung versucht, die dieselben
    Ziele (Evakuierungs-Ziel-SN bzw. Meeting-IN) beibehält und defekte Kanten vermeidet.
    Nur wenn das nicht möglich ist, werden die betroffenen Paare in ein Spillover-Layer verschoben.
    """

    @staticmethod
    def route(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float = P_SUCCESS,
        p_repair: float = P_REPAIR,
    ):
        # --- Zustand ---
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}
        all_qids: Set[int] = {q.id for q in qubits}

        defective_edges: Set[frozenset] = set()
        batch_plans: List[Dict[int, List[TimedNode]]] = []
        batch_defects: List[Set[frozenset]] = []

        total_ins: Set[Coord] = {n for n in G if G.nodes[n].get("type") == "IN"}
        tried_meetings: Dict[frozenset, Set[Coord]] = {}

        def mark_meeting_failed(pair: Tuple[int, int], meet: Coord):
            key = frozenset(pair)
            s = tried_meetings.setdefault(key, set())
            s.add(meet)

        def snapshot_defects(n: int):
            for _ in range(n):
                batch_defects.append(set(defective_edges))

        # --- Layerbildung ---
        layers: List[List[Tuple[int, int]]] = []
        used: Set[int] = set()
        cur: List[Tuple[int, int]] = []
        for qa, qb in pairs:
            a, b = qa.id, qb.id
            if a not in used and b not in used:
                cur.append((a, b)); used |= {a, b}
            else:
                if cur: layers.append(cur)
                cur = [(a, b)]; used = {a, b}
        if cur: layers.append(cur)

        idx = 0
        replan_counts: Dict[int, int] = {}
        while idx < len(layers):
            tried_meetings.clear()
            layer_pairs = layers[idx]
            layer_qids: Set[int] = {x for ab in layer_pairs for x in ab}
            non_layer_qids: Set[int] = all_qids - layer_qids
            layer_starts: Set[Coord] = {current_pos[q] for q in layer_qids}
            occupied_now: Set[Coord] = {current_pos[q] for q in all_qids}

            replan_current_layer = False

            # ===== Schritt 1: Layer-only planen (beste INs) =====
            (
                to_meeting_plans,
                fixed_meetings,
                _preins_ok,
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
                layers[idx+1:idx+1] = [exhausted_pairs_step1]

            if unplaceable_pairs_step1:
                layers[idx+1:idx+1] = [unplaceable_pairs_step1]
                if not fixed_meetings:
                    wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                    batch_plans.append(wait); snapshot_defects(1); idx += 1; continue

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait); snapshot_defects(1); idx += 1; continue

            # Vom Layer genutzte Knoten
            F_layer: Set[Coord] = DefaultRoutingPlanner._collect_layer_nodes(to_meeting_plans, fixed_meetings)
            F_all = set(F_layer) | set(layer_starts)

            # ===== Schritt 2: Non-Layer evakuieren, falls sie Layer belegen =====
            blockers_now: List[int] = [qid for qid in non_layer_qids if current_pos[qid] in F_all]
            blocker_to_pair: Dict[int, Tuple[int, int]] = {}
            evac_plans: Dict[int, List[TimedNode]] = {}
            evac_targets: Dict[int, Coord] = {}

            if blockers_now:
                # Zuordnung Blocker -> Paar
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

                # Ziele für Blocker: freie SN außerhalb des Layer-Footprints
                avoid_for_targets = set(occupied_now) | F_all
                for qid in blockers_now:
                    tgt = DefaultRoutingPlanner._nearest_free_sn(G, current_pos[qid], avoid_for_targets)
                    if tgt is not None and tgt not in F_layer:
                        evac_targets[qid] = tgt
                        avoid_for_targets.add(tgt)

                cannot_place = [qid for qid in blockers_now if qid not in evac_targets]
                if cannot_place:
                    seen_pairs: Set[frozenset] = set()
                    for qid in cannot_place:
                        ab = blocker_to_pair.get(qid)
                        if not ab: continue
                        pkey = frozenset(ab)
                        if pkey in seen_pairs: continue
                        seen_pairs.add(pkey)
                        meet = fixed_meetings.get(pkey)
                        if meet is not None:
                            mark_meeting_failed(ab, meet)
                        to_meeting_plans.pop(ab[0], None); to_meeting_plans.pop(ab[1], None)
                        fixed_meetings.pop(pkey, None)
                    replan_current_layer = True

                # WICHTIG: Warte-Knoten als tabu markieren, um Vertex-Kollisionen zu verhindern
                waiter_qids = (non_layer_qids - set(evac_targets.keys()))
                waiter_nodes = { current_pos[q] for q in waiter_qids }
                blocked_nodes_evacs = F_all | waiter_nodes

                if evac_targets and not replan_current_layer:
                    try:
                        # gemeinsamer MAPF ohne Waiter-Kollisionen
                        evac_plans = DefaultRoutingPlanner._mapf_to_targets(
                            G=G,
                            starts={qid: current_pos[qid] for qid in evac_targets},
                            targets=evac_targets,
                            blocked_nodes=blocked_nodes_evacs,
                            blocked_edges=defective_edges,
                        )
                    except RuntimeError:
                        # Single-agent fallback (weiterhin Waiter-Knoten tabu)
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
                                # Ziel ist bereits tabu, da blocked_nodes_evacs alle Waiter + Layer-Knoten enthält
                            except RuntimeError:
                                ab = blocker_to_pair.get(qid)
                                if ab:
                                    pkey = frozenset(ab)
                                    meet = fixed_meetings.get(pkey)
                                    if meet is not None:
                                        mark_meeting_failed(ab, meet)
                                    to_meeting_plans.pop(ab[0], None); to_meeting_plans.pop(ab[1], None)
                                    fixed_meetings.pop(pkey, None)
                                    replan_current_layer = True

                if evac_plans and not replan_current_layer:
                    waiting_qids = (non_layer_qids - set(evac_targets.keys()))
                    # Handshake-/Chain-Auflösung mit Paar-Propagation
                    evac_plans = RerouteRoutingPlanner._resolve_evacuate_collisions_with_waiters(
                        G=G,
                        evac_plans=evac_plans,
                        targets=evac_targets,
                        current_pos=current_pos,
                        waiting_qids=waiting_qids,
                        blocked_nodes=blocked_nodes_evacs,     # Waiter dauerhaft tabu
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
                batch_plans.append(wait); snapshot_defects(1); idx += 1; continue

            # ===== Schritt 3: Defekte samplen =====
            DefaultRoutingPlanner._sample_edge_failures(
                G, defective_edges, p_fail=(1.0 - p_success), p_repair=p_repair
            )

            # ===== 3a) Non-Layer: Defekt → Re-MAPF (gleiches Ziel); sonst Spillover & niemand bewegt sich =====
            if evac_plans:
                broken_movers = {
                    qid for qid, path in evac_plans.items()
                    if DefaultRoutingPlanner._path_uses_defective_edge(path, defective_edges)
                }
                if broken_movers:
                    try:
                        # neu planen mit gleichen Zielen; Waiter bleiben tabu
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
                                    mark_meeting_failed((a, b), meet)
                                to_meeting_plans.pop(a, None); to_meeting_plans.pop(b, None)
                                fixed_meetings.pop(key, None)
                            layers[idx+1:idx+1] = [list(to_spill_for_nonlayer)]
                        # **keine** Non-Layer-Bewegung in diesem Batch
                        evac_plans.clear()

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait); snapshot_defects(1); idx += 1; continue

            # --- 3b) Layer-PreIN/Hop Defekte → Re-Routing mit gleichen Meetings ---
            pre_in_paths: Dict[int, List[TimedNode]] = {}
            T_pre_sync = 0
            need_reroute_pairs: Set[Tuple[int, int]] = set()

            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]
                cut_a = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[a], meet, 0)
                cut_b = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[b], meet, 0)
                if cut_a is None or cut_b is None:
                    need_reroute_pairs.add((a, b)); continue
                pre_in_paths[a] = cut_a; pre_in_paths[b] = cut_b
                if cut_a: T_pre_sync = max(T_pre_sync, cut_a[-1][1])
                if cut_b: T_pre_sync = max(T_pre_sync, cut_b[-1][1])

            def pair_uses_defect(a: int, b: int, meet: Coord) -> bool:
                pa = pre_in_paths.get(a); pb = pre_in_paths.get(b)
                if pa is None or pb is None:
                    return True
                if DefaultRoutingPlanner._path_uses_defective_edge(pa, defective_edges): return True
                if DefaultRoutingPlanner._path_uses_defective_edge(pb, defective_edges): return True
                pre_a = pa[-1][0]; pre_b = pb[-1][0]
                if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                    return True
                return False

            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]
                if pair_uses_defect(a, b, meet):
                    need_reroute_pairs.add((a, b))

            if need_reroute_pairs:
                ok_pairs, new_plans = RerouteRoutingPlanner._replan_layer_paths_same_meetings(
                    G=G,
                    current_pos=current_pos,
                    layer_pairs=layer_pairs,
                    fixed_meetings=fixed_meetings,
                    keep_pairs=need_reroute_pairs,
                    defective_edges=defective_edges,
                    layer_starts=layer_starts,
                    existing_layer_plans=to_meeting_plans,
                    existing_evac_plans=evac_plans
                )
                to_meeting_plans.update(new_plans)
                not_ok = [ab for ab in need_reroute_pairs if ab not in ok_pairs]
                if not_ok:
                    for (a, b) in not_ok:
                        key = frozenset({a, b})
                        meet = fixed_meetings.get(key)
                        if meet is not None:
                            mark_meeting_failed((a, b), meet)
                        to_meeting_plans.pop(a, None); to_meeting_plans.pop(b, None)
                        fixed_meetings.pop(key, None)
                    layers[idx+1:idx+1] = [not_ok]

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait); snapshot_defects(1); idx += 1; continue

            # PreIN Pfade für Ausführung neu schneiden
            pre_in_paths = {}
            T_pre_sync = 0
            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]
                pa = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[a], meet, 0)
                pb = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[b], meet, 0)
                if pa is None or pb is None:
                    mark_meeting_failed((a, b), meet)
                    to_meeting_plans.pop(a, None); to_meeting_plans.pop(b, None)
                    fixed_meetings.pop(key, None)
                    continue
                pre_in_paths[a] = pa; pre_in_paths[b] = pb
                if pa: T_pre_sync = max(T_pre_sync, pa[-1][1])
                if pb: T_pre_sync = max(T_pre_sync, pb[-1][1])

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait); snapshot_defects(1); idx += 1; continue

            # ===== Schritt 4: Ausführung =====
            # 4a) Non-Layer Evakuierung zuerst (Safety-Net: keine defekten Pfade)
            if evac_plans:
                if any(DefaultRoutingPlanner._path_uses_defective_edge(p, defective_edges)
                    for p in evac_plans.values()):
                    evac_plans.clear()

            if evac_plans:
                micro_evacuate: Dict[int, List[TimedNode]] = {qid: path for qid, path in evac_plans.items()}
                dur = max((p[-1][1] for p in micro_evacuate.values()), default=0)
                for qid in (all_qids - set(micro_evacuate.keys())):
                    micro_evacuate[qid] = [(current_pos[qid], 0), (current_pos[qid], dur)]
                batch_plans.append(micro_evacuate); snapshot_defects(1)
                for qid in evac_plans:
                    current_pos[qid] = evac_plans[qid][-1][0]

            # 4b) bis PRE-IN, synchronisiert
            micro_to_pre: Dict[int, List[TimedNode]] = {}
            exec_layer_qids: Set[int] = set()
            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]
                cut_a = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[a], meet, T_pre_sync)
                cut_b = DefaultRoutingPlanner._retime_until_pre_in_wait(to_meeting_plans[b], meet, T_pre_sync)
                micro_to_pre[a] = cut_a; micro_to_pre[b] = cut_b
                exec_layer_qids.update({a, b})

            others = all_qids - exec_layer_qids
            dur_pre = max((p[-1][1] for p in micro_to_pre.values()), default=0)
            for qid in others:
                micro_to_pre[qid] = [(current_pos[qid], 0), (current_pos[qid], dur_pre)]
            batch_plans.append(micro_to_pre); snapshot_defects(1)
            for qid in exec_layer_qids:
                current_pos[qid] = micro_to_pre[qid][-1][0]

            # 4c) kurzer IN-Hop (mit Hop-Defekt-Gurt)
            micro_in: Dict[int, List[TimedNode]] = {}
            spill_after_micro: List[Tuple[int, int]] = []
            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]
                pre_a = current_pos[a]; pre_b = current_pos[b]
                if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                    spill_after_micro.append((a, b))
                    micro_in[a] = [(pre_a, 0), (pre_a, 2)]
                    micro_in[b] = [(pre_b, 0), (pre_b, 2)]
                else:
                    micro_in[a] = [(pre_a, 0), (meet, 1), (pre_a, 2)]
                    micro_in[b] = [(pre_b, 0), (meet, 1), (pre_b, 2)]

            rest = all_qids - {q for ab in layer_pairs if frozenset(ab) in fixed_meetings for q in ab}
            for qid in rest:
                micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]
            batch_plans.append(micro_in); snapshot_defects(1)

            if spill_after_micro:
                for (a, b) in spill_after_micro:
                    key = frozenset({a, b})
                    meet = fixed_meetings.get(key)
                    if meet is not None:
                        mark_meeting_failed((a, b), meet)
                    fixed_meetings.pop(key, None)
                layers[idx+1:idx+1] = [spill_after_micro]

            # Paare, die fertig wurden → tried_meetings resetten
            succeeded_pairs = [tuple(sorted(k)) for k in fixed_meetings.keys()]
            for a, b in succeeded_pairs:
                tried_meetings.pop(frozenset({a, b}), None)

            idx += 1

        return DefaultRoutingPlanner.stitch_batches(qubits, batch_plans, batch_defects)


    # ---------- Helfer: Neuplanung Layer mit *fixen* Meeting-INs ----------
    @staticmethod
    def _replan_layer_paths_same_meetings(
        G: nx.Graph,
        current_pos: Dict[int, Coord],
        layer_pairs: List[Tuple[int, int]],
        fixed_meetings: Dict[frozenset, Coord],
        keep_pairs: Set[Tuple[int, int]],
        defective_edges: Set[frozenset],
        layer_starts: Set[Coord],
        existing_layer_plans: Optional[Dict[int, List[TimedNode]]] = None,
        existing_evac_plans: Optional[Dict[int, List[TimedNode]]] = None,
    ) -> Tuple[Set[Tuple[int, int]], Dict[int, List[TimedNode]]]:

        ok_pairs: Set[Tuple[int, int]] = set()
        new_plans: Dict[int, List[TimedNode]] = {}

        # Basis-Reservations mit defekten Kanten
        res_base = Reservations(G, blocked_edges=defective_edges)

        # --- 1) Bereits vorhandene zeitliche Reservierungen eintragen ---
        # Qubits, die wir jetzt neu planen (beide Endpunkte aller keep_pairs), werden NICHT reserviert.
        skip_qids: Set[int] = {q for ab in keep_pairs for q in ab}

        if existing_layer_plans:
            RerouteRoutingPlanner._reserve_existing_plans(res_base, existing_layer_plans, skip_qids=skip_qids)
        if existing_evac_plans:
            RerouteRoutingPlanner._reserve_existing_plans(res_base, existing_evac_plans, skip_qids=None)

        # --- 2) Stationäre Qubits (in diesem Layer unbewegt) als *permanent* blockierte Knoten sperren ---
        moving_qids: Set[int] = set()
        if existing_layer_plans:
            moving_qids |= set(existing_layer_plans.keys())
        if existing_evac_plans:
            moving_qids |= set(existing_evac_plans.keys())

        # alles, was keine aktuelle Route hat UND nicht in keep_pairs neu geroutet wird, ist stationär
        stationary_nodes: Set[Coord] = {
            current_pos[qid]
            for qid in current_pos.keys()
            if (qid not in moving_qids) and (qid not in skip_qids)
        }
        # blockiere diese Knoten über den vollen Horizont
        for node in stationary_nodes:
            cap = res_base.node_capacity(node)
            for t in range(0, MAX_TIME + 1):
                res_base.node_caps[node][t] = cap

        # --- 3) PRE-INs aus unbetroffenen Layer-Pfaden sammeln, damit diese nicht doppelt belegt werden ---
        placed_preins: Set[Coord] = set()
        if existing_layer_plans:
            for pair_key, meet in fixed_meetings.items():
                a, b = list(pair_key)
                if (a, b) in keep_pairs or (b, a) in keep_pairs:
                    continue
                for qid in (a, b):
                    path = existing_layer_plans.get(qid)
                    if not path:
                        continue
                    pin = DefaultRoutingPlanner._entry_sn_from_path(path, meet)
                    if pin is not None:
                        placed_preins.add(pin)

        # Hilfsfunktionen
        def prein_of(path: List[TimedNode], meet: Coord) -> Optional[Coord]:
            return DefaultRoutingPlanner._entry_sn_from_path(path, meet)

        def md(a: Coord, b: Coord) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        order = sorted(
            keep_pairs,
            key=lambda ab: md(current_pos[ab[0]], fixed_meetings[frozenset(ab)]) +
                           md(current_pos[ab[1]], fixed_meetings[frozenset(ab)])
        )

        # --- 4) Paarweise neu routen (mit paarspezifischer Startsperre für fremde Layer-Starts) ---
        evac_target_nodes: Set[Coord] = set()
        if existing_evac_plans:
            for path in existing_evac_plans.values():
                if path:
                    evac_target_nodes.add(path[-1][0])

        for (a, b) in order:
            meet = fixed_meetings[frozenset({a, b})]

            # Lokaler Reservations-Snapshot
            res = deepcopy(res_base)

            blocked_evacs = evac_target_nodes - { current_pos[a], current_pos[b] }
            for node in blocked_evacs:
                cap = res.node_capacity(node)
                for t in range(0, MAX_TIME + 1):
                    res.node_caps[node][t] = cap

            # PRE-INs, die bereits belegt sind, für den gesamten Horizont sperren
            for pin in placed_preins:
                cap = res.node_capacity(pin)
                for t in range(0, MAX_TIME + 1):
                    res.node_caps[pin][t] = cap

            # Fremde Layer-Starts außer eigenem Start für A und später B sperren
            for node in (layer_starts - {current_pos[a]}):
                cap = res.node_capacity(node)
                for t in range(0, MAX_TIME + 1):
                    res.node_caps[node][t] = cap
            pa = AStar.search(G, current_pos[a], meet, res)
            if pa is None:
                continue
            Reservations.commit(res, pa)

            for node in (layer_starts - {current_pos[b]}):
                cap = res.node_capacity(node)
                for t in range(0, MAX_TIME + 1):
                    res.node_caps[node][t] = cap
            pb = AStar.search(G, current_pos[b], meet, res)
            if pb is None:
                continue

            pre_a = prein_of(pa, meet); pre_b = prein_of(pb, meet)
            if pre_a is None or pre_b is None:
                continue
            if pre_a in placed_preins or pre_b in placed_preins or pre_a == pre_b:
                continue

            # akzeptieren und kumulativ reservieren
            new_plans[a] = pa; new_plans[b] = pb
            placed_preins.update({pre_a, pre_b})
            Reservations.commit(res_base, pa)
            Reservations.commit(res_base, pb)
            ok_pairs.add((a, b))

        return ok_pairs, new_plans
    
    @staticmethod
    def _reserve_existing_plans(
        res: Reservations,
        plans: Dict[int, List[TimedNode]],
        skip_qids: Optional[Set[int]] = None,
    ) -> None:
        """
        Commit-et alle Pfade in 'plans' in den Reservations-State 'res'.
        'skip_qids' werden ausgelassen (z. B. weil sie gerade neu geroutet werden).
        """
        skip_qids = skip_qids or set()
        for qid, path in plans.items():
            if qid in skip_qids:
                continue
            Reservations.commit(res, path)
