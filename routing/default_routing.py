from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
from copy import deepcopy
import random
import networkx as nx

from routing.common import AStar, Coord, MAX_TIME, Reservations, TimedNode, Qubit
from routing.routing_strategy import RoutingStrategy

MAX_REPLANS = 8
MAX_GLOBAL_ITERS = 8


class DefaultRoutingPlanner(RoutingStrategy):
    """
    Routing in Layern mit (1) Auswahl bester Meeting-INs rein nach Layer,
    (2) gezielter Evakuierung blockierender Non-Layer, (3) Sampling persistenter
    Kantenfehler, (4) deterministischer Ausführung (Non-Layer -> preIN -> kurzer IN-Hop),
    (5) Spillover nur für betroffene Paare.

    Wichtige Nebenbedingungen:
      - Pro Layer: keine zwei Paare teilen sich denselben IN.
      - Keine Pfadknoten dürfen Startknoten eines Layer-Qubits sein.
      - Alle Layer-Qubits müssen unterschiedliche PRE-IN (SN) benutzen.
      - Non-Layer werden (falls nötig) kollisionsfrei auf freie SNs evakuiert,
        die von keinem Layer-Pfad belegt sind. Gelingt das für einen Blocker
        nicht, wird nur das betroffene Paar in ein Spillover-Layer verschoben.
    """

    # ---------- Öffentliche Hauptfunktion ----------
    def route(
        self,
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float,
        p_repair: float,
    ):
        # Zustand
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}
        all_qids: Set[int] = {q.id for q in qubits}

        defective_edges: Set[frozenset] = set()  # persistent über alle Layer
        batch_plans: List[Dict[int, List[TimedNode]]] = []
        batch_defects: List[Set[frozenset]] = []

        # --- Persistente Buchhaltung: probierte Meeting-INs pro Paar ---
        total_ins: Set[Coord] = {n for n in G if G.nodes[n].get("type") == "IN"}
        tried_meetings: Dict[frozenset, Set[Coord]] = {}

        def snapshot_defects(n: int):
            for _ in range(n):
                batch_defects.append(set(defective_edges))

        # ---------- Layerbildung identisch zu Default ----------
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

        # ---------- Hauptschleife ----------
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

            replan_current_layer = False  # ★ falls wir im selben Layer neu planen wollen

            # ===== Schritt 1: Nur Layer planen (beste INs, keine Non-Layer beachten) =====
            (
                to_meeting_plans,          # qid -> Pfad bis MEETING (mit Kollisionvermeidung)
                fixed_meetings,            # pair_key -> meeting IN
                preins_ok,                 # True/False, ob PRE-INs schon eindeutig
                unplaceable_pairs_step1,   # Paare, die trotz Durchprobieren aller Kandidaten nicht platzbar sind
                exhausted_pairs_step1,     # ★ Paare, für die *alle* INs schon global verboten wurden
            ) = DefaultRoutingPlanner._plan_layer_only(
                G=G,
                current_pos=current_pos,
                layer_pairs=layer_pairs,
                layer_starts=layer_starts,
                defective_edges=defective_edges,
                banned_meetings=tried_meetings,   # ★ verbotene/gescheiterte INs berücksichtigen
                all_ins=total_ins,
            )

            # ★ statt RuntimeError: Paare mit erschöpften IN-Möglichkeiten in Spillover
            if exhausted_pairs_step1:
                layers[idx+1:idx+1] = [exhausted_pairs_step1]

            DefaultRoutingPlanner._debug_print_meetings(idx, fixed_meetings, header="-- Nach _plan_layer_only --")

            # Paare aus Schritt 1 nicht planbar → Spillover (wir haben alle Kandidaten versucht)
            if unplaceable_pairs_step1:
                layers[idx+1:idx+1] = [unplaceable_pairs_step1]
                if not fixed_meetings:
                    wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                    batch_plans.append(wait)
                    snapshot_defects(1)
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
                snapshot_defects(1)
                idx += 1
                continue

            # Menge der vom Layer genutzten Knoten (für Evakuierungs-Sperren)
            F_layer: Set[Coord] = DefaultRoutingPlanner._collect_layer_nodes(to_meeting_plans, fixed_meetings)
            F_all = set(F_layer) | set(layer_starts)

            # ===== Schritt 2: Blockierende Non-Layer evakuieren (nur falls nötig) =====
            blockers_now: List[int] = [
                qid for qid in non_layer_qids if current_pos[qid] in F_all
            ]

            blocker_to_pair: Dict[int, Tuple[int, int]] = {}
            evac_plans: Dict[int, List[TimedNode]] = {}

            if blockers_now:
                # Zuordnung Blocker -> Paar
                node_to_pairs: Dict[Coord, List[Tuple[int, int]]] = {}
                pair_path_nodes: Dict[Tuple[int, int], Set[Coord]] = {}
                for (a, b) in layer_pairs:
                    key = frozenset({a, b})
                    if key not in fixed_meetings:
                        continue
                    nodes = DefaultRoutingPlanner._path_nodes_of_pair(to_meeting_plans, a, b)
                    pair_path_nodes[(a, b)] = nodes
                    for n in nodes:
                        node_to_pairs.setdefault(n, []).append((a, b))

                for qid in blockers_now:
                    pos = current_pos[qid]
                    pairs_touching = node_to_pairs.get(pos, [])
                    if pairs_touching:
                        blocker_to_pair[qid] = pairs_touching[0]

                # Ziele für Blocker: freie SN, die NICHT in F_layer liegen
                avoid_for_targets = set(occupied_now) | F_all
                targets: Dict[int, Coord] = {}
                for qid in blockers_now:
                    tgt = DefaultRoutingPlanner._nearest_free_sn(G, current_pos[qid], avoid_for_targets)
                    if tgt is not None and tgt not in F_layer:
                        targets[qid] = tgt
                        avoid_for_targets.add(tgt)

                # Wer keinen Ziel-SN findet → bisher Spillover, jetzt stattdessen IN verbieten & replan
                cannot_place = [qid for qid in blockers_now if qid not in targets]
                newly_affected: List[Tuple[int, int]] = []
                for qid in cannot_place:
                    if qid in blocker_to_pair:
                        ab = blocker_to_pair[qid]
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
                        meet = fixed_meetings.get(pkey)
                        to_meeting_plans.pop(ab[0], None)
                        to_meeting_plans.pop(ab[1], None)
                        fixed_meetings.pop(pkey, None)
                    replan_current_layer = True  # ★ anderes IN im gleichen Layer probieren

                # Evakuierung für die restlichen Blocker planen
                evacuating = {qid: current_pos[qid] for qid in blockers_now if qid in targets}
                if evacuating:
                    try:
                        evac_plans = DefaultRoutingPlanner._mapf_to_targets(
                            G=G,
                            starts=evacuating,
                            targets={qid: targets[qid] for qid in evacuating},
                            blocked_nodes=F_all,                    # niemals Layer-Knoten betreten
                            blocked_edges=defective_edges,          # aktuelle Defekte vermeiden
                        )
                    except RuntimeError:
                        # Fallback agent-weise
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
                                    meet = fixed_meetings.get(pkey)
                                    to_meeting_plans.pop(ab[0], None)
                                    to_meeting_plans.pop(ab[1], None)
                                    fixed_meetings.pop(pkey, None)
                                    replan_current_layer = True  # ★ anderes IN probieren

                # === Kollisionen mit wartenden Non-Layer auflösen (Handover-Regel) ===
                if evac_plans:
                    waiting_qids = (non_layer_qids - set(evacuating.keys()))
                    evac_plans = DefaultRoutingPlanner._resolve_evacuate_collisions_with_waiters(
                        G=G,
                        evac_plans=evac_plans,
                        targets={qid: targets[qid] for qid in evacuating},  # Ziele der bewegenden Non-Layer
                        current_pos=current_pos,
                        waiting_qids=waiting_qids,
                        blocked_nodes=F_all,               # Layer-Knoten tabu
                        defective_edges=defective_edges,   # respektiere Defekte
                        blocker_to_pair=blocker_to_pair
                    )

            # ★ Falls wir neu planen wollen: springe zum Schleifenanfang (gleiches Layer, neue IN-Wahl)
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
                snapshot_defects(1)
                idx += 1
                continue

            # ===== Schritt 3: Sampling defekter Kanten und Validierung =====
            DefaultRoutingPlanner._sample_edge_failures(
                G, defective_edges, p_fail=(1.0 - p_success), p_repair=p_repair
            )

            # Prüfe Non-Layer-Evakuierungspfade: Defekt → betroff. Paar in Spillover (Ausnahme-Fall)
            if evac_plans:
                to_spill_for_nonlayer: Set[Tuple[int, int]] = set()
                for qid, path in evac_plans.items():
                    if DefaultRoutingPlanner._path_uses_defective_edge(path, defective_edges):
                        ab = blocker_to_pair.get(qid)
                        if ab:
                            to_spill_for_nonlayer.add(ab)

                if to_spill_for_nonlayer:
                    # ★ Echte Defekte -> direkt Spillover der betroffenen Paare
                    for (a, b) in to_spill_for_nonlayer:
                        key = frozenset({a, b})
                        meet = fixed_meetings.get(key)
                        to_meeting_plans.pop(a, None)
                        to_meeting_plans.pop(b, None)
                        fixed_meetings.pop(key, None)
                    layers[idx+1:idx+1] = [list(to_spill_for_nonlayer)]

                    # *** Alles-oder-nichts: keine Evakuierung in diesem Microbatch ***
                    evac_plans.clear()

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                snapshot_defects(1)
                idx += 1
                continue

            # preIN-Pfade schneiden und Defekte prüfen (inkl. Kurz-Hop-Kanten)
            pre_in_paths: Dict[int, List[TimedNode]] = {}
            T_pre_sync = 0
            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]
                for qid in (a, b):
                    cut = DefaultRoutingPlanner._retime_until_pre_in_wait(
                        to_meeting_plans[qid], meet, 0
                    )
                    if cut is None:
                        to_meeting_plans.pop(a, None)
                        to_meeting_plans.pop(b, None)
                        fixed_meetings.pop(key, None)
                        replan_current_layer = True
                        break
                    pre_in_paths[qid] = cut
                    if cut:
                        T_pre_sync = max(T_pre_sync, cut[-1][1])

            # ★ bei Replan zurück zum Layer-Anfang
            if replan_current_layer:
                continue

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                snapshot_defects(1)
                idx += 1
                continue

            # Defekte auf Layer-PreIN-Pfaden oder den Hop-Kanten?
            to_spill_layer_defects: List[Tuple[int, int]] = []
            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]
                pa = pre_in_paths.get(a)
                pb = pre_in_paths.get(b)
                if pa is None or pb is None:
                    # Hier landen wir nicht mehr (würden oben replannen), lassen aber konservativ drin
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
                # ★ Ausnahme (Defekte) -> direkt Spillover
                for (a, b) in to_spill_layer_defects:
                    key = frozenset({a, b})
                    meet = fixed_meetings.get(key)
                    to_meeting_plans.pop(a, None)
                    to_meeting_plans.pop(b, None)
                    fixed_meetings.pop(key, None)
                layers[idx+1:idx+1] = [to_spill_layer_defects]

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                snapshot_defects(1)
                idx += 1
                continue

            # ===== Schritt 4: Ausführung in drei Microbatches =====
            # 4a) Non-Layer Evakuierung zuerst
            if evac_plans:
                evac_plans = {
                    qid: path
                    for qid, path in evac_plans.items()
                    if not DefaultRoutingPlanner._path_uses_defective_edge(path, defective_edges)
                }
                micro_evacuate: Dict[int, List[TimedNode]] = {}
                for qid, path in evac_plans.items():
                    micro_evacuate[qid] = path
                # Nicht betroffene warten bis Ende des Evakuierungsbatches
                dur = max((p[-1][1] for p in micro_evacuate.values()), default=0)
                for qid in (all_qids - set(micro_evacuate.keys())):
                    micro_evacuate[qid] = [(current_pos[qid], 0), (current_pos[qid], dur)]
                batch_plans.append(micro_evacuate)
                snapshot_defects(1)
                # Positionen aktualisieren
                for qid in evac_plans:
                    current_pos[qid] = evac_plans[qid][-1][0]

            # 4b) Layer-Anschnitt bis PRE-IN, synchronisiert auf T_pre_sync
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

            # Nicht-Layer + nicht-beteiligte Layer warten sichtbar
            others = all_qids - exec_layer_qids
            dur_pre = max((p[-1][1] for p in micro_to_pre.values()), default=0)
            for qid in others:
                micro_to_pre[qid] = [(current_pos[qid], 0), (current_pos[qid], dur_pre)]
            batch_plans.append(micro_to_pre)
            snapshot_defects(1)

            # Positionen aktualisieren (Layer stehen nun am PRE-IN)
            for qid in exec_layer_qids:
                current_pos[qid] = micro_to_pre[qid][-1][0]

            # 4c) kurzer IN-Hop (PRE-IN -> IN -> PRE-IN) für alle verbleibenden Paare
            micro_in: Dict[int, List[TimedNode]] = {}
            for (a, b) in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]
                pre_a = current_pos[a]
                pre_b = current_pos[b]
                if (frozenset({pre_a, meet}) in defective_edges) or (frozenset({pre_b, meet}) in defective_edges):
                    # Falls Zwischenzeit Defekt, beide warten 2 Ticks
                    micro_in[a] = [(pre_a, 0), (pre_a, 2)]
                    micro_in[b] = [(pre_b, 0), (pre_b, 2)]
                else:
                    micro_in[a] = [(pre_a, 0), (meet, 1), (pre_a, 2)]
                    micro_in[b] = [(pre_b, 0), (meet, 1), (pre_b, 2)]

            # Andere warten 2 Ticks
            rest = all_qids - {q for ab in layer_pairs if frozenset(ab) in fixed_meetings for q in ab}
            for qid in rest:
                micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]
            batch_plans.append(micro_in)
            snapshot_defects(1)
            idx += 1

        # Stitchen wie im Default
        return DefaultRoutingPlanner.stitch_batches(qubits, batch_plans, batch_defects)



    # ---------- Schritt 1: Layer-only Planung ----------
    @staticmethod
    def _plan_layer_only(
        G: nx.Graph,
        current_pos: Dict[int, Coord],
        layer_pairs: List[Tuple[int, int]],
        layer_starts: Set[Coord],
        defective_edges: Set[frozenset],
        banned_meetings: Optional[Dict[frozenset, Set[Coord]]] = None,
        all_ins: Optional[Set[Coord]] = None,
    ) -> Tuple[Dict[int, List[TimedNode]], Dict[frozenset, Coord], bool, List[Tuple[int, int]], List[Tuple[int, int]] ]:
        """
        Plane nur die Layer-Paare (kollisionsfrei) bis zu *besten* Meeting-INs:
          - Minimale *Gesamtdistanz* zu beiden Startknoten (Tie-break via maxdist, balance, Knoten-ID)
          - Einzigartige INs pro Paar
          - Pfade dürfen *keinen* Startknoten eines Layer-Qubits (außer eigenes Start) enthalten
          - Alle PRE-INs im Layer müssen unterschiedlich sein
          - 🔧 NEU: Pfade dürfen KEINE PRE-INs anderer Layer-Qubits im selben Layer enthalten
          - Non-Layer werden NICHT berücksichtigt
        """
        res = Reservations(G, blocked_edges=defective_edges)
        plans: Dict[int, List[TimedNode]] = {}
        fixed_meetings: Dict[frozenset, Coord] = {}
        unplaceable: List[Tuple[int, int]] = []

        banned_meetings = banned_meetings or {}
        all_ins = all_ins or set()
        exhausted_pairs: List[Tuple[int, int]] = []


        # Startknoten zum Blockieren in der Suche (außer eigener Start)
        def forbidden_for(qid: int) -> Set[Coord]:
            return set(layer_starts) - {current_pos[qid]}

        # Kandidatenreihenfolge je Paar vorbereiten
        cand_per_pair: Dict[Tuple[int, int], List[Coord]] = {}
        for (a, b) in layer_pairs:
            qa, qb = current_pos[a], current_pos[b]
            cand_per_pair[(a, b)] = DefaultRoutingPlanner._best_meeting_candidates(
                G, qa, qb, reserved=set(), forbidden_nodes=set()
            )

        # Greedy: Paare mit wenig Kandidaten zuerst
        order = sorted(layer_pairs, key=lambda ab: len(cand_per_pair.get(ab, [])))

        reserved_in: Set[Coord] = set()
        for (a, b) in order:
            qa, qb = current_pos[a], current_pos[b]
            placed = False

            # 🔧 Bereits akzeptierte PRE-INs anderer Layer-Qubits sammeln
            cur_preins_map = DefaultRoutingPlanner._preins_for_plans(plans, fixed_meetings)
            existing_preins: Set[Coord] = set(cur_preins_map.values()) if cur_preins_map else set()

            # Kandidaten dynamisch (unter Berücksichtigung bereits reservierter INs)
            cands_all = DefaultRoutingPlanner._best_meeting_candidates(
                G, qa, qb, reserved=reserved_in, forbidden_nodes=set()
            )
            banned = banned_meetings.get(frozenset({a, b}), set())
            cands = [c for c in cands_all if c not in banned]

            existing_path_nodes: Set[Coord] = set()
            for qid, p in plans.items():
                for c, _ in p:
                    existing_path_nodes.add(c)

            # Falls für dieses Paar *keine* Kandidaten mehr übrig sind: exhaustion merken
            if not cands and all_ins and len(banned) >= len(all_ins):
                exhausted_pairs.append((a, b))

            for meet in cands:
                # Suche kollisionsfrei (nur Layer) bis zum Meeting
                res_try = deepcopy(res)

                # 🔧 PRE-INs anderer bereits gesetzter Layer-Qubits blockieren
                if existing_preins:
                    DefaultRoutingPlanner._block_nodes(res_try, existing_preins)

                # A zuerst
                DefaultRoutingPlanner._block_nodes(res_try, forbidden_for(a))
                pa = AStar.search(G, qa, meet, res_try)
                if pa is None:
                    continue
                Reservations.commit(res_try, pa)

                # >>> NEU: pre_a sofort bestimmen und blockieren
                pre_a = DefaultRoutingPlanner._entry_sn_from_path(pa, meet)
                if pre_a is None:
                    continue

                if pre_a in existing_path_nodes:
                    continue
                DefaultRoutingPlanner._block_nodes(res_try, {pre_a})

                # (PRE-INs anderer bereits gesetzter Layer-Qubits zusätzlich blockiert lassen)
                if existing_preins:
                    DefaultRoutingPlanner._block_nodes(res_try, existing_preins)

                # B danach
                DefaultRoutingPlanner._block_nodes(res_try, forbidden_for(b))
                pb = AStar.search(G, qb, meet, res_try)
                if pb is None:
                    continue

                # PRE-INs extrahieren
                pre_b = DefaultRoutingPlanner._entry_sn_from_path(pb, meet)
                if pre_b is None:
                    continue

                if pre_b in existing_path_nodes:
                    continue

                # Eindeutigkeit im Layer prüfen (keine gleichen PRE-INs)
                tmp_plans = dict(plans); tmp_plans[a] = pa; tmp_plans[b] = pb
                tmp_fixed = dict(fixed_meetings); tmp_fixed[frozenset({a, b})] = meet
                preins = DefaultRoutingPlanner._preins_for_plans(tmp_plans, tmp_fixed)
                if preins is None:
                    continue
                prein_vals = [preins[qid] for qid in preins if qid in {x for ab in layer_pairs for x in ab}]
                if len(set(prein_vals)) != len(prein_vals):
                    continue

                # >>> WICHTIG: auch B reservieren (fehlte bisher in deinem Codepfad)
                Reservations.commit(res_try, pb)

                # akzeptieren
                res = res_try
                plans[a] = pa; plans[b] = pb
                fixed_meetings[frozenset({a, b})] = meet
                reserved_in.add(meet)
                placed = True
                break


            if not placed:
                unplaceable.append((a, b))

        # final prüfen, ob PRE-INs eindeutig (sollten es sein)
        preins_ok = True
        preins = DefaultRoutingPlanner._preins_for_plans(plans, fixed_meetings)
        if preins is None:
            preins_ok = False
        else:
            lv = [preins[qid] for qid in preins if qid in {x for ab in layer_pairs for x in ab}]
            preins_ok = (len(set(lv)) == len(lv))

        return plans, fixed_meetings, preins_ok, unplaceable, exhausted_pairs



    # ---------- Schritt 4 Helfer: kurzen IN-Hop + Vorbereitung ----------
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

    # ---------- Kandidatenwahl (minimale Gesamtdistanz) ----------
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
        # sortiere nach: Summe, dann max, dann Balance, dann Knoten-ID
        cands.sort(key=lambda n: (d0[n] + d1[n], max(d0[n], d1[n]), abs(d0[n] - d1[n]), n))
        return cands

    # ---------- PRE-IN Auswertung ----------
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
    def _prein_conflicts(
        plans: Dict[int, List[TimedNode]],
        fixed_meetings: Dict[frozenset, Coord],
    ) -> List[Tuple[int, int]]:
        """Liefert Liste von Paaren, deren PRE-IN Eindeutigkeit bricht."""
        pre = DefaultRoutingPlanner._preins_for_plans(plans, fixed_meetings)
        if pre is None:
            # alle Paare potenziell betroffen
            return [tuple(sorted(k)) for k in fixed_meetings.keys()]  # type: ignore
        seen: Dict[Coord, int] = {}
        conflicts: Set[Tuple[int, int]] = set()
        for pair_key in fixed_meetings.keys():
            a, b = list(pair_key)
            for qid in (a, b):
                pin = pre[qid]
                if pin in seen and seen[pin] != qid:
                    conflicts.add(tuple(sorted((qid, seen[pin]))))
                else:
                    seen[pin] = qid
        # Konflikte auf Paare mappen: grob/konservativ – jedes Paar, dessen Qubit beteiligt ist
        bad_pairs: Set[Tuple[int, int]] = set()
        for (x, y) in conflicts:
            for pair_key in fixed_meetings.keys():
                a, b = list(pair_key)
                if x in (a, b) or y in (a, b):
                    bad_pairs.add(tuple(sorted((a, b))))
        return list(bad_pairs)

    # ---------- Schneiden/Timing ----------
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
        new_path = path[: first_meet_idx]
        cur = t_pre
        while cur < sync_time:
            new_path.append((pre_in, cur + 1))
            cur += 1
        return new_path

    # ---------- MAPF Wrapper identisch zum Stil im Default ----------
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

        # Blockiere verbotene Knoten über gesamten Horizont
        for node in blocked_nodes:
            cap = res.node_capacity(node)
            for t in range(0, MAX_TIME + 1):
                res.node_caps[node][t] = cap

        plans: Dict[int, List[TimedNode]] = {}
        # Startknoten bei t=0 belegen
        for qid, s in starts.items():
            res.occupy_node(s, 0)

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

    # ---------- Defekt-Sampling & Prüfungen ----------
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

    # ---------- Utility: freies SN in BFS-Nähe ----------
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
                    seen.add(v); q.append(v)
        return None

    # ---------- Stitcher (delegiert an Default-Implementierung) ----------
    @staticmethod
    def _block_nodes(res: Reservations, nodes: Set[Coord]) -> None:
        for node in nodes:
            cap = res.node_capacity(node)
            for t in range(0, MAX_TIME + 1):
                # volle Auslastung -> für A* unbetretbar
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
        blocker_to_pair: Dict[int, Tuple[int, int]]
    ) -> Dict[int, List[TimedNode]]:
        """
        Rekursive/Chain-Handover-Auflösung:
        Findet entlang eines Mover-Pfads ggf. mehrere wartende Blocker in Reihenfolge.
        Bildet daraus eine Handover-Kette:
            mover -> b1 -> b2 -> ... -> bK -> (mover_target)
        und replante das *gesamte* Evakuierungsset (alle bisherigen Mover + beteiligte Blocker)
        gemeinsam via MAPF, um Kollisionen zuverlässig zu vermeiden.

        Fallback, falls Gesamt-Replan fehlschlägt: versuche einfachen Handover nur mit dem ersten Blocker.
        """
        if not evac_plans:
            return {}

        plans = dict(evac_plans)  # Kopie
        wait_pos: Dict[int, Coord] = {qid: current_pos[qid] for qid in waiting_qids}
        targets_local: Dict[int, Coord] = dict(targets)

        # Signatur anpassen: root_pair (des Movers) reinreichen
        def build_blocker_chain_along_path(
            path: List[TimedNode],
            root_pair: Optional[Tuple[int, int]],
        ) -> List[int]:
            chain: List[int] = []
            seen: Set[int] = set()

            # Wir betrachten die Zielknoten der Schritte (node_next)
            for (_, _t_prev), (node_next, _t_next) in zip(path[:-1], path[1:]):
                # Prüfe, ob node_next von einem wartenden Qubit besetzt ist
                for bqid, bpos in wait_pos.items():
                    if bqid in seen:
                        continue
                    if node_next == bpos:
                        chain.append(bqid)
                        seen.add(bqid)

                        # ★ WICHTIG: Paar-Zuordnung des Movers auf den Blocker übertragen
                        if root_pair is not None:
                            blocker_to_pair.setdefault(bqid, root_pair)

            return chain


        changed = True
        while changed:
            changed = False

            # Iteriere über eine Momentaufnahme, da wir 'plans' modifizieren könnten
            for mover_qid, mover_path in plans.items():
                root_pair = blocker_to_pair.get(mover_qid)
                # Finde Blocker entlang des Mover-Pfads (in Reihenfolge des Auftretens)
                chain_blockers = build_blocker_chain_along_path(mover_path, root_pair)
                if not chain_blockers:
                    continue

                # --- Versuche: Replan für das *gesamte* Evakuierungsset + Chain-Blocker ---
                                # --- Versuche: Replan für das *gesamte* Evakuierungsset + Chain-Blocker ---
                # Plan the union of current movers and chain blockers, but only if each has a target.
                chain_agents: List[int] = [mover_qid] + chain_blockers
                agents_to_plan: Set[int] = set(plans.keys()) | set(chain_blockers)

                # Starts for all agents we intend to plan now
                starts: Dict[int, Coord] = {qid: current_pos[qid] for qid in agents_to_plan}

                # Base targets: whatever we already have for those agents
                new_targets_full: Dict[int, Coord] = {
                    qid: targets_local[qid] for qid in agents_to_plan if qid in targets_local
                }

                # Chain assignments: mover -> pos(b1), b1 -> pos(b2), ..., last -> original mover target
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

                # Every agent we plan must have a target (avoids KeyError downstream)
                if valid_chain and agents_to_plan.issubset(new_targets_full.keys()):
                    try:
                        replanned = DefaultRoutingPlanner._mapf_to_targets(
                            G=G,
                            starts=starts,
                            targets=new_targets_full,
                            blocked_nodes=blocked_nodes,
                            blocked_edges=defective_edges,
                        )
                        # Success: update plans and persist the targets for these agents
                        for qid, path in replanned.items():
                            plans[qid] = path
                            targets_local[qid] = new_targets_full[qid]

                        # Chain blockers are no longer "waiting"
                        for bqid in chain_blockers:
                            wait_pos.pop(bqid, None)

                        changed = True
                        break

                    except RuntimeError:
                        # Fallback: simple two-agent handover (mover <-> first blocker)
                        b1 = chain_blockers[0]
                        try:
                            partial = DefaultRoutingPlanner._mapf_to_targets(
                                G=G,
                                starts={mover_qid: current_pos[mover_qid], b1: current_pos[b1]},
                                targets={mover_qid: current_pos[b1], b1: targets_local.get(mover_qid, current_pos[b1])},
                                blocked_nodes=blocked_nodes,
                                blocked_edges=defective_edges,
                            )
                            plans[mover_qid] = partial[mover_qid]
                            plans[b1] = partial[b1]
                            # Persist targets for both so future iterations remain consistent
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
    def _debug_print_meetings(layer_idx: int, fixed_meetings: Dict[frozenset, Coord], header: str = "") -> None:
        if header:
            print(header)
        if not fixed_meetings:
            print(f"[Layer {layer_idx}] Keine Meeting-INs fixiert.")
            return
        print(f"[Layer {layer_idx}] Fixierte Meeting-INs:")
        # Stabil und lesbar: Paare sortiert ausgeben
        for pair_key in sorted(fixed_meetings.keys(), key=lambda k: tuple(sorted(k))):
            a, b = sorted(pair_key)
            meet = fixed_meetings[pair_key]
            print(f"  Paar ({a}, {b}) -> IN {meet}")
