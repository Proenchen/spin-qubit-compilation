from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import random
import networkx as nx

from routing.common import Coord, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner

P_SUCCESS = 0.998
P_REPAIR = 0.04


def chebyshev(p: Coord, q: Coord) -> int:
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def _edgeset(u: Coord, v: Coord) -> frozenset:
    return frozenset({u, v})


# ============================================================
#  NEUE IMPLEMENTIERUNG: Rotation über Kreise (8-Zyklen)
# ============================================================
class CircleRotationRoutingPlanner:
    """
    Variante des RotationRoutingPlanner, bei der statt 4er-Diamanten
    größere Kreise (8-Zyklen über zwei Rauten) verwendet werden.

    ACHTUNG:
    Diese Klasse ist "pur" die Kreis-Variante.
    Die gewünschte Logik "erst alt, dann neu" steckt in
    HybridRotationRoutingPlanner (s.u.).
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

        # Timelines
        timelines: Dict[int, List[TimedNode]] = {q.id: [(current_pos[q.id], 0)] for q in qubits}
        t = 0  # globale Zeit

        defective_edges: Set[frozenset] = set()
        edge_timebands: List[Tuple[int, int, Set[frozenset]]] = []

        # ---------- Knotentyp ----------
        def _is_sn(n: Coord) -> bool:
            return G.nodes[n].get("type") == "SN"

        def _is_in(n: Coord) -> bool:
            return G.nodes[n].get("type") == "IN"

        # ---------- Sampling & Tick ----------
        def _sample_edge_failures():
            for u, v in G.edges():
                e = _edgeset(u, v)
                if e in defective_edges:
                    if random.random() < p_repair:
                        defective_edges.discard(e)
                else:
                    if random.random() < (1.0 - p_success):
                        defective_edges.add(e)

        def _would_use_defect(pending: Dict[int, Coord]) -> bool:
            for qid, newp in pending.items():
                u = current_pos[qid]
                v = newp
                if u != v and _edgeset(u, v) in defective_edges:
                    return True
            return False

        def _commit_tick(pending: Dict[int, Coord], *, sample: bool) -> bool:
            nonlocal t
            if sample:
                _sample_edge_failures()
            moved = False
            if pending and not _would_use_defect(pending):
                for qid, newp in pending.items():
                    current_pos[qid] = newp
                moved = True
            # Zeit schreitet immer fort (Wartetick sonst)
            t += 1
            for qid in all_qids:
                last = timelines[qid][-1]
                cur = (current_pos[qid], t)
                if last != cur:
                    timelines[qid].append(cur)
            edge_timebands.append((t - 1, t, set(defective_edges)))
            return moved

        # ---------- Rauten / Kreise / Rotation ----------
        def _is_diag(u: Coord, v: Coord) -> bool:
            return abs(u[0] - v[0]) == 1 and abs(u[1] - v[1]) == 1

        # ---------- SN-Subgraph ----------
        sn_nodes = [n for n in G.nodes() if _is_sn(n)]
        SN = G.subgraph(sn_nodes).copy()

        def _sn_neighbors_of_meet(meeting: Coord) -> List[Coord]:
            if not _is_in(meeting):
                return []
            return [w for w in G.neighbors(meeting) if _is_sn(w)]

        def _shortest_path_sn(src: Coord, dst: Coord) -> Optional[List[Coord]]:
            try:
                return nx.shortest_path(SN, src, dst)
            except nx.NetworkXNoPath:
                return None

        # ---------- Generische Rotation auf Loops ----------
        def _canonical_loop_tuple(D: List[Coord]) -> Tuple[Coord, ...]:
            return tuple(sorted(D))

        def _rot_dir(loop: List[Coord], u: Coord, v: Coord) -> int:
            i = loop.index(u)
            n = len(loop)
            if loop[(i + 1) % n] == v:
                return +1
            elif loop[(i - 1) % n] == v:
                return -1
            else:
                raise ValueError(f"{u}->{v} ist keine Nachbarschaft im Loop")

        def _compute_pair_rotation_updates_for_loop(
            loop: List[Coord],
            direction: int,
            a_id: int,
            b_id: int,
            la: Coord,
            lb: Coord,
        ) -> Dict[int, Coord]:
            idx = {p: i for i, p in enumerate(loop)}
            out: Dict[int, Coord] = {}
            n = len(loop)

            if la in idx:
                out[a_id] = loop[(idx[la] + direction) % n]
            if lb in idx:
                out[b_id] = loop[(idx[lb] + direction) % n]
            return out

        def _circle_for_edge(u: Coord, v: Coord, target_len: int = 8) -> Optional[List[Coord]]:
            """
            Suche einen einfachen Zyklus der Länge `target_len` im SN-Subgraphen,
            der die Kante (u, v) enthält. Wir konstruieren dafür einen einfachen
            Pfad von v nach u mit genau target_len Knoten (d. h. target_len-1 Kanten),
            wobei die direkte Kante (u, v) nur zum Schluss verwendet wird.

            Rückgabeformat: [u, v, ..., node] – zyklische Liste der Knoten.
            """
            if not (_is_sn(u) and _is_sn(v)):
                return None
            if not SN.has_edge(u, v):
                return None

            start = v
            max_nodes = target_len

            # DFS mit Längenbegrenzung auf target_len Knoten (inkl. u)
            stack: List[Tuple[Coord, List[Coord]]] = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) >= max_nodes:
                    continue

                for w in SN.neighbors(node):
                    if w == u:
                        full_path = path + [u]  # [v, ..., node, u]
                        if len(full_path) == max_nodes:
                            # Zyklus-Order: [u, v, ..., node]
                            loop = [u] + path  # u + [v, ..., node]
                            return loop
                        else:
                            continue

                    if w in path or w == u:
                        continue

                    if len(path) + 1 <= max_nodes:
                        stack.append((w, path + [w]))

            return None

        # ---------- Solo-Plan für ein Paar ----------
        class SoloStep:
            __slots__ = ("updates_pair_only", "sample", "diamonds")

            def __init__(self, updates_pair_only: Dict[int, Coord], sample: bool, diamonds: List[Tuple[List[Coord], int]]):
                self.updates_pair_only = updates_pair_only
                self.sample = sample
                # diamonds: hier allgemeine Loops (Kreise), Name bleibt für Kompatibilität
                self.diamonds = diamonds

        class SoloPlan:
            def __init__(
                self,
                ticks: List[SoloStep],
                pos_trace: Dict[int, List[Coord]],
                used_diamonds: Set[Tuple[Coord, ...]],
                in_idx: Optional[int],
                out_idx: Optional[int],
            ):
                self.ticks = ticks
                self.pos_trace = pos_trace
                self.used_diamonds = used_diamonds
                self.in_idx = in_idx
                self.out_idx = out_idx

            @property
            def length(self) -> int:
                return len(self.ticks)

        def _plan_pair_solo(a_id: int, b_id: int) -> Optional[SoloPlan]:
            la = current_pos[a_id]
            lb = current_pos[b_id]
            if not (_is_sn(la) and _is_sn(lb)):
                return None

            ticks: List[SoloStep] = []
            trace = {a_id: [la], b_id: [lb]}
            used_diamonds: Set[Tuple[Coord, ...]] = set()
            in_idx: Optional[int] = None
            out_idx: Optional[int] = None

            cands = DefaultRoutingPlanner._best_meeting_candidates(
                G, la, lb, reserved=set(), forbidden_nodes=set()
            )

            def _best_pre(meet: Coord, src: Coord) -> Optional[Tuple[Coord, List[Coord]]]:
                best = None
                for pre in _sn_neighbors_of_meet(meet):
                    p = _shortest_path_sn(src, pre)
                    if p is None:
                        continue
                    if best is None or len(p) < len(best[1]):
                        best = (pre, p)
                return best

            chosen = None
            for m in cands:
                if not _is_in(m):
                    continue
                pa = _best_pre(m, la)
                pb = _best_pre(m, lb)
                if pa and pb:
                    chosen = (m, pa, pb)
                    break
            if not chosen:
                return None

            meet, (preA, pathA), (preB, pathB) = chosen
            idxA = 0
            idxB = 0

            # bis PRE-A/B
            while la != preA or lb != preB:
                updates: Dict[int, Coord] = {}
                diamonds_for_step: List[Tuple[List[Coord], int]] = []

                if la != preA and idxA + 1 < len(pathA):
                    uA, vA = pathA[idxA], pathA[idxA + 1]
                    if _is_diag(uA, vA):
                        loopA = _circle_for_edge(uA, vA)
                        if loopA:
                            dirA = _rot_dir(loopA, uA, vA)
                            updA = _compute_pair_rotation_updates_for_loop(
                                loopA, dirA, a_id, b_id, la, lb
                            )
                            # A soll in diesem Tick auf vA landen
                            updA[a_id] = vA
                            updates.update(updA)
                            diamonds_for_step.append((loopA, dirA))
                            used_diamonds.add(_canonical_loop_tuple(loopA))
                        else:
                            updates[a_id] = vA
                    else:
                        updates[a_id] = vA

                if lb != preB and idxB + 1 < len(pathB):
                    uB, vB = pathB[idxB], pathB[idxB + 1]
                    if _is_diag(uB, vB):
                        loopB = _circle_for_edge(uB, vB)
                        if loopB:
                            dirB = _rot_dir(loopB, uB, vB)
                            updB = _compute_pair_rotation_updates_for_loop(
                                loopB, dirB, a_id, b_id, la, lb
                            )
                            updB[b_id] = vB
                            # Verhindere Konflikt, wenn in diesem Tick schon ein anderer Loop Knoten teilt
                            if not (diamonds_for_step and set(diamonds_for_step[0][0]).intersection(loopB)):
                                updates.update(updB)
                                diamonds_for_step.append((loopB, dirB))
                                used_diamonds.add(_canonical_loop_tuple(loopB))
                        else:
                            if b_id not in updates:
                                updates[b_id] = vB
                    else:
                        if b_id not in updates:
                            updates[b_id] = vB

                if not updates:
                    return None

                la = updates.get(a_id, la)
                lb = updates.get(b_id, lb)
                if a_id in updates:
                    idxA = min(idxA + 1, len(pathA) - 1)
                if b_id in updates:
                    idxB = min(idxB + 1, len(pathB) - 1)

                ticks.append(
                    SoloStep(
                        updates_pair_only=updates,
                        sample=True,
                        diamonds=diamonds_for_step,
                    )
                )
                trace[a_id].append(la)
                trace[b_id].append(lb)

            # PRE->IN
            updates = {}
            if la != meet:
                updates[a_id] = meet
            if lb != meet:
                updates[b_id] = meet
            if updates:
                la = updates.get(a_id, la)
                lb = updates.get(b_id, lb)
                in_idx = len(ticks)
                ticks.append(SoloStep(updates_pair_only=updates, sample=True, diamonds=[]))
                trace[a_id].append(la)
                trace[b_id].append(lb)

            # IN->PRE (wird live gesetzt)
            out_idx = len(ticks)
            ticks.append(SoloStep(updates_pair_only={}, sample=False, diamonds=[]))
            trace[a_id].append(preA)
            trace[b_id].append(preB)

            return SoloPlan(ticks, trace, used_diamonds, in_idx, out_idx)

        # ---------- Kompatibilität der Solo-Pläne ----------
        def _plans_compatible_distance(
            p1: SoloPlan,
            ab1: Tuple[int, int],
            p2: SoloPlan,
            ab2: Tuple[int, int],
        ) -> bool:
            a1, b1 = ab1
            a2, b2 = ab2
            L = max(p1.length, p2.length)

            def _pos(plan: SoloPlan, qid: int, i: int) -> Coord:
                if i < len(plan.pos_trace[qid]):
                    return plan.pos_trace[qid][i]
                return plan.pos_trace[qid][-1]

            for i in range(0, L + 1):
                p_a1 = _pos(p1, a1, i)
                p_b1 = _pos(p1, b1, i)
                p_a2 = _pos(p2, a2, i)
                p_b2 = _pos(p2, b2, i)
                for u in (p_a1, p_b1):
                    for v in (p_a2, p_b2):
                        if chebyshev(u, v) < 3:
                            return False
            return True

        def _plans_compatible_diamonds(p1: SoloPlan, p2: SoloPlan) -> bool:
            """
            Zwei Solo-Pläne sind loop-kompatibel, wenn sie in keinem Tick
            *gleichzeitig* denselben Loop benutzen.
            """
            L = max(p1.length, p2.length)

            for i in range(L):
                if i < p1.length:
                    d1 = {
                        _canonical_loop_tuple(D)
                        for (D, _dir) in p1.ticks[i].diamonds
                    }
                else:
                    d1 = set()

                if i < p2.length:
                    d2 = {
                        _canonical_loop_tuple(D)
                        for (D, _dir) in p2.ticks[i].diamonds
                    }
                else:
                    d2 = set()

                if not d1.isdisjoint(d2):
                    return False

            return True

        # ---------- Gruppenbildung ----------
        def _group_parallel(plans: Dict[Tuple[int, int], SoloPlan]) -> List[List[Tuple[int, int]]]:
            remaining_pids = sorted(plans.keys())
            groups: List[List[Tuple[int, int]]] = []
            used: Set[Tuple[int, int]] = set()
            for pid in remaining_pids:
                if pid in used:
                    continue
                grp = [pid]
                used.add(pid)
                for qid in remaining_pids:
                    if qid in used:
                        continue
                    ok = all(
                        _plans_compatible_distance(plans[pid0], pid0, plans[qid], qid)
                        and _plans_compatible_diamonds(plans[pid0], plans[qid])
                        for pid0 in grp
                    )
                    if ok:
                        grp.append(qid)
                        used.add(qid)
                groups.append(grp)
            return groups

        # ---------- Laufzeit-Helfer: Rotationen expandieren ----------
        def _expand_runtime_rotations(
            base_updates: Dict[int, Coord],
            diamonds: List[Tuple[List[Coord], int]],
            exclude_qids: Set[int] = set(),
        ) -> Dict[int, Coord]:
            if not diamonds:
                return dict(base_updates)
            idx_cache: Dict[Tuple[Coord, ...], Dict[Coord, int]] = {}
            out = dict(base_updates)
            for D, direction in diamonds:
                key = tuple(D)
                if key not in idx_cache:
                    idx_cache[key] = {p: i for i, p in enumerate(D)}
                idx = idx_cache[key]
                n = len(D)
                for qid, pos in current_pos.items():
                    if qid in exclude_qids:
                        continue
                    if pos in idx:
                        out[qid] = D[(idx[pos] + direction) % n]
            return out

        def _foreign_qubits_on_any_diamond(
            diamonds: List[Tuple[List[Coord], int]],
            allowed: Set[int],
        ) -> bool:
            if not diamonds:
                return False
            nodes: Set[Coord] = set()
            for D, _ in diamonds:
                nodes |= set(D)
            for qid, pos in current_pos.items():
                if qid in allowed:
                    continue
                if pos in nodes:
                    return True
            return False

        # ---- Live-Map: PRE-SN zum Zeitpunkt eines *erfolgreichen* IN-Hops (pro Paar) ----
        live_pre_by_pair: Dict[Tuple[int, int], Dict[int, Coord]] = {}

        # ---- Interaktions-Reihenfolge: Index der Paare im Original-Array ----
        pair_order: Dict[Tuple[int, int], int] = {}
        for idx, (qa, qb) in enumerate(pairs):
            pid = (qa.id, qb.id)
            pair_order[pid] = idx

        def _is_ready_pair(pid: Tuple[int, int], remaining_pids: Set[Tuple[int, int]]) -> bool:
            idx = pair_order[pid]
            a, b = pid
            for other in remaining_pids:
                if other == pid:
                    continue
                j = pair_order[other]
                if j < idx:
                    x, y = other
                    if x == a or x == b or y == a or y == b:
                        return False
            return True

        # ===================== Hauptroutine =====================
        remaining: Set[Tuple[int, int]] = {(qa.id, qb.id) for qa, qb in pairs}

        while remaining:
            # 1) Nur "ready" Paare dürfen in diesem Durchlauf geplant werden
            ready_pids = [pid for pid in remaining if _is_ready_pair(pid, remaining)]

            # Failsafe: falls wegen inkonsistenter Daten nichts ready ist, nimm alles
            if not ready_pids:
                ready_pids = list(remaining)

            # 2) Solo-Pläne für ready-Paare
            plans: Dict[Tuple[int, int], SoloPlan] = {}
            sequential_fallback: List[Tuple[int, int]] = []
            for pid in sorted(ready_pids, key=lambda p: pair_order[p]):
                a, b = pid
                plan = _plan_pair_solo(a, b)
                if plan is None:
                    sequential_fallback.append((a, b))
                else:
                    plans[pid] = plan

            if not plans and not sequential_fallback:
                _commit_tick({}, sample=True)
                break

            # 3) Parallelgruppen
            groups = _group_parallel(plans) if plans else []

            something_finished = False

            for grp in groups:
                group_qids: Set[int] = {x for ab in grp for x in ab}
                L = max(plans[pid].length for pid in grp) if grp else 0
                parallel_failed = False

                # --- Parallel-Replay MIT Retry bei Defekt ---
                step = 0
                while step < L:
                    updates_pair_only: Dict[int, Coord] = {}
                    sample_flags: List[bool] = []
                    step_diamonds: List[Tuple[List[Coord], int]] = []
                    conflict = False

                    # temporäre PREs für IN-Hops dieses Steps (werden nur bei Erfolg übernommen)
                    pending_live_pre: Dict[Tuple[int, int], Dict[int, Coord]] = {}

                    for pid in grp:
                        plan = plans[pid]
                        if step < plan.length:
                            s = plan.ticks[step]
                        else:
                            s = SoloStep({}, True, [])

                        # IN-Hop: PRE vor Commit zwischenspeichern
                        if plan.in_idx is not None and step == plan.in_idx:
                            a_id, b_id = pid
                            pending_live_pre[pid] = {
                                a_id: current_pos[a_id],
                                b_id: current_pos[b_id],
                            }

                        # OUT-Hop: Ziel aus bereits gemerkten PREs (falls nicht vorhanden, failsafe: aktuelle Pos)
                        if plan.out_idx is not None and step == plan.out_idx:
                            a_id, b_id = pid
                            pre_map = live_pre_by_pair.get(
                                pid,
                                {a_id: current_pos[a_id], b_id: current_pos[b_id]},
                            )
                            s = SoloStep(
                                {a_id: pre_map[a_id], b_id: pre_map[b_id]},
                                False,
                                [],
                            )

                        # Kollision gleicher Qubits?
                        if set(updates_pair_only.keys()) & set(s.updates_pair_only.keys()):
                            conflict = True
                            break
                        updates_pair_only.update(s.updates_pair_only)
                        sample_flags.append(s.sample)
                        step_diamonds.extend(s.diamonds)

                    if conflict:
                        parallel_failed = True
                        break

                    if _foreign_qubits_on_any_diamond(step_diamonds, allowed=group_qids):
                        parallel_failed = True
                        break

                    updates = _expand_runtime_rotations(
                        updates_pair_only,
                        step_diamonds,
                    )
                    do_sample = False not in sample_flags

                    moved = _commit_tick(updates, sample=do_sample)
                    if moved:
                        # IN-Hop(e) dieses Steps waren erfolgreich -> PREs jetzt fest übernehmen
                        for pid, pre in pending_live_pre.items():
                            live_pre_by_pair[pid] = pre
                        step += 1  # NÄCHSTER Step
                    else:
                        # Defekt -> RETRY gleicher Step im nächsten Tick
                        continue

                if parallel_failed:
                    # --- Sequentieller Fallback MIT Retry bei Defekt ---
                    for pid in grp:
                        plan = plans[pid]
                        a_id, b_id = pid
                        step = 0
                        while step < plan.length:
                            s = plan.ticks[step]
                            # IN-Hop -> PRE vor Commit zwischenspeichern, aber erst bei Erfolg übernehmen
                            pending_pre = None
                            if plan.in_idx is not None and step == plan.in_idx:
                                pending_pre = {
                                    a_id: current_pos[a_id],
                                    b_id: current_pos[b_id],
                                }

                            # OUT-Hop -> live PRE nutzen
                            if plan.out_idx is not None and step == plan.out_idx:
                                pre_map = live_pre_by_pair.get(
                                    pid,
                                    {a_id: current_pos[a_id], b_id: current_pos[b_id]},
                                )
                                s = SoloStep(
                                    {a_id: pre_map[a_id], b_id: pre_map[b_id]},
                                    False,
                                    [],
                                )

                            updates = _expand_runtime_rotations(
                                s.updates_pair_only,
                                s.diamonds,
                            )
                            moved = _commit_tick(updates, sample=s.sample)
                            if moved:
                                if pending_pre is not None:
                                    live_pre_by_pair[pid] = pending_pre
                                step += 1
                            else:
                                # Defekt -> RETRY selben Step
                                continue

                # Gruppe abgeschlossen -> Paare entfernen und REPLAN starten
                for pid in grp:
                    if pid in remaining:
                        remaining.remove(pid)
                something_finished = True
                break  # replan

            # Nichts fertig geworden -> sequential_fallback (ein ready Paar), mit Retry
            if not something_finished:
                if sequential_fallback:
                    a, b = sequential_fallback[0]
                    plan = _plan_pair_solo(a, b)
                    if plan is None:
                        _commit_tick({}, sample=True)
                    else:
                        pid = (a, b)
                        step = 0
                        while step < plan.length:
                            s = plan.ticks[step]
                            pending_pre = None
                            if plan.in_idx is not None and step == plan.in_idx:
                                pending_pre = {
                                    a: current_pos[a],
                                    b: current_pos[b],
                                }
                            if plan.out_idx is not None and step == plan.out_idx:
                                pre_map = live_pre_by_pair.get(
                                    pid,
                                    {a: current_pos[a], b: current_pos[b]},
                                )
                                s = SoloStep(
                                    {a: pre_map[a], b: pre_map[b]},
                                    False,
                                    [],
                                )
                            updates = _expand_runtime_rotations(
                                s.updates_pair_only,
                                s.diamonds,
                            )
                            moved = _commit_tick(updates, sample=s.sample)
                            if moved:
                                if pending_pre is not None:
                                    live_pre_by_pair[pid] = pending_pre
                                step += 1
                            else:
                                continue
                    if (a, b) in remaining:
                        remaining.remove((a, b))
                else:
                    _commit_tick({}, sample=True)

        return timelines, edge_timebands


# ============================================================
#  HILFSFUNKTION: erkennen, ob es Defekt-Warte-Ticks gab
# ============================================================
def _has_defect_wait(
    timelines: Dict[int, List[TimedNode]],
    edge_timebands: List[Tuple[int, int, Set[frozenset]]],
) -> bool:
    """
    Heuristik:
    - Für jeden Tick (k -> k+1) schauen wir,
      ob es defekte Kanten gibt UND ob kein Qubit seine Position ändert.
    - Wenn ja, interpretieren wir das als "Warten wegen Defekt".
    """
    if not edge_timebands:
        return False

    T = len(edge_timebands)
    qids = list(timelines.keys())

    # Wir gehen davon aus: für jedes qid hat timelines[qid] genau T+1 Einträge:
    # t=0,1,...,T
    for k in range(T):
        _t0, _t1, defects = edge_timebands[k]
        if not defects:
            continue

        any_moved = False
        for qid in qids:
            if k + 1 >= len(timelines[qid]):
                continue
            pos_before = timelines[qid][k][0]
            pos_after = timelines[qid][k + 1][0]
            if pos_before != pos_after:
                any_moved = True
                break

        if not any_moved:
            # Defekte Kanten vorhanden UND niemand bewegt sich -> Defekt-Warte-Tick
            return True

    return False


class HybridRotationRoutingPlanner:
    """
    Hybrid-Router:

    - Standard: Verhalten wie RotationRoutingPlanner (Diamonds).
    - Wenn ein Paar im sequentiellen Fallback an einer defekten Kante blockiert
      wird, wird im selben Algorithmus-Schritt versucht, einen Solo-Plan
      mit Kreisen (8-Zyklen) zu finden (Circle-Variante).
    - Falls das klappt, wird NUR für dieses Paar der Circle-Plan ausgeführt.
      Danach geht es für andere Paare wieder normal mit Diamond-Rotation weiter.
    - Falls auch der Circle-Plan nicht existiert, wird einfach gewartet
      (Warteticks, bis Kanten wieder repariert wurden).
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

        # Timelines
        timelines: Dict[int, List[TimedNode]] = {q.id: [(current_pos[q.id], 0)] for q in qubits}
        t = 0  # globale Zeit

        defective_edges: Set[frozenset] = set()
        edge_timebands: List[Tuple[int, int, Set[frozenset]]] = []

        # ---------- Knotentyp ----------
        def _is_sn(n: Coord) -> bool:
            return G.nodes[n].get("type") == "SN"

        def _is_in(n: Coord) -> bool:
            return G.nodes[n].get("type") == "IN"

        # ---------- Sampling & Tick ----------
        def _sample_edge_failures():
            for u, v in G.edges():
                e = _edgeset(u, v)
                if e in defective_edges:
                    if random.random() < p_repair:
                        defective_edges.discard(e)
                else:
                    if random.random() < (1.0 - p_success):
                        defective_edges.add(e)

        def _would_use_defect(pending: Dict[int, Coord]) -> bool:
            for qid, newp in pending.items():
                u = current_pos[qid]
                v = newp
                if u != v and _edgeset(u, v) in defective_edges:
                    return True
            return False

        def _commit_tick(pending: Dict[int, Coord], *, sample: bool) -> bool:
            nonlocal t
            if sample:
                _sample_edge_failures()
            moved = False
            if pending and not _would_use_defect(pending):
                for qid, newp in pending.items():
                    current_pos[qid] = newp
                moved = True
            # Zeit schreitet immer fort (Wartetick sonst)
            t += 1
            for qid in all_qids:
                last = timelines[qid][-1]
                cur = (current_pos[qid], t)
                if last != cur:
                    timelines[qid].append(cur)
            edge_timebands.append((t - 1, t, set(defective_edges)))
            return moved

        # ---------- Geometrie / Rauten / Kreise ----------
        def _is_diag(u: Coord, v: Coord) -> bool:
            return abs(u[0] - v[0]) == 1 and abs(u[1] - v[1]) == 1

        # --- Diamond (RotationRoutingPlanner-Teil) ---
        def _diag_sn_neighbors(n: Coord) -> List[Coord]:
            if not _is_sn(n):
                return []
            return [w for w in G.neighbors(n) if _is_sn(w) and _is_diag(n, w)]

        def _diamond_for_edge(u: Coord, v: Coord) -> Optional[List[Coord]]:
            if not (_is_sn(u) and _is_sn(v) and _is_diag(u, v)):
                return None
            Su = [w for w in _diag_sn_neighbors(u) if w != v]
            Sv = [x for x in _diag_sn_neighbors(v) if x != u]
            for w in Su:
                for x in Sv:
                    if _is_diag(w, x):
                        return [u, v, x, w]
            return None

        def _rot_dir_diamond(diamond: List[Coord], u: Coord, v: Coord) -> int:
            i = diamond.index(u)
            return +1 if diamond[(i + 1) % 4] == v else -1

        # ---------- SN-Subgraph ----------
        sn_nodes = [n for n in G.nodes() if _is_sn(n)]
        SN = G.subgraph(sn_nodes).copy()

        def _sn_neighbors_of_meet(meeting: Coord) -> List[Coord]:
            if not _is_in(meeting):
                return []
            return [w for w in G.neighbors(meeting) if _is_sn(w)]

        def _shortest_path_sn(src: Coord, dst: Coord) -> Optional[List[Coord]]:
            try:
                return nx.shortest_path(SN, src, dst)
            except nx.NetworkXNoPath:
                return None

        # ---------- Solo-Plan für ein Paar ----------
        class SoloStep:
            __slots__ = ("updates_pair_only", "sample", "diamonds")

            def __init__(self, updates_pair_only: Dict[int, Coord], sample: bool,
                         diamonds: List[Tuple[List[Coord], int]]):
                self.updates_pair_only = updates_pair_only
                self.sample = sample
                # diamonds: Liste (Loop/Diamond, Richtung), wird zur Laufzeit expandiert
                self.diamonds = diamonds

        class SoloPlan:
            def __init__(
                self,
                ticks: List[SoloStep],
                pos_trace: Dict[int, List[Coord]],
                used_diamonds: Set[Tuple[Coord, ...]],
                in_idx: Optional[int],
                out_idx: Optional[int],
            ):
                self.ticks = ticks
                self.pos_trace = pos_trace
                self.used_diamonds = used_diamonds
                self.in_idx = in_idx
                self.out_idx = out_idx

            @property
            def length(self) -> int:
                return len(self.ticks)

        def _canonical_diamond_tuple(D: List[Coord]) -> Tuple[Coord, ...]:
            return tuple(sorted(D))

        # --- Diamond-Rotationsupdates (für Standard-Router) ---
        def _compute_pair_rotation_updates_for_diamond(
            diamond: List[Coord],
            direction: int,
            a_id: int,
            b_id: int,
            la: Coord,
            lb: Coord,
        ) -> Dict[int, Coord]:
            idx = {p: i for i, p in enumerate(diamond)}
            out: Dict[int, Coord] = {}
            if la in idx:
                out[a_id] = diamond[(idx[la] + direction) % 4]
            if lb in idx:
                out[b_id] = diamond[(idx[lb] + direction) % 4]
            return out

        # ---------- Diamond-Solo-Plan (RotationRoutingPlanner-Logik) ----------
        def _plan_pair_solo(a_id: int, b_id: int) -> Optional[SoloPlan]:
            la = current_pos[a_id]
            lb = current_pos[b_id]
            if not (_is_sn(la) and _is_sn(lb)):
                return None

            ticks: List[SoloStep] = []
            trace = {a_id: [la], b_id: [lb]}
            used_diamonds: Set[Tuple[Coord, ...]] = set()
            in_idx: Optional[int] = None
            out_idx: Optional[int] = None

            cands = DefaultRoutingPlanner._best_meeting_candidates(
                G, la, lb, reserved=set(), forbidden_nodes=set()
            )

            def _best_pre(meet: Coord, src: Coord) -> Optional[Tuple[Coord, List[Coord]]]:
                best = None
                for pre in _sn_neighbors_of_meet(meet):
                    p = _shortest_path_sn(src, pre)
                    if p is None:
                        continue
                    if best is None or len(p) < len(best[1]):
                        best = (pre, p)
                return best

            chosen = None
            for m in cands:
                if not _is_in(m):
                    continue
                pa = _best_pre(m, la)
                pb = _best_pre(m, lb)
                if pa and pb:
                    chosen = (m, pa, pb)
                    break
            if not chosen:
                return None

            meet, (preA, pathA), (preB, pathB) = chosen
            idxA = 0
            idxB = 0

            # bis PRE-A/B
            while la != preA or lb != preB:
                updates: Dict[int, Coord] = {}
                diamonds_for_step: List[Tuple[List[Coord], int]] = []

                if la != preA and idxA + 1 < len(pathA):
                    uA, vA = pathA[idxA], pathA[idxA + 1]
                    if _is_diag(uA, vA):
                        dA = _diamond_for_edge(uA, vA)
                        if dA:
                            dirA = _rot_dir_diamond(dA, uA, vA)
                            updA = _compute_pair_rotation_updates_for_diamond(
                                dA, dirA, a_id, b_id, la, lb
                            )
                            updA[a_id] = vA
                            updates.update(updA)
                            diamonds_for_step.append((dA, dirA))
                            used_diamonds.add(_canonical_diamond_tuple(dA))
                        else:
                            updates[a_id] = vA
                    else:
                        updates[a_id] = vA

                if lb != preB and idxB + 1 < len(pathB):
                    uB, vB = pathB[idxB], pathB[idxB + 1]
                    if _is_diag(uB, vB):
                        dB = _diamond_for_edge(uB, vB)
                        if dB:
                            dirB = _rot_dir_diamond(dB, uB, vB)
                            updB = _compute_pair_rotation_updates_for_diamond(
                                dB, dirB, a_id, b_id, la, lb
                            )
                            updB[b_id] = vB
                            if not (diamonds_for_step and set(diamonds_for_step[0][0]).intersection(dB)):
                                updates.update(updB)
                                diamonds_for_step.append((dB, dirB))
                                used_diamonds.add(_canonical_diamond_tuple(dB))
                        else:
                            if b_id not in updates:
                                updates[b_id] = vB
                    else:
                        if b_id not in updates:
                            updates[b_id] = vB

                if not updates:
                    return None

                la = updates.get(a_id, la)
                lb = updates.get(b_id, lb)
                if a_id in updates:
                    idxA = min(idxA + 1, len(pathA) - 1)
                if b_id in updates:
                    idxB = min(idxB + 1, len(pathB) - 1)

                ticks.append(
                    SoloStep(
                        updates_pair_only=updates,
                        sample=True,
                        diamonds=diamonds_for_step,
                    )
                )
                trace[a_id].append(la)
                trace[b_id].append(lb)

            # PRE->IN
            updates = {}
            if la != meet:
                updates[a_id] = meet
            if lb != meet:
                updates[b_id] = meet
            if updates:
                la = updates.get(a_id, la)
                lb = updates.get(b_id, lb)
                in_idx = len(ticks)
                ticks.append(SoloStep(updates_pair_only=updates, sample=True, diamonds=[]))
                trace[a_id].append(la)
                trace[b_id].append(lb)

            # IN->PRE (wird live gesetzt)
            out_idx = len(ticks)
            ticks.append(SoloStep(updates_pair_only={}, sample=False, diamonds=[]))
            trace[a_id].append(preA)
            trace[b_id].append(preB)

            return SoloPlan(ticks, trace, used_diamonds, in_idx, out_idx)

        # ---------- Circle-spezifische Teile ----------
        def _canonical_loop_tuple(D: List[Coord]) -> Tuple[Coord, ...]:
            return tuple(sorted(D))

        def _rot_dir_loop(loop: List[Coord], u: Coord, v: Coord) -> int:
            i = loop.index(u)
            n = len(loop)
            if loop[(i + 1) % n] == v:
                return +1
            elif loop[(i - 1) % n] == v:
                return -1
            else:
                raise ValueError(f"{u}->{v} ist keine Nachbarschaft im Loop")

        def _circle_for_edge(u: Coord, v: Coord, target_len: int = 8) -> Optional[List[Coord]]:
            """
            Suche einen einfachen Zyklus der Länge `target_len` im SN-Subgraphen,
            der die Kante (u, v) enthält.
            Rückgabeformat: [u, v, ..., node] – zyklische Liste der Knoten.
            """
            if not (_is_sn(u) and _is_sn(v)):
                return None
            if not SN.has_edge(u, v):
                return None

            start = v
            max_nodes = target_len

            # DFS mit Längenbegrenzung
            stack: List[Tuple[Coord, List[Coord]]] = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) >= max_nodes:
                    continue

                for w in SN.neighbors(node):
                    if w == u:
                        full_path = path + [u]  # [v, ..., node, u]
                        if len(full_path) == max_nodes:
                            # Zyklus-Order: [u, v, ..., node]
                            loop = [u] + path
                            return loop
                        else:
                            continue

                    if w in path or w == u:
                        continue

                    if len(path) + 1 <= max_nodes:
                        stack.append((w, path + [w]))

            return None

        def _compute_pair_rotation_updates_for_loop(
            loop: List[Coord],
            direction: int,
            a_id: int,
            b_id: int,
            la: Coord,
            lb: Coord,
        ) -> Dict[int, Coord]:
            idx = {p: i for i, p in enumerate(loop)}
            out: Dict[int, Coord] = {}
            n = len(loop)

            if la in idx:
                out[a_id] = loop[(idx[la] + direction) % n]
            if lb in idx:
                out[b_id] = loop[(idx[lb] + direction) % n]
            return out

        # ---------- Circle-Solo-Plan für EIN Paar ----------
        def _plan_pair_solo_circle(a_id: int, b_id: int) -> Optional[SoloPlan]:
            la = current_pos[a_id]
            lb = current_pos[b_id]
            if not (_is_sn(la) and _is_sn(lb)):
                return None

            ticks: List[SoloStep] = []
            trace = {a_id: [la], b_id: [lb]}
            used_loops: Set[Tuple[Coord, ...]] = set()
            in_idx: Optional[int] = None
            out_idx: Optional[int] = None

            cands = DefaultRoutingPlanner._best_meeting_candidates(
                G, la, lb, reserved=set(), forbidden_nodes=set()
            )

            def _best_pre(meet: Coord, src: Coord) -> Optional[Tuple[Coord, List[Coord]]]:
                best = None
                for pre in _sn_neighbors_of_meet(meet):
                    p = _shortest_path_sn(src, pre)
                    if p is None:
                        continue
                    if best is None or len(p) < len(best[1]):
                        best = (pre, p)
                return best

            chosen = None
            for m in cands:
                if not _is_in(m):
                    continue
                pa = _best_pre(m, la)
                pb = _best_pre(m, lb)
                if pa and pb:
                    chosen = (m, pa, pb)
                    break
            if not chosen:
                return None

            meet, (preA, pathA), (preB, pathB) = chosen
            idxA = 0
            idxB = 0

            # bis PRE-A/B, mit Loops
            while la != preA or lb != preB:
                updates: Dict[int, Coord] = {}
                loops_for_step: List[Tuple[List[Coord], int]] = []

                if la != preA and idxA + 1 < len(pathA):
                    uA, vA = pathA[idxA], pathA[idxA + 1]
                    if _is_diag(uA, vA):
                        loopA = _circle_for_edge(uA, vA)
                        if loopA:
                            dirA = _rot_dir_loop(loopA, uA, vA)
                            updA = _compute_pair_rotation_updates_for_loop(
                                loopA, dirA, a_id, b_id, la, lb
                            )
                            updA[a_id] = vA
                            updates.update(updA)
                            loops_for_step.append((loopA, dirA))
                            used_loops.add(_canonical_loop_tuple(loopA))
                        else:
                            updates[a_id] = vA
                    else:
                        updates[a_id] = vA

                if lb != preB and idxB + 1 < len(pathB):
                    uB, vB = pathB[idxB], pathB[idxB + 1]
                    if _is_diag(uB, vB):
                        loopB = _circle_for_edge(uB, vB)
                        if loopB:
                            dirB = _rot_dir_loop(loopB, uB, vB)
                            updB = _compute_pair_rotation_updates_for_loop(
                                loopB, dirB, a_id, b_id, la, lb
                            )
                            updB[b_id] = vB
                            if not (loops_for_step and set(loops_for_step[0][0]).intersection(loopB)):
                                updates.update(updB)
                                loops_for_step.append((loopB, dirB))
                                used_loops.add(_canonical_loop_tuple(loopB))
                        else:
                            if b_id not in updates:
                                updates[b_id] = vB
                    else:
                        if b_id not in updates:
                            updates[b_id] = vB

                if not updates:
                    return None

                la = updates.get(a_id, la)
                lb = updates.get(b_id, lb)
                if a_id in updates:
                    idxA = min(idxA + 1, len(pathA) - 1)
                if b_id in updates:
                    idxB = min(idxB + 1, len(pathB) - 1)

                ticks.append(
                    SoloStep(
                        updates_pair_only=updates,
                        sample=True,
                        diamonds=loops_for_step,  # hier: Loops, aber gleiche Struktur
                    )
                )
                trace[a_id].append(la)
                trace[b_id].append(lb)

            # PRE->IN
            updates = {}
            if la != meet:
                updates[a_id] = meet
            if lb != meet:
                updates[b_id] = meet
            if updates:
                la = updates.get(a_id, la)
                lb = updates.get(b_id, lb)
                in_idx = len(ticks)
                ticks.append(SoloStep(updates_pair_only=updates, sample=True, diamonds=[]))
                trace[a_id].append(la)
                trace[b_id].append(lb)

            # IN->PRE (wird live gesetzt)
            out_idx = len(ticks)
            ticks.append(SoloStep(updates_pair_only={}, sample=False, diamonds=[]))
            trace[a_id].append(preA)
            trace[b_id].append(preB)

            return SoloPlan(ticks, trace, used_loops, in_idx, out_idx)

        # ---------- Kompatibilität der Solo-Pläne (Diamonds) ----------
        def _plans_compatible_distance(
            p1: SoloPlan,
            ab1: Tuple[int, int],
            p2: SoloPlan,
            ab2: Tuple[int, int],
        ) -> bool:
            a1, b1 = ab1
            a2, b2 = ab2
            L = max(p1.length, p2.length)

            def _pos(plan: SoloPlan, qid: int, i: int) -> Coord:
                if i < len(plan.pos_trace[qid]):
                    return plan.pos_trace[qid][i]
                return plan.pos_trace[qid][-1]

            for i in range(0, L + 1):
                p_a1 = _pos(p1, a1, i)
                p_b1 = _pos(p1, b1, i)
                p_a2 = _pos(p2, a2, i)
                p_b2 = _pos(p2, b2, i)
                for u in (p_a1, p_b1):
                    for v in (p_a2, p_b2):
                        if chebyshev(u, v) < 3:
                            return False
            return True

        def _plans_compatible_diamonds(p1: SoloPlan, p2: SoloPlan) -> bool:
            L = max(p1.length, p2.length)

            for i in range(L):
                if i < p1.length:
                    d1 = {
                        _canonical_diamond_tuple(D)
                        for (D, _dir) in p1.ticks[i].diamonds
                    }
                else:
                    d1 = set()

                if i < p2.length:
                    d2 = {
                        _canonical_diamond_tuple(D)
                        for (D, _dir) in p2.ticks[i].diamonds
                    }
                else:
                    d2 = set()

                if not d1.isdisjoint(d2):
                    return False

            return True

        # ---------- Gruppenbildung ----------
        def _group_parallel(plans: Dict[Tuple[int, int], SoloPlan]) -> List[List[Tuple[int, int]]]:
            remaining_pids = sorted(plans.keys())
            groups: List[List[Tuple[int, int]]] = []
            used: Set[Tuple[int, int]] = set()
            for pid in remaining_pids:
                if pid in used:
                    continue
                grp = [pid]
                used.add(pid)
                for qid in remaining_pids:
                    if qid in used:
                        continue
                    ok = all(
                        _plans_compatible_distance(plans[pid0], pid0, plans[qid], qid)
                        and _plans_compatible_diamonds(plans[pid0], plans[qid])
                        for pid0 in grp
                    )
                    if ok:
                        grp.append(qid)
                        used.add(qid)
                groups.append(grp)
            return groups

        # ---------- Laufzeit-Helfer: Rotationen expandieren ----------
        def _expand_runtime_rotations(
            base_updates: Dict[int, Coord],
            diamonds: List[Tuple[List[Coord], int]],
            exclude_qids: Set[int] = set(),
        ) -> Dict[int, Coord]:
            if not diamonds:
                return dict(base_updates)
            idx_cache: Dict[Tuple[Coord, ...], Dict[Coord, int]] = {}
            out = dict(base_updates)
            for D, direction in diamonds:
                key = tuple(D)
                if key not in idx_cache:
                    idx_cache[key] = {p: i for i, p in enumerate(D)}
                idx = idx_cache[key]
                n = len(D)
                for qid, pos in current_pos.items():
                    if qid in exclude_qids:
                        continue
                    if pos in idx:
                        out[qid] = D[(idx[pos] + direction) % n]
            return out

        def _foreign_qubits_on_any_diamond(
            diamonds: List[Tuple[List[Coord], int]],
            allowed: Set[int],
        ) -> bool:
            if not diamonds:
                return False
            nodes: Set[Coord] = set()
            for D, _ in diamonds:
                nodes |= set(D)
            for qid, pos in current_pos.items():
                if qid in allowed:
                    continue
                if pos in nodes:
                    return True
            return False

        # ---- Live-Map: PRE-SN zum Zeitpunkt eines *erfolgreichen* IN-Hops (pro Paar) ----
        live_pre_by_pair: Dict[Tuple[int, int], Dict[int, Coord]] = {}

        # ---- Interaktions-Reihenfolge: Index der Paare im Original-Array ----
        pair_order: Dict[Tuple[int, int], int] = {}
        for idx, (qa, qb) in enumerate(pairs):
            pid = (qa.id, qb.id)
            pair_order[pid] = idx

        def _is_ready_pair(pid: Tuple[int, int], remaining_pids: Set[Tuple[int, int]]) -> bool:
            idx = pair_order[pid]
            a, b = pid
            for other in remaining_pids:
                if other == pid:
                    continue
                j = pair_order[other]
                if j < idx:
                    x, y = other
                    if x == a or x == b or y == a or y == b:
                        return False
            return True

        # ===================== Hauptroutine =====================
        remaining: Set[Tuple[int, int]] = {(qa.id, qb.id) for qa, qb in pairs}

        while remaining:
            # 1) Nur "ready" Paare dürfen in diesem Durchlauf geplant werden
            ready_pids = [pid for pid in remaining if _is_ready_pair(pid, remaining)]

            # Failsafe
            if not ready_pids:
                ready_pids = list(remaining)

            # 2) Solo-Pläne für ready-Paare (Diamond)
            plans: Dict[Tuple[int, int], SoloPlan] = {}
            sequential_fallback: List[Tuple[int, int]] = []
            for pid in sorted(ready_pids, key=lambda p: pair_order[p]):
                a, b = pid
                plan = _plan_pair_solo(a, b)
                if plan is None:
                    sequential_fallback.append((a, b))
                else:
                    plans[pid] = plan

            if not plans and not sequential_fallback:
                _commit_tick({}, sample=True)
                break

            # 3) Parallelgruppen
            groups = _group_parallel(plans) if plans else []

            something_finished = False

            for grp in groups:
                group_qids: Set[int] = {x for ab in grp for x in ab}
                L = max(plans[pid].length for pid in grp) if grp else 0
                parallel_failed = False

                # --- Parallel-Replay MIT Retry bei Defekt (Diamond) ---
                step = 0
                while step < L:
                    updates_pair_only: Dict[int, Coord] = {}
                    sample_flags: List[bool] = []
                    step_diamonds: List[Tuple[List[Coord], int]] = []
                    conflict = False

                    # temporäre PREs für IN-Hops dieses Steps (werden nur bei Erfolg übernommen)
                    pending_live_pre: Dict[Tuple[int, int], Dict[int, Coord]] = {}

                    for pid in grp:
                        plan = plans[pid]
                        if step < plan.length:
                            s = plan.ticks[step]
                        else:
                            s = SoloStep({}, True, [])

                        # IN-Hop: PRE vor Commit zwischenspeichern
                        if plan.in_idx is not None and step == plan.in_idx:
                            a_id, b_id = pid
                            pending_live_pre[pid] = {
                                a_id: current_pos[a_id],
                                b_id: current_pos[b_id],
                            }

                        # OUT-Hop: Ziel aus bereits gemerkten PREs
                        if plan.out_idx is not None and step == plan.out_idx:
                            a_id, b_id = pid
                            pre_map = live_pre_by_pair.get(
                                pid,
                                {a_id: current_pos[a_id], b_id: current_pos[b_id]},
                            )
                            s = SoloStep(
                                {a_id: pre_map[a_id], b_id: pre_map[b_id]},
                                False,
                                [],
                            )

                        if set(updates_pair_only.keys()) & set(s.updates_pair_only.keys()):
                            conflict = True
                            break
                        updates_pair_only.update(s.updates_pair_only)
                        sample_flags.append(s.sample)
                        step_diamonds.extend(s.diamonds)

                    if conflict:
                        parallel_failed = True
                        break

                    if _foreign_qubits_on_any_diamond(step_diamonds, allowed=group_qids):
                        parallel_failed = True
                        break

                    updates = _expand_runtime_rotations(
                        updates_pair_only,
                        step_diamonds,
                    )
                    do_sample = False not in sample_flags

                    moved = _commit_tick(updates, sample=do_sample)
                    if moved:
                        for pid, pre in pending_live_pre.items():
                            live_pre_by_pair[pid] = pre
                        step += 1
                    else:
                        # Defekt -> RETRY gleicher Step
                        continue

                if parallel_failed:
                    # --- Sequentieller Fallback MIT Hybrid-Logik ---
                    for pid in grp:
                        plan = plans[pid]
                        a_id, b_id = pid
                        step = 0
                        while step < plan.length:
                            s = plan.ticks[step]
                            # IN-Hop -> PRE vor Commit zwischenspeichern
                            pending_pre = None
                            if plan.in_idx is not None and step == plan.in_idx:
                                pending_pre = {
                                    a_id: current_pos[a_id],
                                    b_id: current_pos[b_id],
                                }

                            # OUT-Hop -> live PRE
                            if plan.out_idx is not None and step == plan.out_idx:
                                pre_map = live_pre_by_pair.get(
                                    pid,
                                    {a_id: current_pos[a_id], b_id: current_pos[b_id]},
                                )
                                s = SoloStep(
                                    {a_id: pre_map[a_id], b_id: pre_map[b_id]},
                                    False,
                                    [],
                                )

                            updates = _expand_runtime_rotations(
                                s.updates_pair_only,
                                s.diamonds,
                            )

                            moved = _commit_tick(updates, sample=s.sample)

                            if moved:
                                if pending_pre is not None:
                                    live_pre_by_pair[pid] = pending_pre
                                step += 1
                            else:
                                # --- Hier: Blockade durch Defekt für dieses Paar? ---
                                if s.updates_pair_only:
                                    # Versuche Circle-Plan NUR für dieses Paar
                                    circle_plan = _plan_pair_solo_circle(a_id, b_id)

                                    if circle_plan is None:
                                        # Kein Circle-Plan -> einfach warten
                                        # (der obige _commit_tick war schon ein Wartetick)
                                        continue
                                    else:
                                        # Circle-Plan sequentiell ausführen
                                        step_c = 0
                                        while step_c < circle_plan.length:
                                            sc = circle_plan.ticks[step_c]
                                            pending_pre_c = None
                                            if circle_plan.in_idx is not None and step_c == circle_plan.in_idx:
                                                pending_pre_c = {
                                                    a_id: current_pos[a_id],
                                                    b_id: current_pos[b_id],
                                                }
                                            if circle_plan.out_idx is not None and step_c == circle_plan.out_idx:
                                                pre_map_c = live_pre_by_pair.get(
                                                    pid,
                                                    {a_id: current_pos[a_id], b_id: current_pos[b_id]},
                                                )
                                                sc = SoloStep(
                                                    {a_id: pre_map_c[a_id], b_id: pre_map_c[b_id]},
                                                    False,
                                                    [],
                                                )

                                            updates_c = _expand_runtime_rotations(
                                                sc.updates_pair_only,
                                                sc.diamonds,
                                            )
                                            moved_c = _commit_tick(updates_c, sample=sc.sample)
                                            if moved_c:
                                                if pending_pre_c is not None:
                                                    live_pre_by_pair[pid] = pending_pre_c
                                                step_c += 1
                                            else:
                                                # Defekt auch im Circle-Plan -> RETRY selben Step
                                                continue

                                        # Paar ist über Circle fertig -> Diamond-Plan als "abgeschlossen" markieren
                                        step = plan.length

                                else:
                                    # keine Updates -> normaler Wartetick
                                    continue

                # Gruppe abgeschlossen -> Paare entfernen und REPLAN starten
                for pid in grp:
                    if pid in remaining:
                        remaining.remove(pid)
                something_finished = True
                break  # replan

            # Nichts fertig geworden -> sequential_fallback (ein ready Paar), mit Hybrid-Logik
            if not something_finished:
                if sequential_fallback:
                    a, b = sequential_fallback[0]
                    plan = _plan_pair_solo(a, b)
                    if plan is None:
                        _commit_tick({}, sample=True)
                    else:
                        pid = (a, b)
                        step = 0
                        while step < plan.length:
                            s = plan.ticks[step]
                            pending_pre = None
                            if plan.in_idx is not None and step == plan.in_idx:
                                pending_pre = {
                                    a: current_pos[a],
                                    b: current_pos[b],
                                }
                            if plan.out_idx is not None and step == plan.out_idx:
                                pre_map = live_pre_by_pair.get(
                                    pid,
                                    {a: current_pos[a], b: current_pos[b]},
                                )
                                s = SoloStep(
                                    {a: pre_map[a], b: pre_map[b]},
                                    False,
                                    [],
                                )
                            updates = _expand_runtime_rotations(
                                s.updates_pair_only,
                                s.diamonds,
                            )
                            moved = _commit_tick(updates, sample=s.sample)
                            if moved:
                                if pending_pre is not None:
                                    live_pre_by_pair[pid] = pending_pre
                                step += 1
                            else:
                                # Blockade? -> Circle versuchen
                                if s.updates_pair_only:
                                    circle_plan = _plan_pair_solo_circle(a, b)
                                    if circle_plan is None:
                                        # kein Circle-Plan -> warten
                                        continue
                                    else:
                                        step_c = 0
                                        while step_c < circle_plan.length:
                                            sc = circle_plan.ticks[step_c]
                                            pending_pre_c = None
                                            if circle_plan.in_idx is not None and step_c == circle_plan.in_idx:
                                                pending_pre_c = {
                                                    a: current_pos[a],
                                                    b: current_pos[b],
                                                }
                                            if circle_plan.out_idx is not None and step_c == circle_plan.out_idx:
                                                pre_map_c = live_pre_by_pair.get(
                                                    pid,
                                                    {a: current_pos[a], b: current_pos[b]},
                                                )
                                                sc = SoloStep(
                                                    {a: pre_map_c[a], b: pre_map_c[b]},
                                                    False,
                                                    [],
                                                )

                                            updates_c = _expand_runtime_rotations(
                                                sc.updates_pair_only,
                                                sc.diamonds,
                                            )
                                            moved_c = _commit_tick(updates_c, sample=sc.sample)
                                            if moved_c:
                                                if pending_pre_c is not None:
                                                    live_pre_by_pair[pid] = pending_pre_c
                                                step_c += 1
                                            else:
                                                continue

                                        step = plan.length
                                else:
                                    continue

                    if (a, b) in remaining:
                        remaining.remove((a, b))
                else:
                    _commit_tick({}, sample=True)

        return timelines, edge_timebands
