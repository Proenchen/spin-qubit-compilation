from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import random
import networkx as nx

from routing.common import Coord, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner

P_SUCCESS = 0.998
P_REPAIR = 0.25

# ------------------------- Utils -------------------------
def chebyshev(p: Coord, q: Coord) -> int:
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def _edgeset(u: Coord, v: Coord) -> frozenset:
    return frozenset({u, v})

# =================================================================
#                           ROUTER
# =================================================================
class RotationRoutingPlanner:
    """
    Layer-basierter Router mit vorausgeplanter Parallelisierung und echten Rotationen:

    - Solo-Plan pro Paar (inkl. Diamond-Rotationen).
    - Gruppenbildung mit:
        * Chebyshev-Distanz zwischen Paaren über gesamte Plan-Dauer >= 3,
        * disjunkten (knotenfreien) Diamond-Mengen zwischen Plänen.
    - Parallel-Replay pro Gruppe; bei Fremd-Blockaden (fremde Qubits auf Diamond) sequentieller Fallback.
    - WICHTIG:
        * Beim IN-Hop wird der reale PRE-SN des Qubits nur dann gespeichert, wenn der IN-Hop erfolgreich war.
        * OUT-Hop springt exakt zu diesem gespeicherten PRE zurück.
    - NEU:
        * Nach jeder **erfolgreichen Interaktion** eines Paars (OUT-Hop ausgeführt)
          werden Solo-Pläne & Parallel-Gruppen für die verbleibenden Paare im Layer **neu berechnet**.
    - NEU (Defekt-Handling):
        * Wenn ein Step wegen defekter Kante scheitert (moved==False), wird **dieselbe Bewegung im nächsten Tick wiederholt**
          (statt den Step zu überspringen oder sofort in einen anderen Modus zu wechseln).
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

        # ---------- Rauten/Rotation ----------
        def _is_diag(u: Coord, v: Coord) -> bool:
            return abs(u[0] - v[0]) == 1 and abs(u[1] - v[1]) == 1

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

        def _rot_dir(diamond: List[Coord], u: Coord, v: Coord) -> int:
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
            __slots__ = ("updates_pair_only", "sample", "diamonds")  # diamonds: List[(diamond, dir)]
            def __init__(self, updates_pair_only: Dict[int, Coord], sample: bool, diamonds: List[Tuple[List[Coord], int]]):
                self.updates_pair_only = updates_pair_only
                self.sample = sample
                self.diamonds = diamonds  # jede (diamond, dir) wird später zur Laufzeit expandiert

        class SoloPlan:
            """
            - ticks: Liste SoloStep
            - pos_trace: qid -> [pos_t0, pos_t1, ..., pos_tN] (nur für die zwei Qubits des Paars)
            - used_diamonds: Menge kanonischer Diamond-Knoten-Tupel (für Gruppen-Disjunktheit)
            - in_idx: Index des IN-Hops im Tick-Array (oder None)
            - out_idx: Index des OUT-Hops (oder None)
            """
            def __init__(
                self,
                ticks: List[SoloStep],
                pos_trace: Dict[int, List[Coord]],
                used_diamonds: Set[Tuple[Coord, Coord, Coord, Coord]],
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

        def _canonical_diamond_tuple(D: List[Coord]) -> Tuple[Coord, Coord, Coord, Coord]:
            return tuple(sorted(D))

        def _compute_pair_rotation_updates_for_diamond(
            diamond: List[Coord], direction: int,
            a_id: int, b_id: int, la: Coord, lb: Coord
        ) -> Dict[int, Coord]:
            idx = {p: i for i, p in enumerate(diamond)}
            out: Dict[int, Coord] = {}
            if la in idx:
                out[a_id] = diamond[(idx[la] + direction) % 4]
            if lb in idx:
                out[b_id] = diamond[(idx[lb] + direction) % 4]
            return out

        def _plan_pair_solo(a_id: int, b_id: int) -> Optional[SoloPlan]:
            la = current_pos[a_id]
            lb = current_pos[b_id]
            if not (_is_sn(la) and _is_sn(lb)):
                return None

            ticks: List[SoloStep] = []
            trace = {a_id: [la], b_id: [lb]}
            used_diamonds: Set[Tuple[Coord, Coord, Coord, Coord]] = set()
            in_idx: Optional[int] = None
            out_idx: Optional[int] = None

            cands = DefaultRoutingPlanner._best_meeting_candidates(G, la, lb, reserved=set(), forbidden_nodes=set())

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
                            dirA = _rot_dir(dA, uA, vA)
                            updA = _compute_pair_rotation_updates_for_diamond(dA, dirA, a_id, b_id, la, lb)
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
                            dirB = _rot_dir(dB, uB, vB)
                            updB = _compute_pair_rotation_updates_for_diamond(dB, dirB, a_id, b_id, la, lb)
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

                ticks.append(SoloStep(updates_pair_only=updates, sample=True, diamonds=diamonds_for_step))
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

        # ---------- Layerbildung (greedy, Startabstand >=3) ----------
        def _initial_layers() -> List[List[Tuple[int, int]]]:
            ids = [(qa.id, qb.id) for (qa, qb) in pairs]
            def _pair_far_enough(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
                P = [current_pos[p1[0]], current_pos[p1[1]]]
                Q = [current_pos[p2[0]], current_pos[p2[1]]]
                return all(chebyshev(u, v) >= 3 for u in P for v in Q)

            layers: List[List[Tuple[int, int]]] = []
            for pid in ids:
                placed = False
                for L in layers:
                    if all(_pair_far_enough(pid, x) for x in L):
                        L.append(pid)
                        placed = True
                        break
                if not placed:
                    layers.append([pid])
            return layers

        # ---------- Kompatibilität der Solo-Pläne ----------
        def _plans_compatible_distance(p1: SoloPlan, ab1: Tuple[int, int], p2: SoloPlan, ab2: Tuple[int, int]) -> bool:
            a1, b1 = ab1
            a2, b2 = ab2
            L = max(p1.length, p2.length)
            def _pos(plan: SoloPlan, qid: int, i: int) -> Coord:
                if i < len(plan.pos_trace[qid]):
                    return plan.pos_trace[qid][i]
                return plan.pos_trace[qid][-1]
            for i in range(0, L + 1):
                p_a1 = _pos(p1, a1, i); p_b1 = _pos(p1, b1, i)
                p_a2 = _pos(p2, a2, i); p_b2 = _pos(p2, b2, i)
                for u in (p_a1, p_b1):
                    for v in (p_a2, p_b2):
                        if chebyshev(u, v) < 3:
                            return False
            return True

        def _plans_compatible_diamonds(p1: SoloPlan, p2: SoloPlan) -> bool:
            return p1.used_diamonds.isdisjoint(p2.used_diamonds)

        # ---------- Gruppenbildung ----------
        def _group_parallel(plans: Dict[Tuple[int, int], SoloPlan]) -> List[List[Tuple[int, int]]]:
            remaining = sorted(plans.keys())
            groups: List[List[Tuple[int, int]]] = []
            used: Set[Tuple[int, int]] = set()
            for pid in remaining:
                if pid in used:
                    continue
                grp = [pid]
                used.add(pid)
                for qid in remaining:
                    if qid in used:
                        continue
                    ok = all(
                        _plans_compatible_distance(plans[pid0], pid0, plans[qid], qid) and
                        _plans_compatible_diamonds(plans[pid0], plans[qid])
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
            exclude_qids: Set[int] = set()
        ) -> Dict[int, Coord]:
            if not diamonds:
                return dict(base_updates)
            idx_cache = {}
            out = dict(base_updates)
            for D, direction in diamonds:
                key = tuple(D)
                if key not in idx_cache:
                    idx_cache[key] = {p: i for i, p in enumerate(D)}
                idx = idx_cache[key]
                for qid, pos in current_pos.items():
                    if qid in exclude_qids:
                        continue
                    if pos in idx:
                        out[qid] = D[(idx[pos] + direction) % 4]
            return out

        def _foreign_qubits_on_any_diamond(
            diamonds: List[Tuple[List[Coord], int]], allowed: Set[int]
        ) -> bool:
            if not diamonds:
                return False
            nodes = set()
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

        # ===================== Hauptroutine über Layers =====================
        for layer in _initial_layers():
            remaining: Set[Tuple[int, int]] = {(qa.id, qb.id) for qa, qb in pairs if (qa.id, qb.id) in set(layer)}

            while remaining:
                # 1) Solo-Pläne für verbleibende Paare
                plans: Dict[Tuple[int, int], SoloPlan] = {}
                sequential_fallback: List[Tuple[int, int]] = []
                for pid in sorted(remaining):
                    a, b = pid
                    plan = _plan_pair_solo(a, b)
                    if plan is None:
                        sequential_fallback.append((a, b))
                    else:
                        plans[pid] = plan

                if not plans and not sequential_fallback:
                    _commit_tick({}, sample=True)
                    break

                # 2) Parallelgruppen
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

                            # Fremdblockaden prüfen (bevor wir committen)
                            # (Wenn dauerhaft blockiert, gehen wir später in Fallback/Replan.)
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
                                s = SoloStep({a_id: pre_map[a_id], b_id: pre_map[b_id]}, False, [])

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

                        updates = _expand_runtime_rotations(updates_pair_only, step_diamonds)
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
                                    pending_pre = {a_id: current_pos[a_id], b_id: current_pos[b_id]}

                                # OUT-Hop -> live PRE nutzen
                                if plan.out_idx is not None and step == plan.out_idx:
                                    pre_map = live_pre_by_pair.get(
                                        pid,
                                        {a_id: current_pos[a_id], b_id: current_pos[b_id]},
                                    )
                                    s = SoloStep({a_id: pre_map[a_id], b_id: pre_map[b_id]}, False, [])

                                updates = _expand_runtime_rotations(s.updates_pair_only, s.diamonds)
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

                # Nichts fertig geworden -> sequential_fallback (ein Paar), mit Retry
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
                                    pending_pre = {a: current_pos[a], b: current_pos[b]}
                                if plan.out_idx is not None and step == plan.out_idx:
                                    pre_map = live_pre_by_pair.get(pid, {a: current_pos[a], b: current_pos[b]})
                                    s = SoloStep({a: pre_map[a], b: pre_map[b]}, False, [])
                                updates = _expand_runtime_rotations(s.updates_pair_only, s.diamonds)
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
