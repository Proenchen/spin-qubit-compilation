from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import random
import networkx as nx

from routing.common import Coord, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner

P_SUCCESS = 1.0
P_REPAIR = 0.9


class RotationRoutingPlanner:
    """
    Führt Paare nacheinander (ohne Layerbildung) aus – jetzt mit defekten Kanten, und mit
    strikter Trennung der Knotentypen:

    - Qubits bewegen sich ausschließlich auf SN-Knoten.
    - IN-Knoten dürfen nur für die Interaktion (Meeting-Node) betreten werden:
        synchroner Hop PRE-IN -> IN (Sampling AN), danach synchron IN -> PRE-IN (Sampling AUS).
    - Direkt vor jedem Tick werden Defekte gesampelt (p_success, p_repair),
      ABER beim Schritt vom IN (Meeting-Node) zurück zum PRE-IN wird NICHT gesampelt.
    - Wenn irgendein geplanter Schritt in diesem Tick eine defekte Kante benutzen würde,
      wird in diesem Tick kein Qubit bewegt (Wartetick).
    - Bewegungen eines Ticks werden atomisch betrachtet (alle geplanten Moves + Rotationen).

    WICHTIG:
    - Pfade zu PRE-IN werden ausschließlich im SN-Subgraphen geplant. Der PRE-IN ist
      ein SN-Nachbar des gewählten Meeting-IN.
    - Nebenbedingung: Pfade der Paar-Qubits vermeiden alle SN-Rauten-Kanten, in deren Raute
      sich das jeweils andere Paar-Qubit aktuell befindet (IN-Hops ausgenommen).
    - Meeting-Kandidaten werden nach minimaler Gesamtdistanz (A→best PRE + B→best PRE, mit Avoidance)
      sortiert geprüft.
    - Parallele Rotationen: Pro Tick können ZWEI Rotationen stattfinden, wenn sich die
      betroffenen Rauten nicht überlappen (keinen Knoten teilen). Überlappen sie, gilt
      sequentieller Fallback (deterministisch A zuerst).
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

        # Timelines starten bei t=0 mit der aktuellen Position
        timelines: Dict[int, List[TimedNode]] = {
            q.id: [(current_pos[q.id], 0)] for q in qubits
        }
        t = 0  # globale Zeit

        # Defekte Kanten (persistent) + Zeitbänder pro Tick
        defective_edges: Set[frozenset] = set()
        edge_timebands: List[Tuple[int, int, Set[frozenset]]] = []

        # ---------- Knotentyp-Checks ----------
        def _is_sn(n: Coord) -> bool:
            return G.nodes[n].get("type") == "SN"

        def _is_in(n: Coord) -> bool:
            return G.nodes[n].get("type") == "IN"

        # ---------- Defekt-Sampling & Tick-Handling ----------
        def _sample_edge_failures():
            for u, v in G.edges():
                e = frozenset({u, v})
                if e in defective_edges:
                    if random.random() < p_repair:
                        defective_edges.discard(e)
                else:
                    if random.random() < (1.0 - p_success):
                        defective_edges.add(e)

        def _pending_updates_would_use_defect(pending_updates: Dict[int, Coord]) -> bool:
            for qid, newp in pending_updates.items():
                u = current_pos[qid]
                v = newp
                if u != v and frozenset({u, v}) in defective_edges:
                    return True
            return False

        def _attempt_tick(pending_updates: Dict[int, Coord], *, sample: bool = True) -> bool:
            nonlocal t
            if sample:
                _sample_edge_failures()

            moved = False
            if pending_updates and not _pending_updates_would_use_defect(pending_updates):
                for qid, newp in pending_updates.items():
                    current_pos[qid] = newp
                moved = True

            t += 1
            for qid in all_qids:
                last = timelines[qid][-1]
                cur = (current_pos[qid], t)
                if last != cur:
                    timelines[qid].append(cur)
            edge_timebands.append((t - 1, t, set(defective_edges)))
            return moved

        # ---------- Raute-/Rotations-Helfer (nur auf SN-Knoten) ----------
        def _is_diagonal(u: Coord, v: Coord) -> bool:
            return abs(u[0] - v[0]) == 1 and abs(u[1] - v[1]) == 1

        def _diag_sn_neighbors(n: Coord) -> List[Coord]:
            out: List[Coord] = []
            if not _is_sn(n):
                return out
            for w in G.neighbors(n):
                if _is_sn(w) and _is_diagonal(n, w):
                    out.append(w)
            return out

        def _diamond_for_edge(u: Coord, v: Coord) -> Optional[List[Coord]]:
            if not (_is_sn(u) and _is_sn(v) and _is_diagonal(u, v)):
                return None
            Su = [w for w in _diag_sn_neighbors(u) if w != v]
            Sv = [x for x in _diag_sn_neighbors(v) if x != u]
            for w in Su:
                for x in Sv:
                    if _is_diagonal(w, x):
                        return [u, v, x, w]
            return None

        def _rotation_direction(diamond: List[Coord], u: Coord, v: Coord) -> int:
            i = diamond.index(u)
            return +1 if diamond[(i + 1) % 4] == v else -1

        def _compute_diamond_rotation_updates(
            diamond: List[Coord],
            direction: int,
            exclude: Set[int],
        ) -> Dict[int, Coord]:
            idx_of = {p: i for i, p in enumerate(diamond)}
            updates: Dict[int, Coord] = {}
            for qid, pos in current_pos.items():
                if qid in exclude:
                    continue
                if pos in idx_of:
                    i = idx_of[pos]
                    ni = (i + direction) % 4
                    updates[qid] = diamond[ni]
            return updates

        # ---- Helfer: gemeinsame Raute & Ausweich-Schritt (nur SN) ----
        def _shared_diamond(n1: Coord, n2: Coord) -> Optional[List[Coord]]:
            if not (_is_sn(n1) and _is_sn(n2)):
                return None
            for v in _diag_sn_neighbors(n1):
                d = _diamond_for_edge(n1, v)
                if d is not None and n2 in d:
                    return d
            return None

        def _diamonds_at_node(n: Coord) -> List[List[Coord]]:
            out: List[List[Coord]] = []
            if not _is_sn(n):
                return out
            seen: Set[Tuple[Coord, Coord, Coord, Coord]] = set()
            for v in _diag_sn_neighbors(n):
                d = _diamond_for_edge(n, v)
                if d is None:
                    continue
                canon = tuple(sorted(d))
                if canon not in seen:
                    seen.add(canon)
                    out.append(d)
            return out

        def _escape_step_along_other_diamond(
            node: Coord, other: Coord
        ) -> Optional[Tuple[Coord, List[Coord], int]]:
            Ds = _diamonds_at_node(node)
            Ds = [D for D in Ds if other not in D]
            if not Ds:
                return None
            D = Ds[0]
            i = D.index(node)
            v_forward = D[(i + 1) % 4]
            v_backward = D[(i - 1) % 4]
            for v in (v_forward, v_backward):
                if _is_diagonal(node, v):
                    direction = _rotation_direction(D, node, v)
                    return (v, D, direction)
            return None

        # ---------- Pfad-Helfer: ausschließlich SN-Subgraph ----------
        sn_nodes = [n for n in G.nodes() if _is_sn(n)]
        SN = G.subgraph(sn_nodes).copy()

        def _sn_neighbors_of_meet(meeting: Coord) -> List[Coord]:
            if not _is_in(meeting):
                return []
            return [w for w in G.neighbors(meeting) if _is_sn(w)]

        # ---------- Avoidance (verbotene Kanten wg. Raute des anderen Paar-Qubits) ----------
        def _diamond_edges(diamond: List[Coord]) -> Set[frozenset]:
            return {
                frozenset({diamond[i], diamond[(i + 1) % 4]})
                for i in range(4)
            }

        def _forbidden_edges_due_to(other_pos: Coord) -> Set[frozenset]:
            forb: Set[frozenset] = set()
            for D in _diamonds_at_node(other_pos):
                forb |= _diamond_edges(D)
            return forb

        def _shortest_path_sn_avoiding(src: Coord, dst: Coord, forbidden: Set[frozenset]) -> Optional[List[Coord]]:
            H = SN.copy()
            to_remove = []
            for u, v in H.edges():
                if frozenset({u, v}) in forbidden:
                    to_remove.append((u, v))
            H.remove_edges_from(to_remove)
            try:
                return nx.shortest_path(H, src, dst)
            except nx.NetworkXNoPath:
                return None

        # ---------- Distanzschätzer für Meeting-Sortierung ----------
        def _min_sn_distance_to_any_pre(meeting: Coord, src: Coord, forbidden: Set[frozenset]) -> Optional[int]:
            best: Optional[int] = None
            for pre in _sn_neighbors_of_meet(meeting):
                path = _shortest_path_sn_avoiding(src, pre, forbidden)
                if path is None:
                    continue
                d = max(0, len(path) - 1)
                if best is None or d < best:
                    best = d
            return best

        # ---------- Overlap-Check für parallele Rotation ----------
        def _diamonds_overlap(d1: Optional[List[Coord]], d2: Optional[List[Coord]]) -> bool:
            """True, wenn sich zwei Rauten wenigstens einen Knoten teilen."""
            if d1 is None or d2 is None:
                return False
            return not set(d1).isdisjoint(d2)

        # ---------- Hauptschleife über Paare ----------
        for qa, qb in pairs:
            a, b = qa.id, qb.id
            pa = current_pos[a]
            pb = current_pos[b]

            if not _is_sn(pa) or not _is_sn(pb):
                _attempt_tick({})
                continue

            cands = DefaultRoutingPlanner._best_meeting_candidates(
                G, pa, pb, reserved=set(), forbidden_nodes=set()
            )

            meet: Optional[Coord] = None
            preA: Optional[Coord] = None
            preB: Optional[Coord] = None
            pathA_to_pre: Optional[List[Coord]] = None
            pathB_to_pre: Optional[List[Coord]] = None

            forb_for_A = _forbidden_edges_due_to(pb)
            forb_for_B = _forbidden_edges_due_to(pa)

            # Sortiere Meetings nach minimaler Gesamtdistanz (Avoidance)
            scored: List[Tuple[float, Coord]] = []
            for m in cands:
                if not _is_in(m):
                    continue
                da = _min_sn_distance_to_any_pre(m, pa, forb_for_A)
                db = _min_sn_distance_to_any_pre(m, pb, forb_for_B)
                scored.append(((float("inf") if da is None or db is None else da + db), m))
            cands_sorted = [m for _, m in sorted(scored, key=lambda x: x[0])]

            # Wähle erstes Meeting mit gültigen Avoidance-Pfaden
            for m in cands_sorted:
                if not _is_in(m):
                    continue

                bestA: Optional[Tuple[Coord, List[Coord]]] = None
                for pre in _sn_neighbors_of_meet(m):
                    p = _shortest_path_sn_avoiding(pa, pre, forb_for_A)
                    if p is None:
                        continue
                    if bestA is None or len(p) < len(bestA[1]):
                        bestA = (pre, p)
                if bestA is None:
                    continue

                bestB: Optional[Tuple[Coord, List[Coord]]] = None
                for pre in _sn_neighbors_of_meet(m):
                    p = _shortest_path_sn_avoiding(pb, pre, forb_for_B)
                    if p is None:
                        continue
                    if bestB is None or len(p) < len(bestB[1]):
                        bestB = (pre, p)
                if bestB is None:
                    continue

                meet = m
                preA, pathA_to_pre = bestA
                preB, pathB_to_pre = bestB
                break

            if meet is None or preA is None or preB is None or pathA_to_pre is None or pathB_to_pre is None:
                _attempt_tick({})
                continue

            # --- Vorab-Ausweichschritt, falls beide auf derselben Raute stehen ---
            shared = _shared_diamond(pa, pb)
            if shared is not None and not (pa == preA and pb == preB):
                esc_b = _escape_step_along_other_diamond(pb, pa)
                esc_a = _escape_step_along_other_diamond(pa, pb)

                move_choice = None
                mover_id = None
                if esc_b is not None:
                    move_choice = esc_b
                    mover_id = b
                elif esc_a is not None:
                    move_choice = esc_a
                    mover_id = a

                if move_choice is not None:
                    v, dmd, direction = move_choice
                    pending_updates: Dict[int, Coord] = {}
                    rot = _compute_diamond_rotation_updates(dmd, direction, exclude={a, b})
                    pending_updates.update(rot)
                    pending_updates[mover_id] = v
                    _attempt_tick(pending_updates, sample=True)

                    forb_for_A = _forbidden_edges_due_to(current_pos[b])
                    forb_for_B = _forbidden_edges_due_to(current_pos[a])
                    pathA_to_pre = _shortest_path_sn_avoiding(current_pos[a], preA, forb_for_A)
                    pathB_to_pre = _shortest_path_sn_avoiding(current_pos[b], preB, forb_for_B)
                    if pathA_to_pre is None or pathB_to_pre is None:
                        _attempt_tick({})
                        continue

            # --- Schrittweise Ausführung bis beide PRE-IN erreicht ---
            def _ensure_path_contains_pos_or_replan(
                who_id: int,
                cur_path: List[Coord],
                dst_pre: Coord,
                forbidden: Set[frozenset],
            ) -> Tuple[List[Coord], int, bool]:
                pos = current_pos[who_id]
                if pos in cur_path:
                    return cur_path, cur_path.index(pos), False
                newp = _shortest_path_sn_avoiding(pos, dst_pre, forbidden)
                if newp is None:
                    return cur_path, 0, True
                return newp, 0, True

            forb_for_A = _forbidden_edges_due_to(current_pos[b])
            forb_for_B = _forbidden_edges_due_to(current_pos[a])
            pathA_to_pre, idxA, replA = _ensure_path_contains_pos_or_replan(a, pathA_to_pre, preA, forb_for_A)
            pathB_to_pre, idxB, replB = _ensure_path_contains_pos_or_replan(b, pathB_to_pre, preB, forb_for_B)
            if (replA and current_pos[a] not in pathA_to_pre) or (replB and current_pos[b] not in pathB_to_pre):
                _attempt_tick({})
                continue

            idxA = max(0, min(idxA, len(pathA_to_pre) - 1))
            idxB = max(0, min(idxB, len(pathB_to_pre) - 1))

            while (current_pos[a] != preA) or (current_pos[b] != preB):
                pending_updates: Dict[int, Coord] = {}

                # --- Kandidat A vorbereiten ---
                rotA: Dict[int, Coord] = {}
                vA: Optional[Coord] = None
                diamondA: Optional[List[Coord]] = None
                if current_pos[a] != preA and idxA + 1 < len(pathA_to_pre):
                    uA = pathA_to_pre[idxA]
                    vA = pathA_to_pre[idxA + 1]
                    if _is_sn(uA) and _is_sn(vA):
                        diamondA = _diamond_for_edge(uA, vA)
                        if diamondA is not None:
                            dirA = _rotation_direction(diamondA, uA, vA)
                            rotA = _compute_diamond_rotation_updates(diamondA, dirA, exclude={a})
                # --- Kandidat B vorbereiten ---
                rotB: Dict[int, Coord] = {}
                vB: Optional[Coord] = None
                diamondB: Optional[List[Coord]] = None
                if current_pos[b] != preB and idxB + 1 < len(pathB_to_pre):
                    uB = pathB_to_pre[idxB]
                    vB = pathB_to_pre[idxB + 1]
                    if _is_sn(uB) and _is_sn(vB):
                        diamondB = _diamond_for_edge(uB, vB)
                        if diamondB is not None:
                            dirB = _rotation_direction(diamondB, uB, vB)
                            rotB = _compute_diamond_rotation_updates(diamondB, dirB, exclude={b})

                need_rotA = bool(rotA)
                need_rotB = bool(rotB)

                if need_rotA and need_rotB:
                    # Parallele Rotationen nur, wenn Rauten disjunkt sind; sonst sequentiell (A zuerst).
                    if not _diamonds_overlap(diamondA, diamondB):
                        pending_updates.update(rotA)
                        pending_updates.update(rotB)
                        if vA is not None:
                            pending_updates[a] = vA
                        if vB is not None:
                            pending_updates[b] = vB
                    else:
                        pending_updates.update(rotA)
                        if vA is not None:
                            pending_updates[a] = vA
                else:
                    # Nur eine Seite rotiert (oder keine). Dann dürfen beide (sofern vorhanden) ziehen.
                    if need_rotA:
                        pending_updates.update(rotA)
                    if need_rotB:
                        pending_updates.update(rotB)
                    if vA is not None:
                        pending_updates[a] = vA
                    if vB is not None:
                        pending_updates[b] = vB

                if not pending_updates:
                    _attempt_tick({})
                    # Replan mit aktueller Avoidance
                    forb_for_A = _forbidden_edges_due_to(current_pos[b])
                    forb_for_B = _forbidden_edges_due_to(current_pos[a])
                    if current_pos[a] != preA:
                        newp = _shortest_path_sn_avoiding(current_pos[a], preA, forb_for_A)
                        if newp is not None:
                            pathA_to_pre, idxA = newp, 0
                    if current_pos[b] != preB:
                        newp = _shortest_path_sn_avoiding(current_pos[b], preB, forb_for_B)
                        if newp is not None:
                            pathB_to_pre, idxB = newp, 0
                    continue

                moved = _attempt_tick(pending_updates, sample=True)
                if moved:
                    if a in pending_updates:
                        idxA += 1
                    if b in pending_updates:
                        idxB += 1
                else:
                    # Wartetick – ggf. neu planen
                    forb_for_A = _forbidden_edges_due_to(current_pos[b])
                    forb_for_B = _forbidden_edges_due_to(current_pos[a])
                    if current_pos[a] != preA:
                        newp = _shortest_path_sn_avoiding(current_pos[a], preA, forb_for_A)
                        if newp is not None:
                            pathA_to_pre, idxA = newp, 0
                    if current_pos[b] != preB:
                        newp = _shortest_path_sn_avoiding(current_pos[b], preB, forb_for_B)
                        if newp is not None:
                            pathB_to_pre, idxB = newp, 0

            # --- IN-Hop synchron ---
            while True:
                pend = {}
                if current_pos[a] != meet:
                    pend[a] = meet
                if current_pos[b] != meet:
                    pend[b] = meet
                if not pend:
                    break
                assert all(_is_in(dest) for dest in pend.values())
                if _attempt_tick(pend, sample=True):
                    break

            while True:
                pend = {a: preA, b: preB}
                assert _is_sn(preA) and _is_sn(preB)
                if _attempt_tick(pend, sample=False):
                    break

        return timelines, edge_timebands
