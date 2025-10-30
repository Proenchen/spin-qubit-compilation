from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import random
import networkx as nx

from routing.common import AStar, Coord, MAX_TIME, Reservations, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner

P_SUCCESS = 0.98
P_REPAIR = 0.9


class SequentialDiamondRoutingPlanner:
    """
    Führt Paare nacheinander (ohne Layerbildung) aus – jetzt mit defekten Kanten:
    - Direkt vor jedem Tick werden Defekte gesampelt (p_success, p_repair),
      ABER beim Schritt vom IN (Meeting-Node) zurück zum PRE-IN wird NICHT gesampelt.
    - Wenn irgendein geplanter Schritt in diesem Tick eine defekte Kante benutzen würde,
      wird in diesem Tick kein Qubit bewegt (Wartetick).
    - Bewegungen eines Ticks werden atomisch betrachtet (alle geplanten Moves + Rotationen).
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

        # ---------- Defekt-Sampling & Tick-Handling ----------
        def _sample_edge_failures():
            """Analog zum Default: pro Tick Ausfall/Reparatur sampeln."""
            for u, v in G.edges():
                e = frozenset({u, v})
                if e in defective_edges:
                    if random.random() < p_repair:
                        defective_edges.discard(e)
                else:
                    if random.random() < (1.0 - p_success):
                        defective_edges.add(e)

        def _pending_updates_would_use_defect(pending_updates: Dict[int, Coord]) -> bool:
            """Prüft, ob irgendein geplanter Schritt (u->v) eine defekte Kante benutzen würde."""
            for qid, newp in pending_updates.items():
                u = current_pos[qid]
                v = newp
                if u != v:
                    if frozenset({u, v}) in defective_edges:
                        return True
            return False

        def _attempt_tick(pending_updates: Dict[int, Coord], *, sample: bool = True) -> bool:
            """
            Ein *Tick*:
              1) (Optional) Defekte jetzt sampeln – maßgeblich für diesen Tick,
              2) geplante Bewegungen atomisch anwenden, wenn keine defekte Kante betroffen,
              3) Zeit fortschreiben, Positionen & Defekte protokollieren.
            Rückgabe: True, wenn Moves angewandt wurden (sonst Wartetick).
            """
            nonlocal t
            # 1) frisch samplen (falls gewünscht)
            if sample:
                _sample_edge_failures()

            moved = False
            # 2) atomische Prüfung & Anwendung
            if pending_updates and not _pending_updates_would_use_defect(pending_updates):
                for qid, newp in pending_updates.items():
                    current_pos[qid] = newp
                moved = True

            # 3) t vor, Positionen & Zeitband loggen (immer loggen, auch bei sample=False)
            t += 1
            for qid in all_qids:
                last = timelines[qid][-1]
                cur = (current_pos[qid], t)
                if last != cur:
                    timelines[qid].append(cur)
            edge_timebands.append((t - 1, t, set(defective_edges)))
            return moved

        def append_all_positions():
            """Positionen für *aktuellen* t in die Timelines schreiben (ohne Tick)."""
            for qid in all_qids:
                last = timelines[qid][-1]
                cur = (current_pos[qid], t)
                if last != cur:
                    timelines[qid].append(cur)

        # ---------- Raute-/Rotations-Helfer ----------
        def _is_sn(n: Coord) -> bool:
            return G.nodes[n].get("type") == "SN"

        def _is_diagonal(u: Coord, v: Coord) -> bool:
            return abs(u[0] - v[0]) == 1 and abs(u[1] - v[1]) == 1

        def _diag_sn_neighbors(n: Coord) -> List[Coord]:
            out: List[Coord] = []
            for w in G.neighbors(n):
                if _is_sn(w) and _is_diagonal(n, w):
                    out.append(w)
            return out

        def _diamond_for_edge(u: Coord, v: Coord) -> Optional[List[Coord]]:
            """Liefert die vier SN-Knoten der Raute in Umlaufrichtung [u, v, x, w]."""
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
            """+1 = Uhrzeigersinn entlang [u->v->x->w->u], sonst -1."""
            i = diamond.index(u)
            return +1 if diamond[(i + 1) % 4] == v else -1

        def _compute_diamond_rotation_updates(
            diamond: List[Coord],
            direction: int,
            exclude: Set[int],
        ) -> Dict[int, Coord]:
            """
            Liefert neue Positionen der rotierenden Qubits auf der Raute
            (ohne sie anzuwenden). 'exclude' werden nicht bewegt.
            """
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

        # ---- Helfer: gemeinsame Raute & Ausweich-Schritt ----
        def _shared_diamond(n1: Coord, n2: Coord) -> Optional[List[Coord]]:
            if not (_is_sn(n1) and _is_sn(n2)):
                return None
            for v in _diag_sn_neighbors(n1):
                d = _diamond_for_edge(n1, v)
                if d is not None and n2 in d:
                    return d
            return None

        def _diamonds_at_node(n: Coord) -> List[List[Coord]]:
            """Alle Rauten (4er-Zyklen), an denen n beteiligt ist."""
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
            """
            Bewege 'node' entlang einer Diagonalkante einer Raute, die 'other' NICHT enthält.
            Liefert (v, diamond, direction) oder None.
            """
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

        # ---------- Pfad-Helfer ----------
        def _shortest_path_nodes(src: Coord, dst: Coord) -> List[Coord]:
            """Einfacher kürzester Pfad auf G (knotenweise)."""
            return nx.shortest_path(G, src, dst)

        def _entry_sn_from_path_nodes(path_nodes: List[Coord], meeting: Coord) -> Optional[Coord]:
            """PRE-IN ermitteln (Knoten direkt vor meeting in path_nodes)."""
            if meeting not in path_nodes:
                return None
            k = path_nodes.index(meeting)
            if k == 0:
                return None
            return path_nodes[k - 1]

        # ---------- Hauptschleife über Paare ----------
        for qa, qb in pairs:
            a, b = qa.id, qb.id
            pa = current_pos[a]
            pb = current_pos[b]

            cands = DefaultRoutingPlanner._best_meeting_candidates(
                G, pa, pb, reserved=set(), forbidden_nodes=set()
            )
            meet: Optional[Coord] = None
            pathA: Optional[List[Coord]] = None
            pathB: Optional[List[Coord]] = None

            for m in cands:
                try:
                    pA = _shortest_path_nodes(pa, m)
                    pB = _shortest_path_nodes(pb, m)
                    meet = m
                    pathA = pA
                    pathB = pB
                    break
                except nx.NetworkXNoPath:
                    continue

            if meet is None or pathA is None or pathB is None:
                _attempt_tick({})
                continue

            preA = _entry_sn_from_path_nodes(pathA, meet)
            preB = _entry_sn_from_path_nodes(pathB, meet)
            if preA is None or preB is None:
                _attempt_tick({})
                continue

            # --- Vorab-Ausweichschritt, wenn beide auf derselben Raute stehen (und noch nicht am PRE-IN) ---
            shared = _shared_diamond(pa, pb)
            if shared is not None and not (pa == preA and pb == preB):
                # deterministisch bevorzugt: b bewegen, sonst a
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
                    # Rotationen auf der Raute (andere Qubits), beide Paar-Qubits ausgeschlossen
                    rot = _compute_diamond_rotation_updates(dmd, direction, exclude={a, b})
                    pending_updates.update(rot)
                    pending_updates[mover_id] = v

                    # Atomischer Tick mit (hier) Sampling aktiv
                    _attempt_tick(pending_updates, sample=True)

                    # Pfade zum GLEICHEN Meeting neu berechnen
                    try:
                        pathA = _shortest_path_nodes(current_pos[a], meet)
                        pathB = _shortest_path_nodes(current_pos[b], meet)
                    except nx.NetworkXNoPath:
                        _attempt_tick({})
                        continue
                    preA = _entry_sn_from_path_nodes(pathA, meet)
                    preB = _entry_sn_from_path_nodes(pathB, meet)
                    if preA is None or preB is None:
                        _attempt_tick({})
                        continue

            # --- Pfade bis PRE-IN (inklusive Start, inklusive PRE-IN)
            pathA_to_pre = pathA[: pathA.index(meet)]
            pathB_to_pre = pathB[: pathB.index(meet)]

            # --- Schrittweise Ausführung bis beide PRE-IN erreicht ---
            idxA = 0
            idxB = 0

            while (current_pos[a] != preA) or (current_pos[b] != preB):
                pending_updates: Dict[int, Coord] = {}

                # Vorschlag für A in diesem Tick
                if current_pos[a] != preA and idxA + 1 < len(pathA_to_pre):
                    uA = pathA_to_pre[idxA]
                    vA = pathA_to_pre[idxA + 1]
                    diamondA = _diamond_for_edge(uA, vA)
                    if diamondA is not None:
                        dirA = _rotation_direction(diamondA, uA, vA)
                        rotA = _compute_diamond_rotation_updates(diamondA, dirA, exclude={a})
                        pending_updates.update(rotA)
                    pending_updates[a] = vA

                # Vorschlag für B in diesem Tick
                if current_pos[b] != preB and idxB + 1 < len(pathB_to_pre):
                    uB = pathB_to_pre[idxB]
                    vB = pathB_to_pre[idxB + 1]
                    diamondB = _diamond_for_edge(uB, vB)
                    if diamondB is not None:
                        dirB = _rotation_direction(diamondB, uB, vB)
                        rotB = _compute_diamond_rotation_updates(diamondB, dirB, exclude={b})
                        pending_updates.update(rotB)
                    pending_updates[b] = vB

                # Niemand geplant → reiner Wartetick (trotz Sampling)
                if not pending_updates:
                    _attempt_tick({})
                    continue

                moved = _attempt_tick(pending_updates, sample=True)
                if moved:
                    if a in pending_updates:
                        idxA += 1
                    if b in pending_updates:
                        idxB += 1
                # sonst Wartetick – idxA/idxB unverändert

            # --- kurzer IN-Hop synchron für beide, defekt-sicher und schrittweise ---
            # Erster Hop: PRE-IN -> IN (gleichzeitig)  (Sampling AN)
            while True:
                pend = {}
                if current_pos[a] != meet:
                    pend[a] = meet
                if current_pos[b] != meet:
                    pend[b] = meet

                if not pend:
                    break  # bereits beide am IN

                moved = _attempt_tick(pend, sample=True)
                if moved:
                    break  # geschafft; weiter zum Rück-Hop
                # sonst: Wartetick, erneut versuchen

            # Zweiter Hop: IN -> PRE-IN (gleichzeitig)  (Sampling AUS)
            while True:
                pend = {a: preA, b: preB}
                moved = _attempt_tick(pend, sample=False)  # << kein Sampling beim Rausgehen aus dem IN
                if moved:
                    break
                # sonst: Wartetick, erneut versuchen (weiterhin kein Sampling)

        return timelines, edge_timebands
