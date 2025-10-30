from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
from copy import deepcopy
import random
import networkx as nx

from routing.common import AStar, Coord, MAX_TIME, Reservations, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner

class SequentialDiamondRoutingPlanner:
    """
    Führt Paare nacheinander (ohne Layerbildung) aus.
    - Meeting-IN Auswahl wie im Default (minimale Gesamtdistanz, Tie-Breaks etc.)
    - Kürzeste Pfade bis PRE-IN je PaarQubit.
    - Beim Schritt eines PaarQubits über eine Raute-Diagonalkante (zwischen SN 1/3/4/6 desselben Tiles)
      rotieren alle anderen Qubits, die auf dieser Raute stehen, im Uhr- oder Gegenuhrzeigersinn
      entsprechend der Bewegungsrichtung des PaarQubits (Kollisionen werden so vermieden).
    - Danach kurzer IN-Hop (PRE-IN -> IN -> PRE-IN) wie im Default.
    - Keine defekten Kanten, kein Sampling.

    Erweiterung:
    - Befinden sich beide Paar-Qubits zu Beginn der Paarbearbeitung auf *derselben* Raute,
      wird zunächst *eines* der beiden (deterministisch: b) per Diagonal-Schritt auf eine
      Raute bewegt, die den anderen Paarpartner nicht enthält. Erst danach erfolgt die
      Meeting-Suche. Während dieses Vorab-Schritts werden Rotationen anderer Qubits auf
      der jeweiligen Raute berücksichtigt, *aber* beide Paar-Qubits sind von dieser
      Rotationskaskade ausgenommen (keine gegenseitige Beeinflussung).
    """

    @staticmethod
    def route(
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
    ):
        # --- Zustand ---
        current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}
        all_qids: Set[int] = {q.id for q in qubits}

        # Timelines: jede Liste startet bei t=0 mit der aktuellen Position
        timelines: Dict[int, List[TimedNode]] = {
            q.id: [(current_pos[q.id], 0)] for q in qubits
        }
        t = 0  # globale Zeit

        def append_all_positions():
            """Alle aktuellen Positionen auf Zeit t in die Timelines schreiben (ohne Dubletten)."""
            for qid in all_qids:
                last = timelines[qid][-1]
                cur = (current_pos[qid], t)
                if last != cur:
                    timelines[qid].append(cur)

        # ---------- Raute-Helfer ----------
        def _is_sn(n: Coord) -> bool:
            return G.nodes[n].get("type") == "SN"

        def _is_diagonal(u: Coord, v: Coord) -> bool:
            return abs(u[0]-v[0]) == 1 and abs(u[1]-v[1]) == 1

        def _diag_sn_neighbors(n: Coord) -> List[Coord]:
            """Alle SN-Nachbarn in diagonaler Richtung (|dx|=|dy|=1)."""
            out: List[Coord] = []
            for w in G.neighbors(n):
                if _is_sn(w) and _is_diagonal(n, w):
                    out.append(w)
            return out

        def _diamond_for_edge(u: Coord, v: Coord) -> Optional[List[Coord]]:
            """
            Liefert die vier SN-Knoten der Raute *in Umlaufrichtung* [u, v, x, w],
            so dass (u->v), (v->x), (x->w), (w->u) jeweils diagonale Kanten sind.
            """
            if not (_is_sn(u) and _is_sn(v) and _is_diagonal(u, v)):
                return None

            Su = [w for w in _diag_sn_neighbors(u) if w != v]
            Sv = [x for x in _diag_sn_neighbors(v) if x != u]
            for w in Su:
                for x in Sv:
                    if _is_diagonal(w, x):
                        return [u, v, x, w]  # zyklische Reihenfolge um die Raute
            return None

        def _rotation_direction(diamond: List[Coord], u: Coord, v: Coord) -> int:
            """
            +1 = im Uhrzeigersinn entlang [u->v->x->w->u]
            (Wir geben +1, wenn v der Nachfolger von u in 'diamond' ist, sonst -1.)
            """
            i = diamond.index(u)
            return +1 if diamond[(i + 1) % 4] == v else -1

        def _compute_diamond_rotation_updates(
            diamond: List[Coord],
            direction: int,
            exclude: Set[int],
        ) -> Dict[int, Coord]:
            """
            Liefert nur die *neuen* Positionen der rotierenden Qubits auf der Raute
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

        # ---- Neue Helfer: gemeinsame Raute erkennen & Ausweich-Schritt wählen ----
        def _shared_diamond(n1: Coord, n2: Coord) -> Optional[List[Coord]]:
            """Falls es eine Raute gibt, die sowohl n1 als auch n2 enthält, liefere deren 4er-Zyklus."""
            if not (_is_sn(n1) and _is_sn(n2)):
                return None
            for v in _diag_sn_neighbors(n1):
                d = _diamond_for_edge(n1, v)
                if d is not None and n2 in d:
                    return d
            return None
        
        def _diamonds_at_node(n: Coord) -> List[List[Coord]]:
            """Alle Rauten (4er-Zyklen) zurückgeben, an denen n beteiligt ist."""
            out: List[List[Coord]] = []
            if not _is_sn(n):
                return out
            seen: Set[Tuple[Coord, Coord, Coord, Coord]] = set()
            for v in _diag_sn_neighbors(n):
                d = _diamond_for_edge(n, v)
                if d is None:
                    continue
                # kanonische Repräsentation für Deduplikation
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
            # nur Rauten, die den Partner NICHT enthalten
            Ds = [D for D in Ds if other not in D]
            if not Ds:
                return None

            # deterministische Auswahl: zuerst gefundene "andere" Raute
            D = Ds[0]
            i = D.index(node)
            # zwei mögliche Kanten entlang dieser Raute: successor und predecessor
            v_forward = D[(i + 1) % 4]
            v_backward = D[(i - 1) % 4]

            # Wir wählen bevorzugt v_forward (deterministisch). Prüfen, dass es wirklich diagonal ist.
            for v in (v_forward, v_backward):
                if _is_diagonal(node, v):
                    direction = _rotation_direction(D, node, v)
                    return (v, D, direction)
            return None


        # ---------- Pfad-Helfer ----------
        def _shortest_path_nodes(src: Coord, dst: Coord) -> List[Coord]:
            """Einfachster kürzester Pfad auf G (knotenweise)."""
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
                t += 1
                append_all_positions()
                continue

            preA = _entry_sn_from_path_nodes(pathA, meet)
            preB = _entry_sn_from_path_nodes(pathB, meet)
            if preA is None or preB is None:
                t += 1
                append_all_positions()
                continue

            # --- NEU: Wegrotieren NUR wenn: gleiche Raute UND beide nicht auf ihrem PRE-IN ---
            shared = _shared_diamond(pa, pb)
            if shared is not None and not (pa == preA and pb == preB):
                # deterministisch bevorzugt: b bewegen, sonst a
                esc_b = _escape_step_along_other_diamond(pb, pa)
                esc_a = _escape_step_along_other_diamond(pa, pb)

                move_choice = None
                mover_id = None
                if esc_b is not None:
                    move_choice = esc_b; mover_id = b
                elif esc_a is not None:
                    move_choice = esc_a; mover_id = a

                if move_choice is not None:
                    v, dmd, direction = move_choice
                    pending_updates: Dict[int, Coord] = {}
                    # wie gehabt: rotieren, aber beide Paar-Qubits ausnehmen
                    rot = _compute_diamond_rotation_updates(dmd, direction, exclude={a, b})
                    pending_updates.update(rot)
                    pending_updates[mover_id] = v

                    for qid, newp in pending_updates.items():
                        current_pos[qid] = newp

                    # Positionen & Zeit updaten
                    pa = current_pos[a]; pb = current_pos[b]
                    t += 1
                    append_all_positions()

                    # Pfade zum GLEICHEN Meeting neu berechnen
                    try:
                        pathA = _shortest_path_nodes(pa, meet)
                        pathB = _shortest_path_nodes(pb, meet)
                    except nx.NetworkXNoPath:
                        t += 1
                        append_all_positions()
                        continue

                    preA = _entry_sn_from_path_nodes(pathA, meet)
                    preB = _entry_sn_from_path_nodes(pathB, meet)
                    if preA is None or preB is None:
                        t += 1
                        append_all_positions()
                        continue

                    preA = _entry_sn_from_path_nodes(pathA, meet)
                    preB = _entry_sn_from_path_nodes(pathB, meet)
                    if preA is None or preB is None:
                        t += 1
                        append_all_positions()
                        continue

            # --- Pfade bis PRE-IN (inklusive Start, inklusive PRE-IN)
            pathA_to_pre = pathA[: pathA.index(meet)]
            pathB_to_pre = pathB[: pathB.index(meet)]

            # --- Schrittweise Ausführung bis beide PRE-IN erreicht ---
            idxA = 0
            idxB = 0

            while (current_pos[a] != preA) or (current_pos[b] != preB):
                # Reihenfolge: A bewegt → (Raute-Rotation) → B bewegt → (Raute-Rotation)

                # Move A
                if current_pos[a] != preA and idxA + 1 < len(pathA_to_pre):
                    u = pathA_to_pre[idxA]
                    v = pathA_to_pre[idxA + 1]
                    pending_updates: Dict[int, Coord] = {}
                    diamond = _diamond_for_edge(u, v)
                    if diamond is not None:
                        direction = _rotation_direction(diamond, u, v)
                        rot = _compute_diamond_rotation_updates(diamond, direction, exclude={a,b})
                        pending_updates.update(rot)
                    pending_updates[a] = v
                    for qid, newp in pending_updates.items():
                        current_pos[qid] = newp
                    idxA += 1

                # Move B
                if current_pos[b] != preB and idxB + 1 < len(pathB_to_pre):
                    u = pathB_to_pre[idxB]
                    v = pathB_to_pre[idxB + 1]
                    pending_updates: Dict[int, Coord] = {}
                    diamond = _diamond_for_edge(u, v)
                    if diamond is not None:
                        direction = _rotation_direction(diamond, u, v)
                        rot = _compute_diamond_rotation_updates(diamond, direction, exclude={b,a})
                        pending_updates.update(rot)
                    pending_updates[b] = v
                    for qid, newp in pending_updates.items():
                        current_pos[qid] = newp
                    idxB += 1

                t += 1
                append_all_positions()

            # --- kurzer IN-Hop synchron für beide (2 Zeitschritte) ---
            current_pos[a] = meet
            current_pos[b] = meet
            t += 1
            append_all_positions()

            current_pos[a] = preA
            current_pos[b] = preB
            t += 1
            append_all_positions()

        edge_timebands: List[Tuple[int, int, Set[frozenset]]] = []
        return timelines, edge_timebands
