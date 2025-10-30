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
            # Finde w ∈ Su und x ∈ Sv, die diagonal verbunden sind (schließt den 4er-Zyklus)
            for w in Su:
                for x in Sv:
                    if _is_diagonal(w, x):
                        # Prüfe, dass auch w<->u diagonal ist (per Konstruktion ja) und x<->v diagonal (ja)
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

            # Meeting-Kandidaten wie im Default, erster gültiger wird genommen
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
                # Kein Meeting möglich (sollte praktisch nicht vorkommen auf diesem G)
                # Lasse alle eine Zeiteinheit warten und mache weiter.
                t += 1
                append_all_positions()
                continue

            preA = _entry_sn_from_path_nodes(pathA, meet)
            preB = _entry_sn_from_path_nodes(pathB, meet)
            if preA is None or preB is None:
                # Sicherheitsnetz: wenn PRE-IN nicht bestimmbar, warten wir eine Einheit und weiter
                t += 1
                append_all_positions()
                continue

            # Pfade bis PRE-IN (inklusive Start, inklusive PRE-IN)
            pathA_to_pre = pathA[: pathA.index(meet)]  # endet auf preA
            pathB_to_pre = pathB[: pathB.index(meet)]  # endet auf preB

            # --- Schrittweise Ausführung bis beide PRE-IN erreicht ---
            idxA = 0
            idxB = 0

            while (current_pos[a] != preA) or (current_pos[b] != preB):
                # Einen Zeitschritt simulieren.
                # Reihenfolge: A bewegt (falls nötig) → evtl. Raute-Rotation → B bewegt → evtl. Raute-Rotation.
                # Danach Zeit +1 und alle Positionen loggen.

                # Move A (wenn noch nicht am PRE-IN)
                if current_pos[a] != preA and idxA + 1 < len(pathA_to_pre):
                    u = pathA_to_pre[idxA]
                    v = pathA_to_pre[idxA + 1]

                    # Sammle atomare Updates dieses Sub-Schritts
                    pending_updates: Dict[int, Coord] = {}

                    # 1) ggf. Raute-Rotation vorbereiten (alle außer a,b)
                    diamond = _diamond_for_edge(u, v)
                    if diamond is not None:
                        direction = _rotation_direction(diamond, u, v)
                        rot = _compute_diamond_rotation_updates(diamond, direction, exclude={a, b})
                        pending_updates.update(rot)

                    # 2) PaarQubit A Schritt vorbereiten
                    pending_updates[a] = v

                    # 3) Anwenden (atomar)
                    for qid, newp in pending_updates.items():
                        current_pos[qid] = newp
                    idxA += 1

                # Move B (wenn noch nicht am PRE-IN)
                if current_pos[b] != preB and idxB + 1 < len(pathB_to_pre):
                    u = pathB_to_pre[idxB]
                    v = pathB_to_pre[idxB + 1]

                    pending_updates: Dict[int, Coord] = {}

                    diamond = _diamond_for_edge(u, v)
                    if diamond is not None:
                        direction = _rotation_direction(diamond, u, v)
                        rot = _compute_diamond_rotation_updates(diamond, direction, exclude={a, b})
                        pending_updates.update(rot)

                    pending_updates[b] = v

                    for qid, newp in pending_updates.items():
                        current_pos[qid] = newp
                    idxB += 1

                # Zeit fortschreiben und Snapshot schreiben
                t += 1
                append_all_positions()

            # --- kurzer IN-Hop synchron für beide (2 Zeitschritte) ---
            # Schritt 1: beide in den Meeting-IN
            current_pos[a] = meet
            current_pos[b] = meet
            t += 1
            append_all_positions()

            # Schritt 2: beide zurück auf ihre PRE-IN
            current_pos[a] = preA
            current_pos[b] = preB
            t += 1
            append_all_positions()

        # Lückenlose Zeitleisten sind bereits garantiert (wir haben bei jedem t geschrieben).
        # edge_timebands: hier leer (keine Defekte/Sampling).
        edge_timebands: List[Tuple[int, int, Set[frozenset]]] = []

        return timelines, edge_timebands
