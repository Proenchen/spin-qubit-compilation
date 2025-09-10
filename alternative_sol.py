# -*- coding: utf-8 -*-
"""
Rendezvous + MAPF (ohne Labels, optimiert)
-----------------------------------------
Ziel:
- Für jedes Qubit-Paar den "mittleren" IN-Knoten (gemäß kürzester Wege) finden
- Beide Qubits kollisionsfrei und gleichzeitig dort ankommen lassen
- IN-Knoten dürfen Kapazität 2 (Rendezvous), SN-Knoten Kapazität 1
- Kollisionsfreiheit: zeitliche Reservierungen auf Nodes/Edges (keine Edge-Swaps)
- A*: lexikographische Kosten (erst Moves minimieren, dann Zeit) → Warten wird bevorzugt

"""

from __future__ import annotations

from heapq import heappush, heappop
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
from animation import animate_mapf


# --------------------------
# Netzwerk & Hilfsfunktionen
# --------------------------

Coord = Tuple[int, int]
TimedNode = Tuple[Coord, int]


def build_network() -> nx.Graph:
    """
    Erzeugt ein 2x2-Kachel-Layout aus 8er-Tiles (ohne Zentrum),
    verbindet 8-nachbarschaftlich.
    - Knoten-Attribut 'typ' ∈ {'IN', 'SN'}
    """
    template_pos = {
        0: (-1,  1), 1: (0, 1), 2: (1, 1),
        3: (-1,  0),            4: (1, 0),
        5: (-1, -1), 6: (0, -1), 7: (1, -1),
    }
    corners = {0, 2, 5, 7}
    tile_offsets = [(0, 0), (2, 0), (0, -2), (2, -2)]

    G = nx.Graph()
    for dx, dy in tile_offsets:
        for t_id, (x, y) in template_pos.items():
            coord = (x + dx, y + dy)
            if coord not in G:
                G.add_node(coord, typ=("IN" if t_id in corners else "SN"))

    coords = list(G.nodes())
    coord_set = set(coords)
    for (x, y) in coords:
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                if ddx == 0 and ddy == 0:
                    continue
                v = (x + ddx, y + ddy)
                if v in coord_set:
                    G.add_edge((x, y), v)
    return G


def chebyshev(a: Coord, b: Coord) -> int:
    """Chebyshev-Distanz in Gitterkoordinaten."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def _extend_waits_until(node: Coord, path: List[TimedNode], T_minus_1: int, res: Reservations) -> Optional[List[TimedNode]]:
    """Hängt am Endknoten 'node' Waits an, bis Zeit T-1 erreicht ist (inkl. Reservierung)."""
    if not path:
        return None
    ext = path[:]
    while ext[-1][1] < T_minus_1:
        n, t = ext[-1]
        t2 = t + 1
        if n != node:
            return None
        if not res.can_occupy(node, t2):
            return None
        res.occupy_node(node, t2)
        ext.append((node, t2))
    return ext

def _find_egress_targets(
    G: nx.Graph, meeting: Coord, k: int, T: int, res: Reservations
) -> Optional[List[Coord]]:
    """
    Wählt bis zu k verschiedene Nachbarn für einen Move aus meeting @ (T-1)->T.
    - Prüft: can_traverse(meeting, nbr, T-1) und can_occupy(nbr, T)
    - Bevorzugt SN, dann IN
    - Keine Doppelbelegung desselben Nachbarn
    """
    nbrs = list(G.neighbors(meeting))
    sn = [v for v in nbrs if G.nodes[v]["typ"] == "SN"]
    inn = [v for v in nbrs if G.nodes[v]["typ"] == "IN"]
    candidates = sn + inn

    chosen: List[Coord] = []
    for v in candidates:
        if v in chosen:
            continue
        if not res.can_traverse(meeting, v, T - 1):
            continue
        if not res.can_occupy(v, T):
            continue
        chosen.append(v)
        if len(chosen) == k:
            return chosen
    return None

def _egress_pair_at_time(
    G: nx.Graph,
    meeting: Coord,
    agents: List[str],
    plans: Dict[str, List[TimedNode]],
    T: int,
    res: Reservations,
) -> bool:
    """
    Bewegt die beiden Agenten eines früheren Paars genau in Schritt T aus dem IN heraus.
    - Hängt nötige Waits bis T-1 an
    - Reserviert Edge-Traversen (meeting->nbr) bei t=T-1 sowie Zielknoten @ T
    """
    if len(agents) != 2:
        return False
    a0, a1 = agents
    p0, p1 = plans[a0], plans[a1]
    # Beide müssen tatsächlich am meeting enden
    if not p0 or not p1 or p0[-1][0] != meeting or p1[-1][0] != meeting:
        return False

    # Waits bis T-1 (falls nötig)
    t0 = p0[-1][1]
    t1 = p1[-1][1]
    Tmax_needed = T - 1
    if t0 > Tmax_needed or t1 > Tmax_needed:
        # Dieses Paar ist später angekommen als T-1 -> nichts zu tun
        return True

    p0e = _extend_waits_until(meeting, p0, Tmax_needed, res)
    if p0e is None:
        return False
    p1e = _extend_waits_until(meeting, p1, Tmax_needed, res)
    if p1e is None:
        return False

    # Zwei verschiedene Egress-Ziele finden
    targets = _find_egress_targets(G, meeting, 2, T, res)
    if targets is None:
        return False

    # Reservieren & anhängen (Reihenfolge stabil, aber Ziele verschieden)
    for agent, tgt, path in [(a0, targets[0], p0e), (a1, targets[1], p1e)]:
        # Kante meeting->tgt bei t=T-1, Ziel @ T
        if not res.can_traverse(meeting, tgt, T - 1):
            return False
        if not res.can_occupy(tgt, T):
            return False
        res.traverse_edge(meeting, tgt, T - 1)
        res.occupy_node(tgt, T)
        path.append((tgt, T))
        plans[agent] = path  # update

    return True
# ---------------------------
# Zeitliche Reservierungen
# ---------------------------

class Reservations:
    """
    Führt kapazitätsbehaftete Zeitreservierungen auf Knoten und Kanten:
      - node_caps[node][t] < capacity(node)
      - edge_caps[{u,v}][t] ∈ {0,1} verhindert gleichzeitige Gegenläufigkeit (Edge-Swap)
    """

    def __init__(self, G: nx.Graph) -> None:
        self.node_caps: Dict[Coord, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.edge_caps: Dict[frozenset, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.node_type: Dict[Coord, str] = {n: G.nodes[n]["typ"] for n in G.nodes()}

    def node_capacity(self, node: Coord) -> int:
        # IN darf 2 (Rendezvous), SN nur 1
        return 2 if self.node_type[node] == "IN" else 1

    def can_occupy(self, node: Coord, t: int) -> bool:
        return self.node_caps[node][t] < self.node_capacity(node)

    def occupy_node(self, node: Coord, t: int) -> None:
        self.node_caps[node][t] += 1

    def can_traverse(self, u: Coord, v: Coord, t: int) -> bool:
        # Edge-Kapazität 1 verhindert gleichzeitige gegenläufige Swaps
        return self.edge_caps[frozenset({u, v})][t] == 0

    def traverse_edge(self, u: Coord, v: Coord, t: int) -> None:
        self.edge_caps[frozenset({u, v})][t] += 1

    def commit(self, path: Iterable[TimedNode]) -> None:
        """Schreibt einen zeitlich indizierten Pfad in die Reservierungstabellen."""
        path = list(path)
        if not path:
            return
        self.occupy_node(*path[0])
        for (u, t), (v, t2) in zip(path[:-1], path[1:]):
            self.occupy_node(v, t2)
            if u != v:
                self.traverse_edge(u, v, t)


# ---------------------------------------------
# A* mit Lexiko-Kosten (Moves zuerst, dann Zeit)
# ---------------------------------------------

def a_star_with_reservations(
    G: nx.Graph,
    start: Coord,
    goal: Coord | None,
    reservations: Reservations,
    t0: int = 0,
    max_time: int = 200,
    goal_predicate: Optional[callable] = None,
) -> Optional[List[TimedNode]]:
    """
    Minimiert (moves, time) lexikographisch:
      - Jede Bewegung kostet (dmoves=1, dtime=1)
      - Warten kostet (dmoves=0, dtime=1)
    => unnötige Ausweichbewegungen werden vermieden; lieber warten

    Kollisionsfreiheit:
      - Knoten-/Kantenkapazitäten werden bei Expansion geprüft.
    """

    def is_goal(node: Coord) -> bool:
        if goal_predicate:
            return goal_predicate(node)
        if goal is not None:
            return node == goal
        return False

    def h_moves(n: Coord) -> int:
        # admissible Heuristik: minimal notwendige Moves (Chebyshev)
        return chebyshev(n, goal) if isinstance(goal, tuple) else 0

    # Startzeit ggf. nach vorne schieben, bis Startknoten frei ist
    start_time = None
    for dt in range(0, 41):
        if reservations.can_occupy(start, t0 + dt):
            start_time = t0 + dt
            break
    if start_time is None:
        # kein freier Slot im Suchfenster gefunden
        return None

    start_state = (start, start_time)
    g_cost: Dict[TimedNode, Tuple[int, int]] = {start_state: (0, 0)}  # (moves, time)
    came_from: Dict[TimedNode, TimedNode] = {}

    openq: List[Tuple[Tuple[int, int], TimedNode]] = []
    heappush(openq, ((h_moves(start), 0), start_state))

    def expand(node: Coord, t: int) -> Iterable[Tuple[Coord, int, int, int]]:
        # wait
        yield (node, t + 1, 0, 1)  # (n2,t2,dmoves,dtime)
        # move
        for nbr in G.neighbors(node):
            yield (nbr, t + 1, 1, 1)

    while openq:
        (_, _), (node, t) = heappop(openq)
        gm, gt = g_cost[(node, t)]

        if is_goal(node):
            # Pfad rekonstruieren
            path: List[TimedNode] = [(node, t)]
            cur = (node, t)
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return list(reversed(path))

        if t >= t0 + max_time:
            continue

        for (n2, t2, dm, dt) in expand(node, t):
            # Kapazitäten/Reservierungen prüfen
            if not reservations.can_occupy(n2, t2):
                continue
            if n2 != node and not reservations.can_traverse(node, n2, t):
                continue

            g2 = (gm + dm, gt + dt)  # lexikographisch

            old = g_cost.get((n2, t2))
            if old is None or g2 < old:
                g_cost[(n2, t2)] = g2
                came_from[(n2, t2)] = (node, t)
                f2 = (g2[0] + h_moves(n2), g2[1] + h_moves(n2))
                heappush(openq, (f2, (n2, t2)))

    return None


def compute_meeting_node(G: nx.Graph, q0: Coord, q1: Coord, reserved_meetings: set = None) -> Tuple[Coord, Dict[Coord, int], Dict[Coord, int]]:
    """
    Wählt einen IN-Knoten, der „in der Mitte“ liegt, aber bereits reservierte Meetings vermeidet.
    """
    if reserved_meetings is None:
        reserved_meetings = set()
    
    da = nx.single_source_shortest_path_length(G, q0)
    db = nx.single_source_shortest_path_length(G, q1)
    
    # Finde alle möglichen IN-Knoten
    all_in_nodes = [n for n in G if G.nodes[n]["typ"] == "IN" and n in da and n in db]
    
    # Sortiere nach Qualität (beste zuerst)
    sorted_candidates = sorted(
        all_in_nodes,
        key=lambda n: (max(da[n], db[n]), da[n] + db[n], abs(da[n] - db[n]), n)
    )
    
    # Wähle den besten nicht-reservierten Knoten
    for candidate in sorted_candidates:
        if candidate not in reserved_meetings:
            return candidate, da, db
    
    # Fallback: verwende den besten Knoten (auch wenn reserviert)
    if sorted_candidates:
        return sorted_candidates[0], da, db
    
    raise ValueError("Kein gemeinsamer IN-Knoten erreichbar.")


def synchronize_to_time(path: List[TimedNode], T: int, res: Reservations) -> Optional[List[TimedNode]]:
    """
    Lässt auf dem letzten Knoten warten, bis T erreicht ist (nur waits).
    Reserviert die entsprechenden Zeiten, bricht ab falls Kapazität blockiert.
    """
    if not path:
        return None
    ext = path[:]
    while ext[-1][1] < T:
        n, t = ext[-1]
        t2 = t + 1
        if not res.can_occupy(n, t2):
            return None
        res.occupy_node(n, t2)
        ext.append((n, t2))
    return ext


# ---------------------------------
# Planung: paarweises Rendezvous
# ---------------------------------

def prioritized_pairwise_mapf(
    G: nx.Graph,
    pairs: List[Tuple[Coord, Coord]],
    order: str = "longer_first",
) -> Tuple[List[dict], Dict[str, List[TimedNode]]]:
    infos = []
    reserved_meetings = set()  # Track bereits verwendete Meeting-Knoten
    
    for i, (q0, q1) in enumerate(pairs):
        m, da, db = compute_meeting_node(G, q0, q1, reserved_meetings)
        reserved_meetings.add(m)  # Reserviere diesen Meeting-Knoten
        infos.append({
            "pair_id": i,
            "q0": q0,
            "q1": q1,
            "meeting": m,
            "d0": da[m],
            "d1": db[m],
            "maxd": max(da[m], db[m]),
            "sumd": da[m] + db[m],
        })

    plan_order = (
        sorted(infos, key=lambda p: (p["maxd"], p["sumd"]), reverse=True)
        if order == "longer_first" else infos
    )

    res = Reservations(G)
    plans: Dict[str, List[TimedNode]] = {}

    meeting_log: Dict[Coord, List[dict]] = defaultdict(list)

    for p in plan_order:
        aid, bid = f"pair{p['pair_id']}_A", f"pair{p['pair_id']}_B"
        meeting = p["meeting"]

        a = a_star_with_reservations(G, p["q0"], meeting, res, t0=0, max_time=300)
        if a is None:
            raise RuntimeError(f"Keine Route für {aid} zum Meeting {meeting}")
        res.commit(a)

        b = a_star_with_reservations(G, p["q1"], meeting, res, t0=0, max_time=300)
        if b is None:
            raise RuntimeError(f"Keine Route für {bid} zum Meeting {meeting}")
        res.commit(b)

        # Synchronisieren
        Ta, Tb = a[-1][1], b[-1][1]
        Tm = max(Ta, Tb)

        if Ta < Tm:
            a = synchronize_to_time(a, Tm, res)
            if a is None:
                raise RuntimeError(f"Synchronisation fehlgeschlagen für {aid} @ {meeting} bis t={Tm}")
        if Tb < Tm:
            b = synchronize_to_time(b, Tm, res)
            if b is None:
                raise RuntimeError(f"Synchronisation fehlgeschlagen für {bid} @ {meeting} bis t={Tm}")

        plans[aid] = a
        plans[bid] = b

        if meeting_log[meeting]:
            for rec in meeting_log[meeting]:
                if not rec.get("evacuated") and rec["T"] < Tm:
                    ok = _egress_pair_at_time(G, meeting, rec["agents"], plans, Tm, res)
                    if ok:
                        rec["evacuated"] = True

        later_times = [rec["T"] for rec in meeting_log[meeting] if rec["T"] > Tm and not rec.get("evacuated")]
        evacuated_now = False
        if later_times:
            T_next = min(later_times)
            ok = _egress_pair_at_time(G, meeting, [aid, bid], plans, T_next, res)
            evacuated_now = bool(ok)

        meeting_log[meeting].append({
            "agents": [aid, bid],
            "T": Tm,
            "evacuated": evacuated_now,
        })

    return plan_order, plans



# --------------
# Main
# --------------

if __name__ == "__main__":
    G = build_network()
    #pairs = [((3, 0), (-1, -2)), ((-1, 0), (3, -2))]
    #pairs = [((1, -2), (1, 0)), ((-1, 0), (3, -2)), ((0, -3), (2, 1))]
    #pairs = [((1, -2), (1, 0)), ((-1, 0), (3, -2)), ((0, -3), (2, 1)), ((2, -3), (3, 0))]
    #pairs = [((1, -2), (1, 0)), ((-1, 0), (3, -2)), ((0, -3), (2, 1)), ((2, -3), (3, 0)), ((2, -1), (0, 1))]
    pairs = [((1, -2), (1, 0)), ((-1, 0), (3, -2)), ((0, -3), (2, 1)), ((2, -3), (3, 0)), ((2, -1), (0, 1)),((0, -1), (-1, -2))]
    _, plans = prioritized_pairwise_mapf(G, pairs)

    for agent_id, path in plans.items():
        trace = " -> ".join(f"{n}@t{t}" for n, t in path)
        print(f"{agent_id}: {trace}")

    animate_mapf(G, plans)