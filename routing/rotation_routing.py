from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from routing.common import Coord, TimedNode, Qubit
from routing.default_routing import DefaultRoutingPlanner
from routing.routing_strategy import RoutingStrategy

MAX_WAIT_TIME = 100


class RotationRoutingPlanner(RoutingStrategy):
    def route(
        self,
        G: nx.Graph,
        qubits: List[Qubit],
        pairs: List[Tuple[Qubit, Qubit]],
        p_success: float,
        p_repair: float,
    ):
        rt = RouteRuntime(G, qubits, p_success, p_repair)

        live_pre_by_pair: Dict[Tuple[int, int], Dict[int, Coord]] = {}
        pair_order: Dict[Tuple[int, int], int] = {(qa.id, qb.id): i for i, (qa, qb) in enumerate(pairs)}
        remaining: Set[Tuple[int, int]] = {(qa.id, qb.id) for qa, qb in pairs}

        while remaining:
            ready = [pid for pid in remaining if is_ready_pair(pid, remaining, pair_order)]
            if not ready:
                ready = list(remaining)

            plans: Dict[Tuple[int, int], SoloPlan] = {}
            sequential_fallback: List[Tuple[int, int]] = []

            for pid in sorted(ready, key=lambda p: pair_order[p]):
                a, b = pid
                plan = self._plan_pair_solo(rt, a, b)
                if plan is None:
                    sequential_fallback.append(pid)
                else:
                    plans[pid] = plan

            if not plans and not sequential_fallback:
                rt.commit_tick({}, sample=True)
                break

            groups = group_parallel(plans) if plans else []
            finished = False

            for grp in groups:
                group_qids: Set[int] = {x for ab in grp for x in ab}
                L = max(plans[pid].length for pid in grp) if grp else 0

                parallel_failed = False
                step = 0
                while step < L:
                    updates_pair_only: Dict[int, Coord] = {}
                    sample_flags: List[bool] = []
                    step_diamonds: List[Tuple[List[Coord], int]] = []
                    pending_live_pre: Dict[Tuple[int, int], Dict[int, Coord]] = {}

                    for pid in grp:
                        plan = plans[pid]
                        s = plan.ticks[step] if step < plan.length else SoloStep({}, True, [])

                        if plan.in_idx is not None and step == plan.in_idx:
                            a_id, b_id = pid
                            pending_live_pre[pid] = {a_id: rt.current_pos[a_id], b_id: rt.current_pos[b_id]}

                        if plan.out_idx is not None and step == plan.out_idx:
                            a_id, b_id = pid
                            pre_map = live_pre_by_pair.get(pid, {a_id: rt.current_pos[a_id], b_id: rt.current_pos[b_id]})
                            s = SoloStep({a_id: pre_map[a_id], b_id: pre_map[b_id]}, False, [])

                        if set(updates_pair_only.keys()) & set(s.updates_pair_only.keys()):
                            parallel_failed = True
                            break

                        updates_pair_only.update(s.updates_pair_only)
                        sample_flags.append(s.sample)
                        step_diamonds.extend(s.diamonds)

                    if parallel_failed:
                        break

                    if foreign_qubits_on_any_diamond(rt.current_pos, step_diamonds, allowed=group_qids):
                        parallel_failed = True
                        break

                    updates = expand_runtime_rotations(rt.current_pos, updates_pair_only, step_diamonds)
                    do_sample = False not in sample_flags

                    moved = rt.commit_tick(updates, sample=do_sample)
                    if moved:
                        for pid, pre in pending_live_pre.items():
                            live_pre_by_pair[pid] = pre
                        step += 1
                    else:
                        continue

                if parallel_failed:
                    for pid in grp:
                        plan = plans[pid]
                        a_id, b_id = pid
                        step = 0
                        while step < plan.length:
                            s = plan.ticks[step]
                            pending_pre: Optional[Dict[int, Coord]] = None

                            if plan.in_idx is not None and step == plan.in_idx:
                                pending_pre = {a_id: rt.current_pos[a_id], b_id: rt.current_pos[b_id]}

                            if plan.out_idx is not None and step == plan.out_idx:
                                pre_map = live_pre_by_pair.get(pid, {a_id: rt.current_pos[a_id], b_id: rt.current_pos[b_id]})
                                s = SoloStep({a_id: pre_map[a_id], b_id: pre_map[b_id]}, False, [])

                            updates = expand_runtime_rotations(rt.current_pos, s.updates_pair_only, s.diamonds)
                            moved = rt.commit_tick(updates, sample=s.sample)
                            if moved:
                                if pending_pre is not None:
                                    live_pre_by_pair[pid] = pending_pre
                                step += 1
                            else:
                                continue

                for pid in grp:
                    remaining.discard(pid)

                finished = True
                break

            if finished:
                continue

            if sequential_fallback:
                pid = sequential_fallback[0]
                a, b = pid
                plan = self._plan_pair_solo(rt, a, b)

                if plan is None:
                    rt.commit_tick({}, sample=False)
                else:
                    step = 0
                    while step < plan.length:
                        s = plan.ticks[step]
                        pending_pre: Optional[Dict[int, Coord]] = None

                        if plan.in_idx is not None and step == plan.in_idx:
                            pending_pre = {a: rt.current_pos[a], b: rt.current_pos[b]}

                        if plan.out_idx is not None and step == plan.out_idx:
                            pre_map = live_pre_by_pair.get(pid, {a: rt.current_pos[a], b: rt.current_pos[b]})
                            s = SoloStep({a: pre_map[a], b: pre_map[b]}, False, [])

                        updates = expand_runtime_rotations(rt.current_pos, s.updates_pair_only, s.diamonds)
                        moved = rt.commit_tick(updates, sample=s.sample)
                        if moved:
                            if pending_pre is not None:
                                live_pre_by_pair[pid] = pending_pre
                            step += 1
                        else:
                            continue

                remaining.discard(pid)
            else:
                rt.commit_tick({}, sample=True)

        return rt.timelines, rt.edge_timebands

    def _plan_pair_solo(self, rt: RouteRuntime, a_id: int, b_id: int) -> Optional[SoloPlan]:
        la = rt.current_pos[a_id]
        lb = rt.current_pos[b_id]
        if not (is_sn(rt.G, la) and is_sn(rt.G, lb)):
            return None

        ticks: List[SoloStep] = []
        trace: Dict[int, List[Coord]] = {a_id: [la], b_id: [lb]}
        used_diamonds: Set[Tuple[Coord, Coord, Coord, Coord]] = set()
        in_idx: Optional[int] = None
        out_idx: Optional[int] = None

        cands = DefaultRoutingPlanner._best_meeting_candidates(
            rt.G, la, lb, reserved=set(), forbidden_nodes=set()
        )

        best_choice: Optional[Tuple[Coord, Tuple[Coord, List[Coord]], Tuple[Coord, List[Coord]]]] = None
        for meet in cands:
            if not is_in(rt.G, meet):
                continue

            pa = self._best_pre(rt, meet, la)
            pb = self._best_pre(rt, meet, lb)
            if pa and pb:
                best_choice = (meet, pa, pb)
                break

        if best_choice is None:
            return None

        meet, (preA, pathA), (preB, pathB) = best_choice
        idxA = 0
        idxB = 0

        while la != preA or lb != preB:
            updates: Dict[int, Coord] = {}
            diamonds_for_step: List[Tuple[List[Coord], int]] = []

            if la != preA and idxA + 1 < len(pathA):
                uA, vA = pathA[idxA], pathA[idxA + 1]
                if is_diag(uA, vA):
                    dA = diamond_for_edge(rt.G, uA, vA)
                    if dA:
                        dirA = rot_dir(dA, uA, vA)
                        updA = compute_pair_rotation_updates_for_diamond(dA, dirA, a_id, b_id, la, lb)
                        updA[a_id] = vA
                        updates.update(updA)
                        diamonds_for_step.append((dA, dirA))
                        used_diamonds.add(canonical_diamond_tuple(dA))
                    else:
                        updates[a_id] = vA
                else:
                    updates[a_id] = vA

            if lb != preB and idxB + 1 < len(pathB):
                uB, vB = pathB[idxB], pathB[idxB + 1]
                if is_diag(uB, vB):
                    dB = diamond_for_edge(rt.G, uB, vB)
                    if dB:
                        dirB = rot_dir(dB, uB, vB)
                        updB = compute_pair_rotation_updates_for_diamond(dB, dirB, a_id, b_id, la, lb)
                        updB[b_id] = vB
                        if not (diamonds_for_step and set(diamonds_for_step[0][0]).intersection(dB)):
                            updates.update(updB)
                            diamonds_for_step.append((dB, dirB))
                            used_diamonds.add(canonical_diamond_tuple(dB))
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

        updates_in: Dict[int, Coord] = {}
        if la != meet:
            updates_in[a_id] = meet
        if lb != meet:
            updates_in[b_id] = meet

        if updates_in:
            la = updates_in.get(a_id, la)
            lb = updates_in.get(b_id, lb)
            in_idx = len(ticks)
            ticks.append(SoloStep(updates_pair_only=updates_in, sample=True, diamonds=[]))
            trace[a_id].append(la)
            trace[b_id].append(lb)

        out_idx = len(ticks)
        ticks.append(SoloStep(updates_pair_only={}, sample=False, diamonds=[]))
        trace[a_id].append(preA)
        trace[b_id].append(preB)

        return SoloPlan(ticks=ticks, pos_trace=trace, used_diamonds=used_diamonds, in_idx=in_idx, out_idx=out_idx)

    def _best_pre(self, rt: RouteRuntime, meet: Coord, src: Coord) -> Optional[Tuple[Coord, List[Coord]]]:
        best: Optional[Tuple[Coord, List[Coord]]] = None
        for pre in sn_neighbors_of_meet(rt.G, meet):
            p = shortest_path_sn(rt.SN, src, pre)
            if p is None:
                continue
            if best is None or len(p) < len(best[1]):
                best = (pre, p)
        return best


def chebyshev(p: Coord, q: Coord) -> int:
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def edgeset(u: Coord, v: Coord) -> frozenset:
    return frozenset({u, v})


def is_sn(G: nx.Graph, n: Coord) -> bool:
    return G.nodes[n].get("type") == "SN"


def is_in(G: nx.Graph, n: Coord) -> bool:
    return G.nodes[n].get("type") == "IN"


def is_diag(u: Coord, v: Coord) -> bool:
    return abs(u[0] - v[0]) == 1 and abs(u[1] - v[1]) == 1


def canonical_diamond_tuple(D: List[Coord]) -> Tuple[Coord, Coord, Coord, Coord]:
    return tuple(sorted(D))  # type: ignore[return-value]


def diag_sn_neighbors(G: nx.Graph, n: Coord) -> List[Coord]:
    if not is_sn(G, n):
        return []
    return [w for w in G.neighbors(n) if is_sn(G, w) and is_diag(n, w)]


def diamond_for_edge(G: nx.Graph, u: Coord, v: Coord) -> Optional[List[Coord]]:
    if not (is_sn(G, u) and is_sn(G, v) and is_diag(u, v)):
        return None
    su = [w for w in diag_sn_neighbors(G, u) if w != v]
    sv = [x for x in diag_sn_neighbors(G, v) if x != u]
    for w in su:
        for x in sv:
            if is_diag(w, x):
                return [u, v, x, w]
    return None


def rot_dir(diamond: List[Coord], u: Coord, v: Coord) -> int:
    i = diamond.index(u)
    return 1 if diamond[(i + 1) % 4] == v else -1


def sn_neighbors_of_meet(G: nx.Graph, meeting: Coord) -> List[Coord]:
    if not is_in(G, meeting):
        return []
    return [w for w in G.neighbors(meeting) if is_sn(G, w)]


def shortest_path_sn(SN: nx.Graph, src: Coord, dst: Coord) -> Optional[List[Coord]]:
    try:
        return nx.shortest_path(SN, src, dst)
    except nx.NetworkXNoPath:
        return None


def compute_pair_rotation_updates_for_diamond(
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


def plans_compatible_distance(
    p1: "SoloPlan",
    ab1: Tuple[int, int],
    p2: "SoloPlan",
    ab2: Tuple[int, int],
) -> bool:
    a1, b1 = ab1
    a2, b2 = ab2
    L = max(p1.length, p2.length)

    def pos(plan: "SoloPlan", qid: int, i: int) -> Coord:
        trace = plan.pos_trace[qid]
        return trace[i] if i < len(trace) else trace[-1]

    for i in range(L + 1):
        p_a1 = pos(p1, a1, i)
        p_b1 = pos(p1, b1, i)
        p_a2 = pos(p2, a2, i)
        p_b2 = pos(p2, b2, i)
        for u in (p_a1, p_b1):
            for v in (p_a2, p_b2):
                if chebyshev(u, v) < 3:
                    return False
    return True


def plans_compatible_diamonds(p1: "SoloPlan", p2: "SoloPlan") -> bool:
    L = max(p1.length, p2.length)
    for i in range(L):
        d1 = (
            {canonical_diamond_tuple(D) for (D, _dir) in p1.ticks[i].diamonds}
            if i < p1.length
            else set()
        )
        d2 = (
            {canonical_diamond_tuple(D) for (D, _dir) in p2.ticks[i].diamonds}
            if i < p2.length
            else set()
        )
        if not d1.isdisjoint(d2):
            return False
    return True


def group_parallel(plans: Dict[Tuple[int, int], "SoloPlan"]) -> List[List[Tuple[int, int]]]:
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
                plans_compatible_distance(plans[p0], p0, plans[qid], qid)
                and plans_compatible_diamonds(plans[p0], plans[qid])
                for p0 in grp
            )
            if ok:
                grp.append(qid)
                used.add(qid)
        groups.append(grp)

    return groups


def expand_runtime_rotations(
    current_pos: Dict[int, Coord],
    base_updates: Dict[int, Coord],
    diamonds: List[Tuple[List[Coord], int]],
    exclude_qids: Optional[Set[int]] = None,
) -> Dict[int, Coord]:
    if not diamonds:
        return dict(base_updates)

    exclude_qids = exclude_qids or set()
    idx_cache: Dict[Tuple[Coord, Coord, Coord, Coord], Dict[Coord, int]] = {}
    out = dict(base_updates)

    for D, direction in diamonds:
        key = tuple(D)  # type: ignore[assignment]
        if key not in idx_cache:
            idx_cache[key] = {p: i for i, p in enumerate(D)}
        idx = idx_cache[key]
        for qid, pos in current_pos.items():
            if qid in exclude_qids:
                continue
            if pos in idx:
                out[qid] = D[(idx[pos] + direction) % 4]

    return out


def foreign_qubits_on_any_diamond(
    current_pos: Dict[int, Coord],
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


def is_ready_pair(
    pid: Tuple[int, int],
    remaining: Set[Tuple[int, int]],
    pair_order: Dict[Tuple[int, int], int],
) -> bool:
    idx = pair_order[pid]
    a, b = pid
    for other in remaining:
        if other == pid:
            continue
        if pair_order[other] >= idx:
            continue
        x, y = other
        if x in (a, b) or y in (a, b):
            return False
    return True


@dataclass(frozen=True)
class SoloStep:
    updates_pair_only: Dict[int, Coord]
    sample: bool
    diamonds: List[Tuple[List[Coord], int]]


class SoloPlan:
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


class RouteRuntime:
    def __init__(
        self,
        G: nx.Graph,
        qubits: List[Qubit],
        p_success: float,
        p_repair: float,
    ):
        self.G = G
        self.p_success = p_success
        self.p_repair = p_repair

        self.current_pos: Dict[int, Coord] = {q.id: q.pos for q in qubits}
        self.all_qids: Set[int] = {q.id for q in qubits}

        self.timelines: Dict[int, List[TimedNode]] = {q.id: [(q.pos, 0)] for q in qubits}
        self.t = 0
        self.wait_streak = 0

        self.defective_edges: Set[frozenset] = set()
        self.edge_timebands: List[Tuple[int, int, Set[frozenset]]] = []

        sn_nodes = [n for n in G.nodes() if is_sn(G, n)]
        self.SN = G.subgraph(sn_nodes).copy()

    def sample_edge_failures(self) -> None:
        for u, v in self.G.edges():
            e = edgeset(u, v)
            if e in self.defective_edges:
                if random.random() < self.p_repair:
                    self.defective_edges.discard(e)
            else:
                if random.random() < (1.0 - self.p_success):
                    self.defective_edges.add(e)

    def would_use_defect(self, pending: Dict[int, Coord]) -> bool:
        for qid, newp in pending.items():
            u = self.current_pos[qid]
            v = newp
            if u != v and edgeset(u, v) in self.defective_edges:
                return True
        return False

    def commit_tick(self, pending: Dict[int, Coord], *, sample: bool) -> bool:
        if sample:
            self.sample_edge_failures()

        moved = False
        if pending and not self.would_use_defect(pending):
            for qid, newp in pending.items():
                self.current_pos[qid] = newp
            moved = True

        self.wait_streak = 0 if moved else (self.wait_streak + 1)

        self.t += 1
        for qid in self.all_qids:
            last = self.timelines[qid][-1]
            cur = (self.current_pos[qid], self.t)
            if last != cur:
                self.timelines[qid].append(cur)

        self.edge_timebands.append((self.t - 1, self.t, set(self.defective_edges)))

        if self.wait_streak >= MAX_WAIT_TIME:
            raise RuntimeError(
                f"Routing stuck: {self.wait_streak} aufeinanderfolgende Timesteps "
                f"ohne Bewegung (t={self.t})."
            )

        return moved
