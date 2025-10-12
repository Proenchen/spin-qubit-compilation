import random

from routing.network import NetworkBuilder
from routing.default_routing import DefaultRoutingPlanner
from routing.routing_with_reroute import RoutingPlannerWithRerouting
from utils.animation import animate_mapf

if __name__ == "__main__":
    G, qubits, pairs = NetworkBuilder.place_qubits_and_make_pairs(
        width=3,
        height=3,
        n_qubits=6,
        rounds=6,
        seed=42,  
    )
    random.seed()

    """ qubits: List[Qubit] = [
        Qubit(0, (1, -2)),
        Qubit(1, (1,  0)),
        Qubit(2, (-1, 0)),
        Qubit(3, (3, -2)),
        Qubit(4, (0, -3)),
        Qubit(5, (2,  1))
    ]

    pairs: List[Tuple[Qubit, Qubit]] = [
        (qubits[0], qubits[1]),  #1
        (qubits[0], qubits[2]),  #2
        (qubits[2], qubits[3]),  #3
        (qubits[4], qubits[5]),  #3
        (qubits[0], qubits[2]),  #4
        (qubits[1], qubits[4]),  #4
        (qubits[3], qubits[5]),  #4
        (qubits[0], qubits[5]),  #5
        (qubits[1], qubits[3]),  #5
        (qubits[4], qubits[5]),  #6
    ] """

    #planner = DefaultRoutingPlanner()
    planner = RoutingPlannerWithRerouting()
    timelines, edge_timebands = planner.route(G, qubits, pairs)
    animate_mapf(G, timelines, edge_timebands=edge_timebands, smooth=True)
