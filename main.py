import random

from typing import List, Tuple
from routing.common import Qubit
from routing.network import NetworkBuilder
from routing.routing_with_reroute import RerouteRoutingPlanner
from routing.default_routing import DefaultRoutingPlanner
from routing.rotation_routing import RotationRoutingPlanner
from utils.animation import animate_mapf

if __name__ == "__main__":
    random.seed(5)
    G, qubits, pairs = NetworkBuilder.place_qubits_and_make_pairs(
        width=3,
        height=3,
        n_qubits=10,
        rounds=3,
        seed=24,  
    )
    """ qubits: List[Qubit] = [
        Qubit(0, (2, -3)),
        Qubit(1, (1,  0)),
        Qubit(2, (-1, 0)),
        Qubit(3, (2, -1)),
        Qubit(4, (-1, -2)),
        Qubit(5, (2,  1)),
    ]

    pairs: List[Tuple[Qubit, Qubit]] = [
        (qubits[2], qubits[3]),  #1
        (qubits[0], qubits[2]),  #2
        (qubits[2], qubits[3]),  #3
        (qubits[4], qubits[5]),  #3
        (qubits[0], qubits[2]),  #4
        (qubits[1], qubits[4]),  #4
        (qubits[3], qubits[5]),  #4
        (qubits[0], qubits[5]),  #5
        (qubits[1], qubits[3]),  #5
        (qubits[4], qubits[5]),  #6 
    ] 
 """
    #planner = DefaultRoutingPlanner()
    #planner = RerouteRoutingPlanner()
    planner = RotationRoutingPlanner()
    timelines, edge_timebands = planner.route(G, qubits, pairs)
    animate_mapf(G, timelines, edge_timebands=edge_timebands, smooth=True)
