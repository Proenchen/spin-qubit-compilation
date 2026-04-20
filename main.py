import random

from routing.routing_strategy import RoutingStrategy
from routing.routing_with_reroute import RerouteRoutingPlanner
from routing.default_routing import DefaultRoutingPlanner
from routing.rotation_routing import RotationRoutingPlanner
from routing.rotation_cycles_routing import HybridRotationRoutingPlanner
from placements.placement_strategy import PlacementStrategy
from placements.random_strategy import RandomPlacementStrategy
from placements.reverse_traversal_strategy import ReverseTraversalPlacementStrategy
from placements.interaction_placement_strategy import InteractionPlacementStrategy
from simulation import SimulationConfig, RoutingSimulator


if __name__ == "__main__":

    placement: PlacementStrategy = RandomPlacementStrategy()
    routing: RoutingStrategy = RotationRoutingPlanner()

    # For defective edges
    random.seed(10)

    config = SimulationConfig(
        width=3,
        height=3,
        n_qubits=8,
        rounds=3,
        p_success=1,
        p_repair=0.05,

        # For Qubit-Placement and interactions
        seed=13
    )

    simulator = RoutingSimulator(
        placement_strategy=placement,
        routing_strategy=routing,
        config=config,
    )

    simulator.run()
    
