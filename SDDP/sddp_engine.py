import numpy as np
import random

__all__ = [
    "generate_scenarios",
    "get_stage_info"
]


def generate_scenarios(scenario_trees, problem):
    scenarios = []
    if problem == "ProductionPlanning":
        for tree in scenario_trees:
            scenario = random.sample(tree, 1)[0]
            scenarios.append(scenario)
    elif problem == ("EnergyPlanning" or "InventoryControl"):
        for tree in scenario_trees:
            scenario = [random.sample(nodes, 1)[0] for nodes in tree]
            scenarios.append(scenario)
    else:
        raise ValueError

    return scenarios


def get_stage_info(scenarios, constraints, stage_idx, solutions, problem):
    if problem == "EnergyPlanning":
        water_flow = scenarios[0][stage_idx]
        demand = scenarios[1][stage_idx]
        constraints["stage{}".format(stage_idx)][0, -1] = - solutions[2] - water_flow
        constraints["stage{}".format(stage_idx)][2, -1] = demand
        return constraints, water_flow, demand
    elif problem == "ProductionPlanning":
        demand = scenarios[stage_idx]
        constraints["stage{}".format(stage_idx)][:3, -1] = np.array(demand) - solutions[3]
        return constraints, demand
    elif problem == "InventoryControl":
        pass
    else:
        raise ValueError
