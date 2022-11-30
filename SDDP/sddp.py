import time
import random
import numpy as np
import attr
import argparse
import scipy.stats as st

from collections import Counter

__all__ = [
    "SDDP"
]


class SDDP:
    def __init__(self, prob_class,
                 n_samples=20,
                 stopping_criterion_alpha=0.05,
                 stopping_criterion_threshold=0.1):

        self.prob_class = prob_class
        self.prob_name = self.prob_class.prob_name
        self.n_stage = self.prob_class.n_stages

        # SDDP related
        self.stage = self.prob_class.stage

        self.objective_value = None
        self.value_func = None
        self.solution_set = None
        self.constraints = None

        self.cuts = dict.fromkeys(["stage{}".format(x) for x in range(self.n_stage - 1)])
        for i in range(0, self.n_stage - 1):
            if self.prob_name == "EnergyPlanning":
                self.cuts["stage{}".format(i)] = {"gradient": [0], "constant": [0]}
            elif self.prob_name == "ProductionPlanning":
                self.cuts["stage{}".format(i)] = {"gradient": [np.array([0, 0, 0])], "constant": [0]}
            elif self.prob_name == "MertonsPortfolioOptimization":
                self.cuts["stage{}".format(i)] = {"gradient": [np.array([0, 0])], "constant": [-100]}
            else:
                raise NotImplementedError

        self.cuts["stage{}".format(self.n_stage - 1)] = {"gradient": None, "constant": None}

        # Scenario related
        self.scenario_tree_node = 3
        self.scenario_trees, self.rv_mean, self.rv_std = self.prob_class.create_scenarioTree(
            num_node=self.scenario_tree_node,
            moment_matching=True)
        self.scenario = None

        # Forward pass related
        self.n_samples = n_samples

        # Criterion related
        self.alpha = stopping_criterion_alpha
        self.threshold = stopping_criterion_threshold

    def reset(self):
        self.objective_value = None
        self.value_func = None
        self.solution_set = None
        self.constraints = None

        self.cuts = dict.fromkeys(["stage{}".format(x) for x in range(self.n_stage - 1)])
        for i in range(0, self.n_stage - 1):
            if self.prob_name == "EnergyPlanning":
                self.cuts["stage{}".format(i)] = {"gradient": [0], "constant": [0]}
            elif self.prob_name == "ProductionPlanning":
                self.cuts["stage{}".format(i)] = {"gradient": [np.array([0, 0, 0])], "constant": [0]}
            elif self.prob_name == "MertonsPortfolioOptimization":
                self.cuts["stage{}".format(i)] = {"gradient": [np.array([0, 0])], "constant": [0]}
            else:
                raise NotImplementedError

        self.cuts["stage{}".format(self.n_stage - 1)] = {"gradient": None, "constant": None}

        # Scenario related
        self.scenario_tree_node = 5
        self.scenario_trees, self.rv_mean, self.rv_std = self.prob_class.create_scenarioTree(
            num_node=self.scenario_tree_node,
            moment_matching=False)
        self.scenario = None

    def generate_scenarios(self, scenario_trees):
        scenario = [random.choice(nodes) for nodes in scenario_trees]
        return scenario

    def one_iteration(self, do_backward_pass=True):
        upper_bound_params = self.forward_pass(n_samples=self.n_samples)
        if do_backward_pass:
            self.backward_pass(scenario_trees=self.scenario_trees)

        optimality_gap, done, upper_bound = self.check_stopping_criterion(upper_bound_params=upper_bound_params,
                                                                          alpha=self.alpha)

        return optimality_gap, done, upper_bound

    def forward_pass_one_sample(self, scenario):
        objective_value, value_function, solution_set = {}, {}, {}

        prev_solution = self.prob_class.prev_solution
        for stage_idx in range(self.n_stage):

            stage = self.stage(n_stages=self.n_stage, stage_number=stage_idx, prev_solution=prev_solution,
                               scenario=scenario[stage_idx], cuts=self.cuts["stage{}".format(stage_idx)])
            stage.solve()
            objective_value["stage{}".format(stage_idx)] = stage.objective_value
            value_function["stage{}".format(stage_idx)] = stage.value_func.value

            if self.prob_name == "EnergyPlanning":
                solution_set["stage{}".format(stage_idx)] = [stage.init_water.value, stage.final_water.value,
                                                             stage.hydro.value, stage.thermal.value]
                prev_solution = stage.final_water.value
            elif self.prob_name == "ProductionPlanning":
                solution_set["stage{}".format(stage_idx)] = [stage.production.value, stage.outsource.value,
                                                             stage.storage.value]
                prev_solution = stage.storage.value
            elif self.prob_name == "MertonsPortfolioOptimization":
                solution_set["stage{}".format(stage_idx)] = [stage.stock.value, stage.bond.value,
                                                             stage.consumption.value]
                prev_solution = np.array([stage.stock.value, stage.bond.value])
            else:
                raise NotImplementedError
        return objective_value, value_function, solution_set

    def forward_pass(self, n_samples=20):
        upper_bound_samples, obj_val_list, value_func_list, solution_list, scenario_list = [], [], [], [], []

        for i in range(n_samples):
            scenario = self.generate_scenarios(self.scenario_trees)
            scenario_list.append(scenario)

            objective_value, value_function, solution_set = self.forward_pass_one_sample(scenario=scenario)
            upper_bound_samples.append(
                sum(np.array(list(objective_value.values())) - np.array(list(value_function.values()))))
            obj_val_list.append(objective_value)
            value_func_list.append(value_function)
            solution_list.append(solution_set)

        idx = np.random.choice(n_samples)
        self.objective_value = obj_val_list[idx]
        self.value_func = value_func_list[idx]
        self.solution_set = solution_list[idx]
        self.scenario = scenario_list[idx]

        return np.mean(upper_bound_samples), np.std(upper_bound_samples, ddof=1) / np.sqrt(n_samples)

    def backward_pass(self, scenario_trees):
        for stage_idx in reversed(range(1, self.n_stage)):
            gradient_lst = []
            constant_lst = []
            for scenario in scenario_trees[stage_idx]:
                if self.prob_name == "EnergyPlanning":
                    prev_solution = self.solution_set["stage{}".format(stage_idx - 1)][1]
                elif self.prob_name == "ProductionPlanning":
                    prev_solution = self.solution_set["stage{}".format(stage_idx - 1)][2]
                elif self.prob_name == "MertonsPortfolioOptimization":
                    prev_solution = np.array([self.solution_set["stage{}".format(stage_idx - 1)][0],
                                              self.solution_set["stage{}".format(stage_idx - 1)][1]])
                else:
                    raise NotImplementedError

                stage = self.stage(n_stages=self.n_stage, stage_number=stage_idx,
                                   prev_solution=prev_solution,
                                   scenario=scenario, cuts=self.cuts["stage{}".format(stage_idx)])
                gradient, constant = stage.generate_benders_cut()
                gradient_lst.append(gradient)
                constant_lst.append(constant)

            self.cuts["stage{}".format(stage_idx - 1)]["gradient"].append(np.mean(gradient_lst, axis=0))
            self.cuts["stage{}".format(stage_idx - 1)]["constant"].append(np.mean(constant_lst))
        a = 0

    def get_lower_bound(self):
        stage_idx = 0
        stage = self.stage(n_stages=self.n_stage, stage_number=stage_idx,
                           prev_solution=self.prob_class.prev_solution,
                           scenario=self.scenario[stage_idx], cuts=self.cuts["stage{}".format(stage_idx)])
        stage.solve()
        a = stage.objective_value
        return stage.objective_value

    def check_stopping_criterion(self, upper_bound_params, alpha=0.05):
        ub, ub_se = upper_bound_params
        lb = self.get_lower_bound()
        z = st.norm.ppf(1 - alpha / 2)
        optimality_gap = ub + z * ub_se - lb
        return optimality_gap, optimality_gap < self.threshold, ub + z * ub_se
