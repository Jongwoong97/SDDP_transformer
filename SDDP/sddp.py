import random
import numpy as np
import scipy.stats as st


__all__ = [
    "SDDP"
]


class SDDP:
    def __init__(self, prob_class,
                 level_1_dominance=False,
                 n_samples=20,
                 stopping_criterion_alpha=0.05,
                 stopping_criterion_threshold=0.1):

        self.prob_class = prob_class
        self.prob_name = self.prob_class.prob_name
        self.n_stage = self.prob_class.n_stages
        self.activate_dominance = level_1_dominance

        # SDDP related
        self.stage = self.prob_class.stage

        self.objective_value = None
        self.value_func = None
        self.solution_set = None
        self.constraints = None
        self.all_solution_set = dict.fromkeys(["stage{}".format(x) for x in range(self.n_stage - 1)])
        for key in self.all_solution_set.keys():
            self.all_solution_set[key] = []

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
        if self.stage == 7:
            self.scenario_tree_node = 5
        else:
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

    def generate_scenarios(self, scenario_trees):
        scenario = [random.choice(nodes) for nodes in scenario_trees]
        return scenario

    def level_1_dominance(self, stage_idx):
        selected_cuts_idx = set()
        selected_cuts = {"gradient": [], "constant": []}
        candidate_cuts = self.cuts["stage{}".format(stage_idx)]
        if stage_idx == self.n_stage - 1:
            return candidate_cuts
        else:
            if self.all_solution_set["stage0"]:
                for j in range(len(self.all_solution_set["stage{}".format(stage_idx)])):
                    max_val = -10000
                    max_idx = -1
                    if self.prob_name == "EnergyPlanning":
                        x = self.all_solution_set["stage{}".format(stage_idx)][j][1]
                        for i in range(len(candidate_cuts["gradient"])):
                            gradient = candidate_cuts["gradient"][i]
                            constant = candidate_cuts["constant"][i]
                            if (gradient * x + constant) > max_val:
                                max_val = (gradient * x + constant)
                                max_idx = i
                    elif self.prob_name == "MertonsPortfolioOptimization":
                        x = self.all_solution_set["stage{}".format(stage_idx)][j][0:2]
                        for i in range(len(candidate_cuts["gradient"])):
                            gradient = candidate_cuts["gradient"][i]
                            constant = candidate_cuts["constant"][i]
                            if (gradient @ x + constant) > max_val:
                                max_val = (gradient @ x + constant)
                                max_idx = i
                    elif self.prob_name == "ProductionPlanning":
                        x = self.all_solution_set["stage{}".format(stage_idx)][j][2]
                        for i in range(len(candidate_cuts["gradient"])):
                            gradient = candidate_cuts["gradient"][i]
                            constant = candidate_cuts["constant"][i]
                            if (gradient @ x + constant) > max_val:
                                max_val = (gradient @ x + constant)
                                max_idx = i
                    selected_cuts_idx.add(max_idx)
            else:
                return candidate_cuts
            for idx in list(selected_cuts_idx):
                selected_cuts["gradient"].append(candidate_cuts["gradient"][idx])
                selected_cuts["constant"].append(candidate_cuts["constant"][idx])
            return selected_cuts

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
            if self.activate_dominance:
                cuts = self.level_1_dominance(stage_idx)
                stage = self.stage(n_stages=self.n_stage, stage_number=stage_idx, prev_solution=prev_solution,
                                   scenario=scenario[stage_idx], cuts=cuts)
            else:
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
        for key in self.all_solution_set.keys():
            self.all_solution_set[key].append(solution_list[idx][key])

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

                if self.activate_dominance:
                    cuts = self.level_1_dominance(stage_idx)
                    stage = self.stage(n_stages=self.n_stage, stage_number=stage_idx,
                                       prev_solution=prev_solution,
                                       scenario=scenario, cuts=cuts)
                else:
                    stage = self.stage(n_stages=self.n_stage, stage_number=stage_idx,
                                       prev_solution=prev_solution,
                                       scenario=scenario, cuts=self.cuts["stage{}".format(stage_idx)])
                gradient, constant = stage.generate_benders_cut()
                gradient_lst.append(gradient)
                constant_lst.append(constant)

            self.cuts["stage{}".format(stage_idx - 1)]["gradient"].append(np.mean(gradient_lst, axis=0))
            self.cuts["stage{}".format(stage_idx - 1)]["constant"].append(np.mean(constant_lst))

    def get_lower_bound(self):
        stage_idx = 0
        stage = self.stage(n_stages=self.n_stage, stage_number=stage_idx,
                           prev_solution=self.prob_class.prev_solution,
                           scenario=self.scenario[stage_idx], cuts=self.cuts["stage{}".format(stage_idx)])
        stage.solve()
        return stage.objective_value

    def check_stopping_criterion(self, upper_bound_params, alpha=0.05):
        ub, ub_se = upper_bound_params
        lb = self.get_lower_bound()
        z = st.norm.ppf(1 - alpha / 2)
        optimality_gap = ub + z * ub_se - lb
        return optimality_gap, optimality_gap < self.threshold, ub + z * ub_se
