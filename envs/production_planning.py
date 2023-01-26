import numpy as np
import cvxpy as cp
from scipy.stats import truncnorm
from envs.utils import *


__all__ = [
    "ProductionPlanning",
]


class ProductionPlanning:
    def __init__(self,
                 n_stages=7,
                 pdim=3,
                 max_production_resource=10,
                 production_cost=(1, 2, 5),
                 outsourcing_cost=(6, 12, 20),
                 storage_cost=(3, 7, 10),
                 demand_mean_low=(3, 1.5, 1),
                 demand_mean_high=(6, 4, 2),
                 demand_std_low=(0.2, 0.1, 0.05),
                 demand_std_high=(0.4, 0.2, 0.1),
                 random_demand=None,
                 paramdict=None,
                 ):
        if random_demand is None:
            random_demand = [(5, 3, 1), (6, 2, 1), (1, 2, 2)]
        self.demand_mean_low = demand_mean_low
        self.demand_mean_high = demand_mean_high
        self.demand_std_low = demand_std_low
        self.demand_std_high = demand_std_high

        self.prob_name = "ProductionPlanning"
        self.n_stages = n_stages
        self.pdim = pdim
        self.max_production_resource = max_production_resource
        self.production_cost = np.array(production_cost)
        self.outsourcing_cost = np.array(outsourcing_cost)
        self.storage_cost = np.array(storage_cost)
        self.random_demand = random_demand

        self.input_size = 9
        self.prev_solution = np.array([0, 0, 0])

        self.stage = ProductionPlanningStage

        self.paramdict = paramdict

    def create_scenarioTree(self, num_node=3, moment_matching=False):
        scenarioTree = [[(0, 0, 0)]]
        demand_mean_list = []
        demand_std_list = []
        demand_mean, demand_std = self.get_params()
        for idx in range(self.n_stages - 1):
            demand_mean_list.append(demand_mean)
            demand_std_list.append(demand_std)
            scenario = [] # pdim x num_node
            for i in range(self.pdim):
                lower = 0
                upper = 10
                mu = demand_mean[i]
                sigma = demand_std[i]
                batch_sample = truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=num_node)
                # batch_sample = np.random.normal(loc=demand_mean[i], scale=demand_std[i],
                #                                 size=num_node)
                # batch_sample = np.clip(batch_sample, 0, None)
                if moment_matching:
                    normalized_batch_sample = (batch_sample - np.mean(batch_sample))
                    normalized_batch_sample = normalized_batch_sample / np.std(batch_sample)
                    rescaled = normalized_batch_sample * sigma
                    rescaled = rescaled + mu
                    scenario.append(rescaled.tolist())
                else:
                    scenario.append(batch_sample.tolist())

            scenarioTree.append([tuple(np.array(scenario)[:, i]) for i in range(num_node)])

        # fix demand node
        # for idx in range(self.n_stages - 1):
        #     demand_mean_list.append([4, 2.3, 1.3])
        #     demand_std_list.append([0, 0, 0])
        #     scenarioTree.append(self.random_demand)
        return scenarioTree, demand_mean_list, demand_std_list

    def get_params(self):
        if self.paramdict:
            demand_mu = self.paramdict['mu']
            demand_std = self.paramdict['sigma']
        else:
            demand_mu = []
            demand_std = []
            for i in range(self.pdim):
                mu = np.random.uniform(self.demand_mean_low[i], self.demand_mean_high[i])
                std = np.random.uniform(self.demand_std_low[i], self.demand_std_high[i])
                demand_mu.append(mu)
                demand_std.append(std)
        # demand_mu = [4.5, 2.75, 1.5]
        # demand_std = [0.3, 0.15, 0.075]
        return demand_mu, demand_std


class ProductionPlanningStage(ProductionPlanning):
    def __init__(self, n_stages, stage_number, prev_solution, scenario, cuts):  # one of scenarios from scenario tree
        super(ProductionPlanningStage, self).__init__(n_stages=n_stages)

        assert n_stages == self.n_stages, "Error: different n_stage value"

        self.stage_number = stage_number
        self.prev_solution = prev_solution

        # Variables
        self.production = cp.Variable(shape=self.pdim, nonneg=True)
        self.outsource = cp.Variable(shape=self.pdim, nonneg=True)
        self.storage = cp.Variable(shape=self.pdim, nonneg=True)

        if self.stage_number == self.n_stages - 1:
            self.value_func = cp.Parameter(value=0)
        else:
            self.value_func = cp.Variable()

        # Parameters
        self.demand = scenario  # stage 0에서는 (0, 0, 0)

        A = np.array([[-1, 0, 0, -1, 0, 0, 1, 0, 0],
                      [0, -1, 0, 0, -1, 0, 0, 1, 0],
                      [0, 0, -1, 0, 0, -1, 0, 0, 1],
                      [1, 2, 5, 0, 0, 0, 0, 0, 0]], dtype=float)

        # Attributes
        self.constraints = [
            A[i, :self.pdim] @ self.production + A[i, self.pdim:2*self.pdim] @ self.outsource +
            A[i, 2*self.pdim:3*self.pdim] @ self.storage - self.prev_solution[i] + self.demand[i] == 0
            for i in range(3)
        ]
        self.constraints += [
            self.production_cost @ self.production <= self.max_production_resource
        ]

        self.cuts = []
        self.gradient = cuts["gradient"]
        self.constant = cuts["constant"]
        if self.gradient:
            if len(self.gradient) != len(self.constant):
                raise ValueError
            else:
                for i in range(len(self.gradient)):
                    self.cuts += [self.value_func >= self.gradient[i] @ self.storage + self.constant[i]]

        self.objective_value = None
        self.dual_vars = None

    def problem(self):
        if not self.gradient:  # Final Stage
            objective = cp.Minimize(self.outsourcing_cost @ self.outsource + self.value_func)
            constraints = self.constraints
        else:  # Except Final Stage
            objective = cp.Minimize(self.outsourcing_cost @ self.outsource + self.storage_cost @ self.storage + self.value_func)
            constraints = self.constraints + self.cuts
        prob = cp.Problem(objective, constraints)

        return prob

    def solve(self):
        prob = self.problem()
        prob.solve(solver=cp.CPLEX)

        self.objective_value = prob.value
        self.dual_vars = list(prob.solution.dual_vars.values())

        return self.objective_value, self.production.value, self.outsource.value, self.storage.value, self.value_func.value

    def generate_benders_cut(self):  # Calculate cut's gradient and constant
        self.solve()
        gradient = - np.array(self.dual_vars[:self.pdim])

        constant = self.objective_value - gradient @ self.prev_solution

        return gradient, constant

