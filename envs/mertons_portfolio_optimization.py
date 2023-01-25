import numpy as np
import cvxpy as cp
from config import *

__all__ = [
    "MertonsPortfolioOptimization"
]


class MertonsPortfolioOptimization(object):
    def __init__(self,
                 n_stages,
                 risk_free_rate_of_bond=0.03,
                 scaling_coeff_of_bequest_utility=1,
                 mean_return=0.06,
                 mean_return_low=0.04,
                 mean_return_high=0.08,
                 volatility=0.2,
                 volatility_low=0.15,
                 volatility_high=0.25,
                 paramdict=None,
                 ):
        self.prob_name = "MertonsPortfolioOptimization"
        self.n_stages = n_stages
        self.risk_free_rate_of_bond = risk_free_rate_of_bond
        self.scaling_coeff_of_bequest_utility = scaling_coeff_of_bequest_utility
        self.mean_return = mean_return
        self.mean_return_low = mean_return_low
        self.mean_return_high = mean_return_high
        self.volatility = volatility
        self.volatility_low = volatility_low
        self.volatility_high = volatility_high
        self.delta_t = 1 / (n_stages-1)
        self.prev_solution = np.array([0, 100/(1+self.risk_free_rate_of_bond * self.delta_t)])
        self.change_param = True

        self.stage = MertonsPortfolioOptimizationStage

        self.paramdict = paramdict

    def create_scenarioTree(self, num_node=3, moment_matching=True):
        scenarioTree = [[0]]
        lognormal_mean = []
        lognormal_std = []
        mu, std = self.get_params()
        param_a, param_b = (mu-(std**2)/2)*self.delta_t, std*np.sqrt(self.delta_t)
        for idx in range(self.n_stages - 1):
            lognormal_mean.append(param_a)
            lognormal_std.append(param_b)
            batch_sample = np.random.normal(loc=param_a, scale=param_b, size=num_node)
            if moment_matching:
                normalized_batch_sample = (batch_sample - np.mean(batch_sample))
                normalized_batch_sample = normalized_batch_sample / np.std(batch_sample)
                rescaled = normalized_batch_sample * param_b
                rescaled = rescaled + param_a
                scenarioTree.append(np.exp(rescaled).tolist())
            else:
                scenarioTree.append(np.exp(batch_sample).tolist())
        return scenarioTree, lognormal_mean, lognormal_std

    def get_params(self):
        if self.paramdict:
            mu, std = self.paramdict['mu'], self.paramdict['sigma']
        elif self.change_param:
            mu = np.random.uniform(self.mean_return_low, self.mean_return_high)
            std = np.random.uniform(self.volatility_low, self.volatility_high)
        else:
            mu = self.mean_return
            std = self.volatility
        return mu, std


class MertonsPortfolioOptimizationStage(MertonsPortfolioOptimization):
    def __init__(self, n_stages, stage_number, prev_solution, scenario, cuts):  # one of scenarios from scenario tree
        super(MertonsPortfolioOptimizationStage, self).__init__(n_stages=n_stages)

        assert n_stages == self.n_stages, "Error: different n_stage value"

        self.stage_number = stage_number
        self.prev_solution = prev_solution

        # Variables
        self.stock = cp.Variable(nonneg=True)
        self.bond = cp.Variable(nonneg=True)
        self.consumption = cp.Variable(nonneg=True)
        self.wealth = cp.Variable(nonneg=True)

        if self.stage_number == self.n_stages - 1:
            self.value_func = cp.Parameter(value=0)
        else:
            self.value_func = cp.Variable()

        # Parameters
        self.stock_return = cp.Parameter(value=scenario)  # stage 0에서는 0

        # Attributes
        if self.stage_number == self.n_stages - 1:
            self.constraints = [
                self.wealth + self.consumption == (1 + self.risk_free_rate_of_bond * self.delta_t) * self.prev_solution[1] + self.stock_return * self.prev_solution[0],
            ]
        else:
            self.constraints = [
                self.wealth == (1+self.risk_free_rate_of_bond * self.delta_t) * self.prev_solution[1] + self.stock_return * self.prev_solution[0],
                self.stock + self.bond + self.consumption == self.wealth
            ]

        self.cuts = []
        self.gradient = cuts["gradient"]
        self.constant = cuts["constant"]
        if self.gradient:
            if len(self.gradient) != len(self.constant):
                raise ValueError
            else:
                for i in range(len(self.gradient)):
                    self.cuts += [self.value_func >= self.gradient[i][0] * self.stock + self.gradient[i][1] * self.bond + self.constant[i]]

        self.objective_value = None
        self.dual_vars = None

    def problem(self):
        if utility_risk_aversion_coeff == 1:
            if self.stage_number == self.n_stages - 1:
                objective = cp.Minimize(-BigM * (cp.log(self.wealth + smallm) + cp.log(self.consumption + smallm)) + self.value_func)
            else:
                objective = cp.Minimize(-BigM * cp.log(self.consumption + smallm) + self.value_func)
        else:
            if self.stage_number == self.n_stages - 1:
                objective = cp.Minimize(-1 * (1 / (1 - utility_risk_aversion_coeff) * cp.power(self.wealth, 1 - utility_risk_aversion_coeff)) + self.value_func)
            else:
                objective = cp.Minimize(-1 * (1 / (1 - utility_risk_aversion_coeff) * cp.power(self.consumption, 1 - utility_risk_aversion_coeff)) + self.value_func)

        if not self.gradient:  # Final Stage
            constraints = self.constraints
        else:  # Except Final Stage
            constraints = self.constraints + self.cuts
        prob = cp.Problem(objective, constraints)

        return prob

    def solve(self):
        prob = self.problem()
        prob.solve(solver=cp.MOSEK)
        # Int 범위 제한 관련 TypeError 발생 시, dual_var id 설정
        self.objective_value = prob.value
        self.dual_vars = list(prob.solution.dual_vars.values())

        return self.objective_value, self.stock.value, self.bond.value, self.consumption.value, self.wealth.value, self.value_func.value

    def generate_benders_cut(self):  # Calculate cut's gradient and constant
        self.solve()
        gradient = - self.dual_vars[0] * np.array([self.stock_return.value, 1 + self.risk_free_rate_of_bond * self.delta_t])
        constant = self.objective_value - gradient[0] * self.prev_solution[0] - gradient[1] * self.prev_solution[1]

        return gradient, constant
