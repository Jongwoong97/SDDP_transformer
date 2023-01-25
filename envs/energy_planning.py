import numpy as np
import cvxpy as cp

__all__ = [
    "EnergyPlanning"
]


class EnergyPlanning(object):
    def __init__(self,
                 n_stages,
                 initial_reservoir=40,
                 demand=20,
                 water_flow_mean=20,
                 water_flow_std=5,
                 water_inflow_mean_low=15,
                 water_inflow_mean_high=25,
                 water_inflow_std_low=4,
                 water_inflow_std_high=6,
                 utility_coeff=0.1,
                 utility_scale=5,
                 hydro_cost=2,
                 thermal_cost=7,
                 paramdict=None,
                 ):
        self.prob_name = "EnergyPlanning"
        self.n_stages = n_stages
        self.initial_reservoir = initial_reservoir
        self.demand = demand
        self.water_inflow_mean = water_flow_mean
        self.water_inflow_scale = water_flow_std
        self.water_inflow_mean_low = water_inflow_mean_low
        self.water_inflow_mean_high = water_inflow_mean_high
        self.water_inflow_std_low = water_inflow_std_low
        self.water_inflow_std_high = water_inflow_std_high
        self.utility_coeff = utility_coeff
        self.utility_scale = utility_scale
        self.hydro_cost = hydro_cost
        self.thermal_cost = thermal_cost

        self.input_size = 6
        self.prev_solution = initial_reservoir

        self.stage = EnergyPlanningStage

        self.paramdict = paramdict

    def create_scenarioTree(self, num_node=3, moment_matching=True):
        scenarioTree = [[0]]
        water_inflow_mean = []
        water_inflow_std = []
        mu, std = self.get_params()
        for idx in range(self.n_stages - 1):
            water_inflow_mean.append(mu)
            water_inflow_std.append(std)
            batch_sample = np.random.normal(loc=mu, scale=std,
                                            size=num_node)
            if moment_matching:
                normalized_batch_sample = (batch_sample - np.mean(batch_sample))
                normalized_batch_sample = normalized_batch_sample / np.std(batch_sample)
                rescaled = normalized_batch_sample * std
                rescaled = rescaled + mu
                scenarioTree.append(rescaled.tolist())
            else:
                scenarioTree.append(batch_sample.tolist())
        return scenarioTree, water_inflow_mean, water_inflow_std

    def get_params(self):
        if self.paramdict:
            mu = self.paramdict['mean']
            std = self.paramdict['scale']
        else:
            mu = np.random.uniform(self.water_inflow_mean_low, self.water_inflow_mean_high)
            std = np.random.uniform(self.water_inflow_std_low, self.water_inflow_std_high)
        # mu = 20
        # std = 5
        return mu, std


class EnergyPlanningStage(EnergyPlanning):
    def __init__(self, n_stages, stage_number, prev_solution, scenario, cuts):  # one of scenarios from scenario tree
        super(EnergyPlanningStage, self).__init__(n_stages=n_stages)

        assert n_stages == self.n_stages, "Error: different n_stage value"

        self.stage_number = stage_number
        self.prev_solution = prev_solution

        # Variables
        self.init_water = cp.Variable(nonneg=True)
        self.final_water = cp.Variable(nonneg=True)  # stage 0에서는 self.initial_reservoir
        self.hydro = cp.Variable(nonneg=True)
        self.thermal = cp.Variable(nonneg=True)
        if self.stage_number == self.n_stages - 1:
            self.value_func = cp.Parameter(value=0)
        else:
            self.value_func = cp.Variable()

        # Parameters
        self.water_inflow = cp.Parameter(value=scenario)  # stage 0에서는 0

        # Attributes
        self.constraints = [
            self.init_water == self.prev_solution + self.water_inflow,
            self.final_water == self.init_water - self.hydro,
            self.hydro + self.thermal >= self.demand
        ]
        self.constraints_mtx = np.array([[1, 0, 0, 0, 0, -(self.water_inflow.value + self.prev_solution)],
                                        [-1, 1, 1, 0, 0, 0],
                                        [0, 0, -1, -1, 0, self.demand]], dtype=np.float32)

        self.cuts = []
        self.gradient = cuts["gradient"]
        self.constant = cuts["constant"]
        if self.gradient:
            if len(self.gradient) != len(self.constant):
                raise ValueError
            else:
                self.cuts_mtx = np.zeros((len(self.gradient), 6), dtype=np.float32)
                for i in range(len(self.gradient)):
                    self.cuts += [self.value_func >= self.gradient[i] * self.final_water + self.constant[i]]
                    self.cuts_mtx[i, 1] = self.gradient[i]
                    self.cuts_mtx[i, 4] = -1
                    self.cuts_mtx[i, 5] = self.constant[i]

        self.objective_value = None
        self.dual_vars = None

    def problem(self):
        objective = cp.Minimize(self.hydro * self.hydro_cost + self.thermal * self.thermal_cost
                                + cp.exp(- self.utility_coeff * self.final_water + self.utility_scale)
                                + self.value_func)

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

        return self.objective_value, self.init_water.value, self.final_water.value, self.hydro.value, \
               self.thermal.value, self.value_func.value

    def generate_benders_cut(self):  # Calculate cut's gradient and constant
        self.solve()
        gradient = - self.dual_vars[0]
        constant = self.objective_value - gradient * self.prev_solution

        return gradient, constant
