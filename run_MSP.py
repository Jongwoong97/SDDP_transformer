import cvxpy as cp
import numpy as np
import time

# Paramter setting ========================================================
h_cost = 2
t_cost = 7
utility_coeff = 0.1
utility_scale = 5
initial_reservoir = 40
demand = 20
# =========================================================================

def MSP_EP(stageNum=6, scenario_node=5, mm=True, paramdict={}):
    mean = paramdict['mean']
    scale = paramdict['scale']
    start = time.time()
    scenarios = [[]]
    for _ in range(stageNum - 1):
        batch_sample = np.random.normal(loc=mean, scale=scale, size=scenario_node)
        if mm:
            normalized_batch_sample = (batch_sample - np.mean(batch_sample))
            normalized_batch_sample = normalized_batch_sample/np.std(batch_sample)
            rescaled = normalized_batch_sample*scale
            rescaled = rescaled + mean
            scenarios.append(rescaled)
        else:
            scenarios.append(batch_sample)
    number_of_current_node = 1
    curIdx = 0
    totIndice = []
    stagewise_Indice = []
    for stage in range(stageNum):
        ind = list(range(curIdx, curIdx + number_of_current_node))
        totIndice += ind
        stagewise_Indice.append(ind)
        curIdx += number_of_current_node
        number_of_current_node = number_of_current_node*scenario_node
    stagewise_ind_length = [len(item) for item in stagewise_Indice]

    def find_stage(node: int):
        if node == 0:
            return 0
        stage = 1
        while True:
            if node <= sum([item for stagei, item in enumerate(stagewise_ind_length) if stagei <=stage]) - 1:
                return stage
            stage += 1

    def retrieve_parent_index(node: int):
        cur_Stage = find_stage(node)
        if cur_Stage < 1:
            raise Exception('0 stage has no parent node')
        modulo = stagewise_Indice[cur_Stage].index(node)
        quotient = modulo // scenario_node
        remainder = modulo % scenario_node
        return stagewise_Indice[cur_Stage-1][quotient], remainder

    # STAGE 0
    water_init_0 = cp.Variable(nonneg=True)
    water_final_0 = cp.Variable(nonneg=True)
    hydro_0 = cp.Variable(nonneg=True)
    thermal_0 = cp.Variable(nonneg=True)
    totVars = [(water_init_0, water_final_0, hydro_0, thermal_0)]

    objective = hydro_0*h_cost + thermal_0*t_cost + cp.exp(-utility_coeff*water_final_0 + utility_scale)
    constraints = []
    constraints += [water_init_0 == initial_reservoir]  # Initial condition
    constraints += [water_final_0 == water_init_0 - hydro_0]  # water level balance
    constraints += [hydro_0 + thermal_0 >= demand]  # stage 0 demand
    # for idx, var in enumerate([water_init_0, water_final_0, hydro_0, thermal_0]):
    #     constraints += [var >= 0]  # Non-negativity

    for stage, indSet in enumerate(stagewise_Indice):
        node_probability = 1/len(indSet)
        # Stage t problem
        if stage > 0:
            for nodeIdx in indSet:
                water_init_t = cp.Variable(nonneg=True)
                water_final_t = cp.Variable(nonneg=True)
                hydro_t = cp.Variable(nonneg=True)
                thermal_t = cp.Variable(nonneg=True)
                totVars.append((water_init_t, water_final_t, hydro_t, thermal_t))

                objective += node_probability*(hydro_t*h_cost + thermal_t*t_cost + cp.exp(-utility_coeff*water_final_t + utility_scale))
                parentIdx, scenIdx = retrieve_parent_index(nodeIdx)
                prevVar = totVars[parentIdx]
                R = scenarios[stage][scenIdx]
                # Initial condition
                constraints += [water_init_t == prevVar[1] + R]  # Initial water level
                constraints += [water_final_t == water_init_t - hydro_t]  # water level balance
                constraints += [hydro_t + thermal_t >= demand]  # stage 0 demand
                # for idx, var in enumerate([water_init_t, water_final_t, hydro_t, thermal_t]):
                #     constraints += [var >= 0]  # Non-negativity

    problem = cp.Problem(cp.Minimize(objective), constraints)
    # print('Problem Defined. Now Solving...')
    try:
        problem.solve(verbose=False, solver=cp.MOSEK)
    except:
        problem.solve(verbose=False, solver=cp.ECOS)
    optStage0 = [float(item.value) for item in totVars[0]]
    optStage0_n = [str(item) for item in totVars[0]]
    pt = [item + ': ' + str(optStage0[idx]) + ',' for idx, item in enumerate(optStage0_n)]
    # print('===========================================================================')
    # print('optimalObj: ', problem.value)
    # print('optimalVal: ', pt)
    # print('Time:', time.time() - start)
    # print('===========================================================================')
    optStage0 = [item.value for item in totVars[0]]
    return optStage0, problem.value


if __name__ == '__main__':
    MSP_EP(stageNum=7, scenario_node=3, paramdict={'mean': 20, 'scale': 5}, mm=True)
