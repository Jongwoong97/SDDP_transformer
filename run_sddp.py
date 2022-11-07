import time
import random
import numpy as np
import pickle
import copy
from envs.energy_planning import *
from envs.production_planning import *
from envs.inventory_control import *
from envs.mertons_portfolio_optimization import *
from SDDP.sddp import *
import argparse
from data.data_processing import *
from multiprocessing import Pool


def run_SDDP(args):
    max_iter = args.max_iter
    n_stages = args.num_stages
    problem = args.prob
    if problem == 'EnergyPlanning':
        prob_class = EnergyPlanning(n_stages=n_stages)
    elif problem == 'ProductionPlanning':
        prob_class = ProductionPlanning(n_stages=n_stages)
    elif problem == 'InventoryControl':
        prob_class = InventoryControl(n_stages=n_stages)
    elif problem == 'MertonsPortfolioOptimization':
        prob_class = MertonsPortfolioOptimization(n_stages=n_stages)
    else:
        raise ValueError

    assert n_stages >= 2

    # print(args.prob + ": SDDP algorithm start!")
    sddp = SDDP(prob_class=prob_class)
    solution_list, objective_list, value_function_list, cut_list = [], [], [], []

    start = time.time()
    for idx in range(max_iter):
        opt_gap, done, _ = sddp.one_iteration()
        # solution_list.append(sddp.solution_set)
        # objective_list.append(sddp.objective_value)
        # value_function_list.append(sddp.value_func)
        # cut_list.append(copy.deepcopy(sddp.cuts))

        # print(f"\nObjective value: {sddp.objective_value['stage0']}")
        # print(f"Solution: {np.around(np.hstack(sddp.solution_set[f'stage{0}']), 3)}")
        # print(f"Solution(stage1): {np.around(np.hstack(sddp.solution_set[f'stage{1}']), 3)}")
        # print(f"Optimality gap: {opt_gap}")
        # print(f"total running time with iter {idx}: {time.time() - start}")
        # print(f"# of 1st stage cuts: {len(sddp.cuts['stage0']['gradient']) - 1}")

        # if idx > 80:
        #     print("---" * 20)
        #     return solution_list[-1]["stage0"], objective_list[-1]["stage0"], time.time() - start, None

        if done:
            print("---" * 20)
            break

    return solution_list, objective_list, cut_list, time.time() - start, sddp


def save_sample_data(solution, obj_value, cut_list):
    save_path = 'D:/sddp_data/EnergyPlanning/stages_7/sample'
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "solution.pickle"), "wb") as fw:
        pickle.dump(solution, fw)

    with open(os.path.join(save_path, "objective.pickle"), "wb") as fw:
        pickle.dump(obj_value, fw)

    with open(os.path.join(save_path, "cut.pickle"), "wb") as fw:
        pickle.dump(cut_list, fw)


def main(process):
    parser = argparse.ArgumentParser(description='Pytorch SDDP_RL')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='problem to solve')
    parser.add_argument('--prob', type=str, default='ProductionPlanning',
                        help='problem to solve')
    parser.add_argument('--num_stages', type=int, default=7,
                        help='Number of Stages')
    parser.add_argument('--max_episode', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='D:/sddp_data')
    parser.add_argument('--save_mode', type=str, default='train')
    args = parser.parse_args()
    print("------------test stages: {}-------------".format(args.num_stages))
    print("------------test episode: {}-------------".format(args.max_episode))
    for i in range(args.max_episode):
        solution, obj_value, cut_list, running_time, sddp = run_SDDP(args)

        # save_sample_data(solution, obj_value, cut_list)

        # if sddp is None:
        #     print("Episode {}/{} exceed max iteration".format(i + 1, args.max_episode))
        #     continue
        raw_data = {"cuts": sddp.cuts, "rv_mean": sddp.rv_mean, "rv_std": sddp.rv_std}
        data = preprocessing(raw_data, args)
        #
        #
        save_data(data, args)
        print("Process {}: Episode {}/{} result saved".format(process, i + 1, args.max_episode))


# SDDP solution
if __name__ == '__main__':
    with Pool(4) as p:
        p.map(main, [1, 2, 3, 4])

