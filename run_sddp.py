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

    if args.sample_type == "target":
        do_backward_pass = True
        precuts = []
    else:
        do_backward_pass = False
        load_path = 'D:/sddp_data/EnergyPlanning/stages_7/sample/{}'.format(args.sample_type)
        with open(os.path.join(load_path, "pred_cuts.pickle"), "rb") as fr:
            precuts = pickle.load(fr)

    # print(args.prob + ": SDDP algorithm start!")
    sddp = SDDP(prob_class=prob_class)
    solution_list, objective_list, value_function_list, cut_list, time_list, opt_gap_list, ub_list = [], [], [], [], [], [], []

    start = time.time()
    for idx in range(max_iter):
        temp = time.time()
        opt_gap, done, upper_bound = sddp.one_iteration(do_backward_pass=do_backward_pass)
        if not do_backward_pass:
            for i in range(n_stages - 1):
                sddp.cuts["stage{}".format(i)]["gradient"].append(precuts[i][idx+1][1] / (-precuts[i][idx+1][-5]))
                sddp.cuts["stage{}".format(i)]["constant"].append(precuts[i][idx + 1][-4] / (-precuts[i][idx + 1][-5]))

        solution_list.append(sddp.solution_set)
        objective_list.append(sddp.objective_value)
        # value_function_list.append(sddp.value_func)
        cut_list.append(copy.deepcopy(sddp.cuts))
        time_list.append(time.time() - temp)
        opt_gap_list.append(opt_gap)
        ub_list.append(upper_bound)

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

        # if args.save_mode == 'sample_scenario':
        #     break

    return solution_list, objective_list, cut_list, time_list, sddp, opt_gap_list, ub_list


def save_sample_data(solution, obj_value, cut_list, time_list=None, opt_gap_list=None, ub_list=None, sample_type='default', args=None):
    save_path = 'D:/sddp_data/{}/stages_7/sample/{}'.format(args.prob, sample_type)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "solution.pickle"), "wb") as fw:
        pickle.dump(solution, fw)

    with open(os.path.join(save_path, "objective.pickle"), "wb") as fw:
        pickle.dump(obj_value, fw)

    with open(os.path.join(save_path, "cut.pickle"), "wb") as fw:
        pickle.dump(cut_list, fw)

    if time_list:
        with open(os.path.join(save_path, "time.pickle"), "wb") as fw:
            pickle.dump(time_list, fw)
    if opt_gap_list:
        with open(os.path.join(save_path, "opt_gap.pickle"), "wb") as fw:
            pickle.dump(opt_gap_list, fw)
    if ub_list:
        with open(os.path.join(save_path, "ub.pickle"), "wb") as fw:
            pickle.dump(ub_list, fw)


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
    parser.add_argument('--sample_type', type=str, default="target", choices=["predict", "target"])
    parser.add_argument('--mm', type=str, default='False', choices=['True', 'False'])
    args = parser.parse_args()
    print("------------test stages: {}-------------".format(args.num_stages))
    print("------------test episode: {}-------------".format(args.max_episode))
    times = []
    for i in range(args.max_episode):
        solution, obj_value, cut_list, time_list, sddp, opt_gap_list, ub_list = run_SDDP(args)
        times.append(sum(time_list))
        # save_sample_data(solution, obj_value, cut_list, args.sample_type)
        if len(obj_value) >= 100:
            save_sample_data(solution, obj_value, cut_list, time_list, opt_gap_list, ub_list, "long_iteration", args)

        print("mean time: ", np.mean(times))
        # save_sample_data(solution, obj_value, cut_list, time_list, opt_gap_list, ub_list, "target", args)

        # if sddp is None:
        #     print("Episode {}/{} exceed max iteration".format(i + 1, args.max_episode))
        #     continue

        raw_data = {"cuts": sddp.cuts, "rv_mean": sddp.rv_mean, "rv_std": sddp.rv_std}
        data = preprocessing(raw_data, args)
        save_data(data, args)
        print("Process {}: Episode {}/{} result saved".format(process, i + 1, args.max_episode))


# SDDP solution
if __name__ == '__main__':
    # with Pool(4) as p:
    #     p.map(main, [1, 2, 3, 4])

    main(0)

