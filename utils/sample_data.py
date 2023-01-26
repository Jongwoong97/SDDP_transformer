import os.path
from envs import utils
import pickle
import numpy as np
from visualization.attention_score_graph import plot_head_map


def get_sample_data(args, sample_mean, sample_std):
    A, B, b, c = utils.get_parameters(args)

    _A = np.concatenate((A, np.zeros_like(A)), axis=1)
    _B = np.concatenate((B, np.zeros_like(B)), axis=1)
    _b = np.concatenate((b, np.zeros_like(b)), axis=1)
    _c = np.tile(np.concatenate((c, np.zeros_like(c)), axis=0), (_A.shape[0], 1))

    if type(sample_mean) is list:
        rv_mean = [sample_mean] * (args.num_stages - 1)
    else:
        rv_mean = [[sample_mean]] * (args.num_stages - 1)

    if type(sample_std) is list:
        rv_std = [sample_std] * (args.num_stages - 1)
    else:
        rv_std = [[sample_std]] * (args.num_stages - 1)

    if args.prob == "MertonsPortfolioOptimization":
        initial_cut_constant = -100
    else:
        initial_cut_constant = 0

    # initial_cut = np.concatenate((np.zeros((1, A.shape[1]), dtype=float), np.array([[-1, initial_cut_constant, 1, 0, 0]], dtype=float)), axis=1)
    initial_cut = np.concatenate((np.zeros((1, A.shape[1]), dtype=float), np.array([[-1, initial_cut_constant, 1]], dtype=float)), axis=1)
    feature_all = []
    label_all = [initial_cut]*(args.num_stages - 1)
    for stage in range(args.num_stages-1):
        stage_information = np.tile((1 + stage) / (args.num_stages - 1), (_A.shape[0], 1))

        # features
        if args.prob == "ProductionPlanning":
            b_mean = np.array(rv_mean[stage] + [10]).reshape(_A.shape[0], 1)
            b_std = np.array(rv_std[stage] + [0]).reshape(_A.shape[0], 1)
            if args.feature_type == "objective_information":
                feature = np.concatenate((_A, _B, _c, b_mean, b_std, stage_information), axis=1)
            else:
                feature = np.concatenate((_A, _B, b_mean, b_std, stage_information), axis=1)
        elif args.prob == "EnergyPlanning":
            b_mean = np.array(rv_mean[stage] + [0, 20]).reshape(_A.shape[0], 1)
            b_std = np.array(rv_std[stage] + [0, 0]).reshape(_A.shape[0], 1)
            if args.feature_type == "objective_information":
                feature = np.concatenate((_A, _B, _c, b_mean, b_std, stage_information), axis=1)
            else:
                feature = np.concatenate((_A, _B, b_mean, b_std, stage_information), axis=1)
        elif args.prob == "MertonsPortfolioOptimization":
            _B[0][0] = rv_mean[stage][0]
            _B[0][4] = rv_std[stage][0]
            if args.feature_type == "objective_information":
                feature = np.concatenate((_A, _B, _b, _c, stage_information), axis=1)
            else:
                feature = np.concatenate((_A, _B, _b, stage_information), axis=1)
        else:
            raise NotImplementedError
        feature_all.append(feature)

    return feature_all, label_all


def get_max_cut_cnt_from_target(problem, n_stages, sample_type):
    load_path = 'D:/sddp_data/{}/stages_{}/sample/{}'.format(problem, n_stages, sample_type)

    with open(os.path.join(load_path, "solution.pickle"), "rb") as f:
        solution = pickle.load(f)

    with open(os.path.join(load_path, "cut.pickle"), "rb") as f:
        cut = pickle.load(f)

    print(cut[0]['stage0']['gradient'] == cut[1]['stage0']['gradient'])
    print(solution[0]['stage0'] == solution[1]['stage0'])

    num_cuts = len(cut[-1]["stage0"]["gradient"])
    num_max_cut_dict = {key: np.zeros((len(cut), num_cuts)) for key in list(cut[0].keys())[:-1]}

    for it in range(len(cut)):
        for stage in num_max_cut_dict.keys():
            for sol in solution[:it+1]:
                if problem == "EnergyPlanning":
                    x = sol[stage][1]
                elif problem == "MertonsPortfolioOptimization":
                    x = np.array((sol[stage][0], sol[stage][1]))
                else:
                    raise NotImplementedError
                max_val = -100
                idx = -1
                for i in range(len(cut[it][stage]['gradient'])):
                    if problem == "EnergyPlanning":
                        tmp = cut[it][stage]["gradient"][i] * x + cut[it][stage]["constant"][i]
                    elif problem == "MertonsPortfolioOptimization":
                        tmp = cut[it][stage]["gradient"][i] @ x + cut[it][stage]["constant"][i]
                    else:
                        raise NotImplementedError
                    if tmp > max_val:
                        idx = i
                        max_val = tmp
                num_max_cut_dict[stage][it, idx] += 1

    max_cut_cnts = num_max_cut_dict['stage0'] / np.sum(num_max_cut_dict['stage0'], axis=1, keepdims=True)

    initial_cut = np.zeros((1, max_cut_cnts.shape[1]))
    initial_cut[0, 0] = 1
    max_cut_cnts = np.concatenate((initial_cut, max_cut_cnts), axis=0)
    plot_head_map(mma=max_cut_cnts,
                  source_labels=list(map(str, np.arange(max_cut_cnts.shape[1]))),
                  target_labels=list(map(str, np.arange(max_cut_cnts.shape[0]))))

    return max_cut_cnts


def get_max_cut_cnt_from_prediction(pred_cuts, problem, n_stages):
    load_path = 'D:/sddp_data/{}/stages_{}/sample/target'.format(problem, n_stages)

    with open(os.path.join(load_path, "solution.pickle"), "rb") as f:
        solution = pickle.load(f)

    print(solution[0]['stage0'] == solution[1]['stage0'])

    stage = 'stage0'

    num_cuts = len(pred_cuts)
    num_max_cut_dict = np.zeros((num_cuts, num_cuts))

    for it in range(num_cuts):
        for sol in solution[:it + 1]:
            x = sol[stage][1]
            max_val = -100
            idx = -1
            for i in range(it+1):
                tmp = pred_cuts[i][1] * x / (-pred_cuts[i][-5]) + pred_cuts[i][-4] / (-pred_cuts[i][-5])
                if tmp >= max_val:
                    idx = i
                    max_val = tmp
            num_max_cut_dict[it, idx] += 1

    max_cut_cnts = num_max_cut_dict / np.sum(num_max_cut_dict, axis=1, keepdims=True)

    plot_head_map(mma=max_cut_cnts,
                  source_labels=list(map(str, np.arange(max_cut_cnts.shape[1]))),
                  target_labels=list(map(str, np.arange(max_cut_cnts.shape[0]))))

    return max_cut_cnts


if __name__ == '__main__':
    get_max_cut_cnt_from_target("MertonsPortfolioOptimization", 7, "target")
    # get_max_cut_cnt_from_target("EnergyPlanning", 7, "predict")