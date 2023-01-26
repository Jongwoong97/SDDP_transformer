import pickle
import os.path
import numpy as np
from envs import utils


def preprocessing(data, args):
    A, B, b, c = utils.get_parameters(args)

    _A = np.concatenate((A, np.zeros_like(A)), axis=1)
    _B = np.concatenate((B, np.zeros_like(B)), axis=1)
    _b = np.concatenate((b, np.zeros_like(b)), axis=1)
    _c = np.tile(np.concatenate((c, np.zeros_like(c)), axis=0), (_A.shape[0], 1))

    feature_all = []
    label_all = []
    for stage in range(args.num_stages-1):

        # features
        if args.prob == "ProductionPlanning":
            b_mean = np.array(data["rv_mean"][stage] + [10]).reshape(_A.shape[0], 1)
            b_std = np.array(data["rv_std"][stage] + [0]).reshape(_A.shape[0], 1)
            feature = np.concatenate((_A, _B, _c, b_mean, b_std, np.tile((1 + stage) / (args.num_stages - 1), (_A.shape[0], 1))), axis=1)
        elif args.prob == "EnergyPlanning":
            b_mean = np.array([data["rv_mean"][stage]] + [0, 20]).reshape(_A.shape[0], 1)
            b_std = np.array([data["rv_std"][stage]] + [0, 0]).reshape(_A.shape[0], 1)
            feature = np.concatenate((_A, _B, _c, b_mean, b_std, np.tile((1 + stage) / (args.num_stages - 1), (_A.shape[0], 1))), axis=1)
        elif args.prob == "MertonsPortfolioOptimization":
            _B[0][0] = data["rv_mean"][stage]
            _B[0][4] = data["rv_std"][stage]
            feature = np.concatenate((_A, _B, _b, _c, np.tile((1 + stage) / (args.num_stages - 1), (_A.shape[0], 1))), axis=1)
        else:
            raise NotImplementedError
        feature_all.append(feature)

        # labels
        gradients = np.array(data["cuts"]["stage{}".format(stage)]["gradient"])
        constants = np.array(data["cuts"]["stage{}".format(stage)]["constant"])
        label = np.zeros((gradients.shape[0], A.shape[1]+2))
        if args.prob == "ProductionPlanning":
            pdim = 3
            label[:, 2*pdim:3*pdim] = gradients
        elif args.prob == "EnergyPlanning":
            label[:, 1] = gradients
        elif args.prob == "MertonsPortfolioOptimization":
            label[:, :2] = gradients
        label[:, -2] = -1
        label[:, -1] = constants

        # start/end token one_hot_encoding
        token_arr = np.ones(label.shape[0])
        token_arr[0] = 0
        token_arr[-1] = 2

        encoding = np.eye(3)[token_arr.astype(int)]
        label = np.concatenate((label, encoding), axis=1)
        label_all.append(label)

    return {"features": feature_all, "labels": label_all}


def save_data(data, args):
    new_data = {}
    list_path = [args.save_path, args.prob, "stages_{}".format(args.num_stages), args.save_mode]

    if args.save_mode != 'sample_scenario':
        if args.mm == 'True':
            list_path.append("mm")
        list_path.append("original")

    save_path = os.path.join(*list_path)
    os.makedirs(save_path, exist_ok=True)

    for key, value in data.items():
        if os.path.exists(os.path.join(save_path, key + ".pickle")):
            with open(os.path.join(save_path,  key + ".pickle"), "rb") as fr:
                prev_data = pickle.load(fr)
            prev_data += value
            new_data[key] = prev_data
        else:
            new_data[key] = value

    for key, value in new_data.items():
        with open(os.path.join(save_path, key + ".pickle"), "wb") as fw:
            pickle.dump(value, fw)

