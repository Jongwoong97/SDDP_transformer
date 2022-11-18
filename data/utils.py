import numpy as np
import pickle
import os.path

def change_start_end_token():
    if os.path.exists("D:/sddp_data/ProductionPlanning/stages_7/uncertainty_3/{}.pickle".format("labels")):
        with open("D:/sddp_data/ProductionPlanning/stages_7/uncertainty_3/{}.pickle".format("labels"), "rb") as fr:
            data = pickle.load(fr)
    else:
        raise FileNotFoundError

    new_data = []
    for d in data:
        print("origin shape: ", d.shape)
        d = np.delete(d, [0, -1], axis=0)
        print("new shape: ", d.shape)
        token_arr = np.ones((d.shape[0], 1))
        token_arr[0, 0] = 0
        token_arr[-1, 0] = 2
        new_data.append(np.concatenate((d, token_arr), axis=1))

    with open("D:/sddp_data/ProductionPlanning/stages_7/uncertainty_3/new/one_hot/labels.pickle", "wb") as fw:
        pickle.dump(new_data, fw)


def change_start_end_token_one_hot():
    if os.path.exists("D:/sddp_data/ProductionPlanning/stages_7/uncertainty_3/new/{}.pickle".format("labels_one_hot")):
        with open("D:/sddp_data/ProductionPlanning/stages_7/uncertainty_3/new/{}.pickle".format("labels_one_hot"), "rb") as fr:
            data = pickle.load(fr)
    else:
        raise FileNotFoundError

    new_data = []
    for d in data:
        print("origin shape: ", d.shape)
        encoding = np.eye(3)[d[:, -1].astype(int)]
        print("new shape: ", np.concatenate((d[:, :-1], encoding), axis=1).shape)
        new_data.append(np.concatenate((d[:, :-1], encoding), axis=1))
    print(new_data[0])
    with open("D:/sddp_data/ProductionPlanning/stages_7/uncertainty_3/new/one_hot/labels.pickle", "wb") as fw:
        pickle.dump(new_data, fw)


def delete_objective_function(load_path, save_path, idx_delete_start, idx_delete_end):
    if os.path.exists(load_path):
        with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
            data = pickle.load(fr)
    else:
        raise FileNotFoundError

    os.makedirs(save_path, exist_ok=True)
    new_data = []
    for d in data:
        d = np.delete(d, np.s_[idx_delete_start:idx_delete_end], axis=1)
        new_data.append(d)
    with open(os.path.join(save_path, "features.pickle"), "wb") as fw:
        pickle.dump(new_data, fw)


def except_outlier(load_path, save_path, low, high):
    with open(os.path.join(load_path, "labels.pickle"), "rb") as fr:
        data_label = pickle.load(fr)

    with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
        data_features = pickle.load(fr)

    new_data_labels = []
    new_data_features = []
    for d in range(len(data_label)):
        if high >= data_label[d].shape[0] >= low:
            new_data_labels.append(data_label[d])
            new_data_features.append(data_features[d])

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "labels.pickle"), "wb") as fw:
        pickle.dump(new_data_labels, fw)
    with open(os.path.join(save_path, "features.pickle"), "wb") as fw:
        pickle.dump(new_data_features, fw)


def change_dataset_order(load_path, save_path):
    with open(os.path.join(load_path, "labels.pickle"), "rb") as fr:
        data_label = pickle.load(fr)

    with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
        data_features = pickle.load(fr)

    train_data_length = (5 * len(data_features)) // 6
    new_data_labels = data_label[train_data_length:] + data_label[:train_data_length]
    new_data_features = data_features[train_data_length:] + data_features[:train_data_length]

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "labels.pickle"), "wb") as fw:
        pickle.dump(new_data_labels, fw)
    with open(os.path.join(save_path, "features.pickle"), "wb") as fw:
        pickle.dump(new_data_features, fw)


def change_token_to_integer(load_path, save_path):
    if os.path.exists(load_path):
        with open(os.path.join(load_path, "labels.pickle"), "rb") as fr:
            data = pickle.load(fr)
    else:
        raise FileNotFoundError

    os.makedirs(save_path, exist_ok=True)
    new_data = []
    for d in data:
        integer_token = np.ones((d.shape[0], 1))
        integer_token[0, 0] = 0
        integer_token[-1, 0] = 2
        new_data.append(np.concatenate((d[:, :-3], integer_token), axis=1))
    with open(os.path.join(save_path, "labels.pickle"), "wb") as fw:
        pickle.dump(new_data, fw)


def preprocess_sample_scenario_cuts(load_path, save_path):
    with open(os.path.join(load_path, "labels.pickle"), "rb") as fr:
        data_label = pickle.load(fr)

    cuts = {}
    for j in range(6):
        new_data = [data_label[i][1][:-3] for i in range(j, len(data_label), 6)]
        cuts[f'stage{j}'] = np.array(new_data)

    with open(os.path.join(save_path, f"cuts.pickle"), "wb") as fw:
        pickle.dump(cuts, fw)



if __name__ == "__main__":
    """
        Threshold:
        EnergyPlanning: 7~80
        MertonsPortfolioOptimization: 40
    """
    # except_outlier(load_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/train",
    #                save_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/train/except_outliers",
    #                low=0,
    #                high=40)

    """
        idx_delete_start, idx_delete_end:
        EnergyPlanning: 16, 24
        MertonsPortfolioOptimization: 18, 26
    """
    # delete_objective_function(load_path="D:/sddp_data/EnergyPlanning/stages_7/train/mm/original",
    #                           save_path="D:/sddp_data/EnergyPlanning/stages_7/train/mm",
    #                           idx_delete_start=16,
    #                           idx_delete_end=24)

    # delete_objective_function(load_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/predict/original/except_outliers",
    #                           save_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/predict/except_outliers",
    #                           idx_delete_start=18,
    #                           idx_delete_end=26)

    preprocess_sample_scenario_cuts(load_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/sample_scenario",
                                    save_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/sample_scenario")

    # change_token_to_integer(load_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/predict",
    #                         save_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/predict/change_loss")