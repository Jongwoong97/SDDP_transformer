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


def except_outlier(load_path, save_path, threshold):
    with open(os.path.join(load_path, "labels.pickle"), "rb") as fr:
        data_label = pickle.load(fr)

    with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
        data_features = pickle.load(fr)

    new_data_labels = []
    new_data_features = []
    for d in range(len(data_label)):
        if data_label[d].shape[0] <= threshold:
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


if __name__ == "__main__":
    """
        Threshold:
        EnergyPlanning: 80
        MertonsPortfolioOptimization: 60
    """
    # except_outlier(load_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/predict/original",
    #                save_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/predict/original/except_outliers",
    #                threshold=60)

    """
        idx_delete_start, idx_delete_end:
        EnergyPlanning: 16, 24
        MertonsPortfolioOptimization: 18, 26
    """
    delete_objective_function(load_path="D:/sddp_data/EnergyPlanning/stages_7/predict/original",
                              save_path="D:/sddp_data/EnergyPlanning/stages_7/predict",
                              idx_delete_start=16,
                              idx_delete_end=24)

    # delete_objective_function(load_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/predict/original/except_outliers",
    #                           save_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/predict/except_outliers",
    #                           idx_delete_start=18,
    #                           idx_delete_end=26)
