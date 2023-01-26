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
        # integer_token = np.arange(1, d.shape[0]+1).reshape((d.shape[0], 1))
        integer_token = np.ones((d.shape[0], 1)) + 1
        integer_token[0, 0] = 1
        integer_token[-1, 0] = 3
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


def change_stage_information(load_path, save_path, n_stages):
    if os.path.exists(load_path):
        with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
            data = pickle.load(fr)
    else:
        raise FileNotFoundError

    os.makedirs(save_path, exist_ok=True)
    new_data = []
    for d in data:
        d[:, -1] = n_stages - d[:, -1] * (n_stages-1)
        new_data.append(d)
    with open(os.path.join(save_path, "features.pickle"), "wb") as fw:
        pickle.dump(new_data, fw)


def change_stage_information_to_origin(load_path, save_path, n_stages):
    if os.path.exists(load_path):
        with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
            data = pickle.load(fr)
    else:
        raise FileNotFoundError

    os.makedirs(save_path, exist_ok=True)
    new_data = []
    for d in data:
        d[:, -1] = (d[:, -1] * (n_stages-2) + 1)/(n_stages-1)
        new_data.append(d)
    with open(os.path.join(save_path, "features.pickle"), "wb") as fw:
        pickle.dump(new_data, fw)


def change_stage_information_to_ratio(load_path, save_path):
    if os.path.exists(load_path):
        with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
            data = pickle.load(fr)
    else:
        raise FileNotFoundError

    os.makedirs(save_path, exist_ok=True)
    new_data = []
    for d in data:
        d[:, -1] /= 14
        new_data.append(d)
    with open(os.path.join(save_path, "features.pickle"), "wb") as fw:
        pickle.dump(new_data, fw)


def concat_data(load_path_data1, load_path_data2, save_path):
    with open(os.path.join(load_path_data1, "labels.pickle"), "rb") as fr:
        data1_label = pickle.load(fr)

    with open(os.path.join(load_path_data1, "features.pickle"), "rb") as fr:
        data1_features = pickle.load(fr)

    with open(os.path.join(load_path_data2, "labels.pickle"), "rb") as fr:
        data2_label = pickle.load(fr)

    with open(os.path.join(load_path_data2, "features.pickle"), "rb") as fr:
        data2_features = pickle.load(fr)

    new_data_features = data1_features + data2_features
    new_data_labels = data1_label + data2_label

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "labels.pickle"), "wb") as fw:
        pickle.dump(new_data_labels, fw)
    with open(os.path.join(save_path, "features.pickle"), "wb") as fw:
        pickle.dump(new_data_features, fw)


def devide_data(load_path, save_path_data1, save_path_data2, n_stages, seg):
    with open(os.path.join(load_path, "labels.pickle"), "rb") as fr:
        data_label = pickle.load(fr)

    with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
        data_features = pickle.load(fr)

    new_data1_features = data_features[:(n_stages-1)*seg]
    new_data1_labels = data_label[:(n_stages-1)*seg]

    new_data2_features = data_features[(n_stages - 1) * seg:]
    new_data2_labels = data_label[(n_stages - 1) * seg:]

    os.makedirs(save_path_data1, exist_ok=True)
    with open(os.path.join(save_path_data1, "labels.pickle"), "wb") as fw:
        pickle.dump(new_data1_labels, fw)
    with open(os.path.join(save_path_data1, "features.pickle"), "wb") as fw:
        pickle.dump(new_data1_features, fw)

    os.makedirs(save_path_data2, exist_ok=True)
    with open(os.path.join(save_path_data2, "labels.pickle"), "wb") as fw:
        pickle.dump(new_data2_labels, fw)
    with open(os.path.join(save_path_data2, "features.pickle"), "wb") as fw:
        pickle.dump(new_data2_features, fw)

def move_one_data(load_path, save_path):
    with open(os.path.join(load_path, "labels.pickle"), "rb") as fr:
        data_label = pickle.load(fr)

    with open(os.path.join(load_path, "features.pickle"), "rb") as fr:
        data_features = pickle.load(fr)

    dl = data_label.pop(0)
    data_label.append(dl)

    df = data_features.pop(0)
    data_features.append(df)

    with open(os.path.join(save_path, "labels.pickle"), "wb") as fw:
        pickle.dump(data_label, fw)
    with open(os.path.join(save_path, "features.pickle"), "wb") as fw:
        pickle.dump(data_features, fw)



if __name__ == "__main__":
    """
        Threshold:
        EnergyPlanning: 7~80
        MertonsPortfolioOptimization: 40
        ProductionPlanning: 100
    """

    """
        idx_delete_start, idx_delete_end:
        EnergyPlanning: 16, 24
        MertonsPortfolioOptimization: 18, 26
        ProductionPlanning: 36, 54
    """

    # delete_objective_function(load_path="D:/sddp_data/ProductionPlanning/stages_7/sample_scenario",
    #                           save_path="D:/sddp_data/ProductionPlanning/stages_7/sample_scenario",
    #                           idx_delete_start=36,
    #                           idx_delete_end=54)

    # except_outlier(load_path="D:/sddp_data/EnergyPlanning/stages_10/train",
    #                save_path="D:/sddp_data/EnergyPlanning/stages_10/train/except_outliers",
    #                low=0,
    #                high=80)


    # change_token_to_integer(load_path="D:/sddp_data/ProductionPlanning/stages_7/sample_scenario",
    #                         save_path="D:/sddp_data/ProductionPlanning/stages_7/sample_scenario")

    # preprocess_sample_scenario_cuts(load_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/sample_scenario",
    #                                 save_path="D:/sddp_data/MertonsPortfolioOptimization/stages_7/sample_scenario")
    #

    # change_stage_information(load_path="D:/sddp_data/EnergyPlanning/stages_7/train/mm/except_outliers/change_loss",
    #                          save_path="D:/sddp_data/EnergyPlanning/stages_7/train/mm/except_outliers/change_loss/stage_information_rest/integer_stage_inform",
    #                          n_stages=7)
    #

    # change_stage_information_to_origin(load_path="D:/sddp_data/ProductionPlanning/stages_7/train/original",
    #                          save_path="D:/sddp_data/ProductionPlanning/stages_7/train/original/change_loss",
    #                          n_stages=7)

    # change_stage_information_to_ratio(load_path="D:/sddp_data/EnergyPlanning/stages_15/predict/change_loss/stage_information_rest/integer_stage_inform",
    #                                   save_path="D:/sddp_data/EnergyPlanning/stages_15/predict/change_loss/stage_information_rest")

    # concat_data(load_path_data1="D:/sddp_data/EnergyPlanning/stages_7/train/original",
    #             load_path_data2="D:/sddp_data/EnergyPlanning/stages_7/train/mm/original",
    #             save_path="D:/sddp_data/EnergyPlanning/stages_7/train/total/original")

    # devide_data(load_path="D:/sddp_data/EnergyPlanning/stages_15/predict/change_loss/stage_information_rest/integer_stage_inform",
    #             save_path_data1="D:/sddp_data/EnergyPlanning/stages_15/train/change_loss/stage_information_rest/integer_stage_inform/devided_data",
    #             save_path_data2="D:/sddp_data/EnergyPlanning/stages_15/predict/change_loss/stage_information_rest/integer_stage_inform/devided_data",
    #             n_stages=15,
    #             seg=96)

    # move_one_data(load_path="D:/sddp_data/EnergyPlanning/stages_7/train/original/origin",
    #               save_path="D:/sddp_data/EnergyPlanning/stages_7/train/original")