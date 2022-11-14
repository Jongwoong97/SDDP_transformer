import pickle

from scipy.stats import *
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os

def get_2dim_graph(x, y, x_label, y_label, save_path=''):
    sns.lineplot(x, y)
    # plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    plt.show()
    # plt.savefig(os.path.join(save_path, f'{title}.png'))
    # plt.clf()


def get_dist_graph(data, x_label, y_label):
    sns.displot(data=data, log_scale=True, kind="kde")
    # sns.lineplot(x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def get_shape_inform(data):
    shape_list = []
    cnt = 0
    for d in data:
        if d.shape[0] > 100:
            cnt += 1
        shape_list.append(d.shape[0])
    # plt.plot(shape_list)
    return shape_list


if __name__ == '__main__':
    load_path = "D:/sddp_data/EnergyPlanning/stages_7/sample/long_iteration"

    with open(os.path.join(load_path, "objective.pickle"), "rb") as fr:
        data = pickle.load(fr)

    stage_0_objs = []
    cnt = 0
    for d in data:
        stage_0_objs.append(d['stage0'])
        cnt += 1

    # obj_diff = []
    # for i in range(1, len(data)):
    #     obj_diff.append((data[i]['stage0']-data[i-1]['stage0'])/data[-1]['stage0'])

    get_2dim_graph(x=np.arange(len(stage_0_objs)),
                   y=stage_0_objs,
                   x_label='iteration',
                   y_label='obj value')

    with open("D:/sddp_data/EnergyPlanning/stages_7/train/mm/original/{}.pickle".format("labels"), "rb") as fr:
        data = pickle.load(fr)

    get_dist_graph(get_shape_inform(data), x_label="# of cuts", y_label="count")