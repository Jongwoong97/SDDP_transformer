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
    # sns.displot(data=data, log_scale=True)
    sns.displot(data=data, color='royalblue')
    plt.title("Histogram of # Cutting Planes")
    plt.vlines(80, 0, 1200, color='red', linestyles='--', label=r'$x_\alpha$ for $F(x_\alpha) = \alpha$')
    plt.legend()
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


def get_obj_graph(datas):
    plt.plot()
    plt.grid(color="gray", alpha=0.5, linestyle="-")
    plt.title("Objective Value per # Cutting Planes")
    plt.xlabel("# cutting planes")
    plt.ylabel("objective value")

    for key, value in datas.items():
        if key == 'Optimal':
            plt.plot(value, label=key, linestyle=':', color='gold', linewidth=2.2)
        elif key == "SDDP":
            plt.plot(value, label=key, linestyle='--', color='royalblue')
        else:
            plt.plot(value, label=key, color='salmon')
    plt.legend()
    plt.show()


def get_iter_time_graph_sddp(data):
    plt.plot()
    plt.grid(color="gray", alpha=0.5, linestyle="--")
    plt.title("SDDP: Elapsed Time per Iteration")
    plt.xlabel("iteration")
    plt.ylabel("time(s)")
    plt.plot(data, color='royalblue')
    plt.show()


def get_ub_lb_graph_sddp(data_lb, data_ub, data_opt_gap):
    plt.plot()
    plt.grid(color="gray", alpha=0.5, linestyle="--")
    plt.title("SDDP: Lower Bound vs Upper Bound")
    plt.xlabel("iteration")
    plt.ylabel("bound")

    plt.plot(data_lb, label="lower bound", color='lightsalmon')
    plt.plot(data_ub, label="upper bound", color='cornflowerblue')
    plt.plot(data_opt_gap, label="optimality gap", color='dimgray', linestyle=':')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    load_path = "D:/sddp_data/EnergyPlanning/stages_7/sample/long_iteration/origin"

    with open(os.path.join(load_path, "objective.pickle"), "rb") as fr:
        data_obj = pickle.load(fr)

    with open(os.path.join(load_path, "ti me.pickle"), "rb") as fr:
        data_time = pickle.load(fr)

    with open(os.path.join(load_path, "opt_gap.pickle"), "rb") as fr:
        data_opt_gap = pickle.load(fr)

    with open(os.path.join(load_path, "ub.pickle"), "rb") as fr:
        data_ub = pickle.load(fr)

    data_opt_gap = [800] + data_opt_gap
    data_ub = [800] + data_ub
    data_lb = [data_ub[i] - data_opt_gap[i] for i in range(len(data_opt_gap))]

    get_ub_lb_graph_sddp(data_lb, data_ub, data_opt_gap)


    get_iter_time_graph_sddp(data_time)

    stage_0_objs = []
    cnt = 0
    for d in data_obj:
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
        data_obj = pickle.load(fr)

    get_dist_graph(get_shape_inform(data_obj), x_label="# of cutting planes", y_label="count")