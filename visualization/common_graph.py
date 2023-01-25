import pickle

from scipy.stats import *
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

# parameters = {'axes.labelsize': 20,
#               'axes.titlesize': 20,
#               'xtick.labelsize': 15,
#               'ytick.labelsize': 15,
#               'legend.fontsize': 10}

parameters = {'axes.labelsize': 15,
              'axes.titlesize': 0,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15,
              'legend.fontsize': 20}

plt.rcParams.update(parameters)


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
    plt.vlines(80, 0, 800, color='red', linestyles='--', label=r'$x_\alpha$ for $F(x_\alpha) = \alpha$') # EP: (80, 1200), FP: (40, 2500)
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
    plt.legend(fontsize=15)

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


def get_leaning_evaluation_graph(load_path, save_path):
    problem = "FP"
    stage = "stage7"
    models = ["transformer", "transformer(decoder)"]
    evaluation_metrics = ["loss", "infeasibility", "error"]
    modes = ["train", "validation"]

    def lineplot(datas, data_labels, x="Step", y="Value", x_label="epoch", y_label="value", title=None):
        plt.figure(figsize=(10, 8))
        plt.grid(color="gray", alpha=0.5, linestyle="--")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()

        for i in range(len(datas)):
            plt.plot(datas[i][x], datas[i][y], color='royalblue' if i==0 else 'salmon', label=data_labels[i])
        plt.legend()
        plt.savefig(os.path.join(save_path, title + ".png"), dpi=300)
        plt.close()

    for metric in evaluation_metrics:
        if metric == "infeasibility":
            datas = []
            data_labels = []
            for model in models:
                datas.append(pd.read_csv(os.path.join(load_path, problem + "_" + stage + "_" + model + "_" + metric + ".csv"), index_col=0))
                model_name = "TranSDDP"
                if model == "transformer":
                    data_labels.append(model_name)
                else:
                    data_labels.append(model_name + "-Decoder")

            lineplot(datas, data_labels, y_label=metric + " ratio", title=metric.title())
            continue

        for mode in modes:
            datas = []
            data_labels = []
            for model in models:
                datas.append(pd.read_csv(os.path.join(load_path, problem + "_" + stage + "_" + model + "_" + metric + "_" + mode + ".csv"), index_col=0))
                model_name = "TranSDDP"
                if model == "transformer":
                    data_labels.append(model_name)
                else:
                    data_labels.append(model_name + "-Decoder")
            lineplot(datas, data_labels, y_label=metric, title=metric.title() + f"({mode})")


def get_time_comparison_graph(msp_time, sddp_time, sddp_t_train_time, sddp_t_eval_time, sddp_td_train_time, sddp_td_eval_time,
                              vfgl_time, l1_time, neural_sddp_eval_time, neural_sddp_training_time, prob):
    x = np.arange(0, 200)

    plt.figure(figsize=(10, 8))
    plt.grid(color="gray", alpha=0.5, linestyle="--")
    plt.title("Total Computation Time per # Problems")
    plt.xlabel("# problems")
    plt.ylabel("computation time (s)")

    # plt.plot(x, msp_time*x, label="MSP", linestyle=':', color='green')
    plt.plot(x, msp_time * x, label="MSP", color='green', linewidth="2")
    # plt.plot(x, sddp_time*x, label="SDDP", linestyle="--", color='blue')
    plt.plot(x, sddp_time * x, label="SDDP", color='blue', linewidth="2")
    # plt.plot(x, vfgl_time*x, label="VFGL", linestyle="--", color='indigo')
    plt.plot(x, vfgl_time * x, label="VFGL", color='indigo', linewidth="2")
    plt.plot(x, l1_time*x, label="Level 1 Dominance", color='deepskyblue', linewidth="2")
    plt.plot(x, neural_sddp_training_time+neural_sddp_eval_time*x, label=r'$\nu$-SDDP', color='y', linewidth="2")
    plt.plot(x, sddp_t_train_time+sddp_t_eval_time*x, label="TranSDDP", color='red', linewidth="2")
    plt.plot(x, sddp_td_train_time+sddp_td_eval_time*x, label="TranSDDP-Decoder", color='m', linewidth="2")
    plt.plot(0, neural_sddp_training_time, 'yD', label=r'$\nu$-SDDP: training time', linewidth="2")
    plt.plot(0, sddp_t_train_time, 'rD', label='TranSDDP: training time', linewidth="2")
    plt.plot(0, sddp_td_train_time, 'mD', label='TranSDDP-Decoder: training time', linewidth="2")

    plt.legend(prop={'size': 13})
    plt.savefig(f"D:/sddp_data/obj_data/computation_time/{prob}.png", dpi=300)

if __name__ == '__main__':
    # load_path = "D:/sddp_data/EnergyPlanning/stages_15/sample/long_iteration"
    #
    # with open(os.path.join(load_path, "objective.pickle"), "rb") as fr:
    #     data_obj = pickle.load(fr)
    #
    # with open(os.path.join(load_path, "time.pickle"), "rb") as fr:
    #     data_time = pickle.load(fr)
    #
    # with open(os.path.join(load_path, "opt_gap.pickle"), "rb") as fr:
    #     data_opt_gap = pickle.load(fr)
    #
    # with open(os.path.join(load_path, "ub.pickle"), "rb") as fr:
    #     data_ub = pickle.load(fr)
    #
    # data_opt_gap = [800] + data_opt_gap
    # data_ub = [800] + data_ub
    # data_lb = [data_ub[i] - data_opt_gap[i] for i in range(len(data_opt_gap))]
    #
    # get_ub_lb_graph_sddp(data_lb, data_ub, data_opt_gap)
    #
    # get_iter_time_graph_sddp(data_time)
    #
    # stage_0_objs = []
    # cnt = 0
    # for d in data_obj:
    #     stage_0_objs.append(d['stage0'])
    #     cnt += 1
    #
    # # obj_diff = []
    # # for i in range(1, len(data)):
    # #     obj_diff.append((data[i]['stage0']-data[i-1]['stage0'])/data[-1]['stage0'])
    #
    # get_2dim_graph(x=np.arange(len(stage_0_objs)),
    #                y=stage_0_objs,
    #                x_label='iteration',
    #                y_label='obj value')
    #
    # with open("D:/sddp_data/EnergyPlanning/stages_7/train/original/{}.pickle".format("labels"), "rb") as fr:
    #     data_obj = pickle.load(fr)
    #
    # get_dist_graph(get_shape_inform(data_obj), x_label="# of cutting planes", y_label="count")


    ''' error ratio, loss, infeasibility graph'''
    # get_leaning_evaluation_graph(load_path="D:/sddp_data/MertonsPortfolioOptimization/result/tensorboardresult",
    #                              save_path="D:/sddp_data/MertonsPortfolioOptimization/result/tensorboardresult/graph")

    '''computation time 비교 그래프'''

    # EP
    get_time_comparison_graph(msp_time=331.58,
                              sddp_time=183.01,
                              sddp_t_train_time=119*60,
                              sddp_t_eval_time=1.99,
                              sddp_td_train_time=99*60,
                              sddp_td_eval_time=1.71,
                              vfgl_time=428.559,
                              l1_time=121.096,
                              neural_sddp_eval_time=2.047 / 102,
                              neural_sddp_training_time=(807 * 60 + 51),
                              prob="EP")

    # MPO
    get_time_comparison_graph(msp_time=229.93,
                              sddp_time=93.35,
                              sddp_t_train_time=95*60,
                              sddp_t_eval_time=0.46,
                              sddp_td_train_time=72*60,
                              sddp_td_eval_time=0.33,
                              vfgl_time=257.387,
                              l1_time=91.595,
                              neural_sddp_eval_time=2.0939/102,
                              neural_sddp_training_time=(352*60+50),
                              prob="MPO")

    # PO
    get_time_comparison_graph(msp_time=483.046,
                              sddp_time=98.637,
                              sddp_t_train_time=136*60,
                              sddp_t_eval_time=2.410,
                              sddp_td_train_time=116*60,
                              sddp_td_eval_time=2.018,
                              vfgl_time=266.514,
                              l1_time=49.486,
                              neural_sddp_eval_time=1.75 / 102,
                              neural_sddp_training_time=(658 * 60),
                              prob="PO")