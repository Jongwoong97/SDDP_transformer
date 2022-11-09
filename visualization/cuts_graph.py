from matplotlib import pyplot as plt
import numpy as np
import os


def get_cut_graph(target_cut, pred_cut, var_idx, args, save_path):
    if args.prob == "ProductionPlanning":
        var_start_idx = 6
    elif args.prob == "EnergyPlanning":
        var_start_idx = 1
    elif args.prob == "MertonsPortfolioOptimization":
        var_start_idx = 0
    else:
        raise NotImplementedError

    x = np.array(range(-500, 500))

    fig, ax = plt.subplots(2, 3, figsize=(20, 15))

    for stage in range((len(target_cut))//2):
        ax[0, int(stage % ax.shape[-1])].grid(color="gray", alpha=0.5, linestyle="--")
        ax[0, int(stage % ax.shape[-1])].set_title("stage{}_target_{}".format(stage, var_idx))
        ax[1, int(stage % ax.shape[-1])].grid(color="gray", alpha=0.5, linestyle="--")
        ax[1, int(stage % ax.shape[-1])].set_title("stage{}_pred_{}".format(stage, var_idx))

        for i in range(len(target_cut[stage])):
            ax[0, int(stage % ax.shape[-1])].plot(x, target_cut[stage][i][var_start_idx+var_idx] * x + target_cut[stage][i][-4], label=i)

        ax[0, int(stage % ax.shape[-1])].legend()

        for i in range(len(pred_cut[stage])):
            ax[1, int(stage % ax.shape[-1])].plot(x, pred_cut[stage][i][var_start_idx+var_idx] * x / (-pred_cut[stage][i][-5]) + pred_cut[stage][i][-4] / (-pred_cut[stage][i][-5]), label=i)
        ax[1, int(stage % ax.shape[-1])].legend()

    plt.savefig(os.path.join(save_path, "stage0~2_cuts_{}.png".format(var_idx)))
    plt.clf()

    fig, ax = plt.subplots(2, 3, figsize=(20, 15))

    for stage in range((len(target_cut))//2, len(target_cut)):
        ax[0, int(stage % ax.shape[-1])].grid(color="gray", alpha=0.5, linestyle="--")
        ax[0, int(stage % ax.shape[-1])].set_title("stage{}_target_{}".format(stage, var_idx))
        ax[1, int(stage % ax.shape[-1])].grid(color="gray", alpha=0.5, linestyle="--")
        ax[1, int(stage % ax.shape[-1])].set_title("stage{}_pred_{}".format(stage, var_idx))

        for i in range(len(target_cut[stage])):
            ax[0, int(stage % ax.shape[-1])].plot(x, target_cut[stage][i][var_start_idx+var_idx] * x + target_cut[stage][i][-4], label=i)

        ax[0, int(stage % ax.shape[-1])].legend()

        for i in range(len(pred_cut[stage])):
            ax[1, int(stage % ax.shape[-1])].plot(x, pred_cut[stage][i][var_start_idx+var_idx] * x / (-pred_cut[stage][i][-5]) + pred_cut[stage][i][-4] / (-pred_cut[stage][i][-5]), label=i)

        ax[1, int(stage % ax.shape[-1])].legend()

    plt.savefig(os.path.join(save_path, "stage3~5_cuts_{}.png".format(var_idx)))
    plt.clf()


def get_cut_graph_stage0(target_cut, pred_cut, var_idx, args, save_path):
    if args.prob == "ProductionPlanning":
        var_start_idx = 6
    elif args.prob == "EnergyPlanning":
        var_start_idx = 1
    elif args.prob == "MertonsPortfolioOptimization":
        var_start_idx = 0
    else:
        raise NotImplementedError

    x = np.array(range(-500, 500))

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    ax[0, 0].grid(color="gray", alpha=0.5, linestyle="--")
    ax[0, 1].grid(color="gray", alpha=0.5, linestyle="--")
    ax[1, 0].grid(color="gray", alpha=0.5, linestyle="--")
    ax[1, 1].grid(color="gray", alpha=0.5, linestyle="--")
    if var_idx == 0:
        ax[0, 0].set_title("SDDP Cuts(Stock)")
        ax[0, 1].set_title("SDDP-Transformer Cuts(Stock)")
        ax[1, 0].set_title("Piecewise Linear Function of SDDP(Stock)")
        ax[1, 1].set_title("Piecewise Linear Function of SDDP-Transformer(Stock)")
    else:
        ax[0, 0].set_title("SDDP Cuts(Bond)")
        ax[0, 1].set_title("SDDP-Transformer Cuts(Bond)")
        ax[1, 0].set_title("Piecewise Linear Function of SDDP(Bond)")
        ax[1, 1].set_title("Piecewise Linear Function of SDDP-Transformer(Bond)")

    for i in range(len(target_cut[0])):
        ax[0, 0].plot(x, target_cut[0][i][var_start_idx+var_idx] * x + target_cut[0][i][-4], label=i)
    ax[0, 0].legend()

    for i in range(len(pred_cut[0])):
        ax[0, 1].plot(x, pred_cut[0][i][var_start_idx+var_idx] * x / (-pred_cut[0][i][-5]) + pred_cut[0][i][-4] / (-pred_cut[0][i][-5]), label=i)
    # ax[0, 1].legend()

    target_cut_length = target_cut[0][:][var_start_idx+var_idx].shape[0]
    target_y = np.max(np.reshape(target_cut[0][:][var_start_idx+var_idx], (target_cut_length, 1)) @ np.reshape(x, (1, x.shape[0]))
                      + np.reshape(target_cut[0][:][-4], (target_cut_length, 1)), axis=0)
    print(target_y[0])
    ax[1, 0].plot(x, target_y)

    predict_cut_length = pred_cut[0][:][var_start_idx+var_idx].shape[0]
    predict_y = np.max(np.reshape(pred_cut[0][:][var_start_idx+var_idx]/(-pred_cut[0][:][-5]), (predict_cut_length, 1)) @ np.reshape(x, (1, x.shape[0]))
                       + np.reshape(pred_cut[0][:][-4] / (-pred_cut[0][:][-5]), (predict_cut_length, 1)), axis=0)
    ax[1, 1].plot(x, predict_y)

    plt.savefig(os.path.join(save_path, "stage0_cuts_{}.png".format(var_idx)))
    plt.clf()
