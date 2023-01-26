from matplotlib import pyplot as plt
import numpy as np
import os


def get_cut_graph(target_cut, pred_cut, var_idx, args, save_path):
    x = np.array(range(-500, 500))

    var_name = ""
    if args.prob == "ProductionPlanning":
        var_start_idx = 6
        if var_idx == 0:
            var_name = "1st product storage"
        elif var_idx == 1:
            var_name = "2nd product storage"
        else:
            var_name = "3rd product storage"
    elif args.prob == "EnergyPlanning":
        var_start_idx = 1
        var_name = "reservoir final"
    elif args.prob == "MertonsPortfolioOptimization":
        x = x/5
        var_start_idx = 0
        if var_idx == 0:
            var_name = "stock"
        else:
            var_name = "bond"
    else:
        raise NotImplementedError

    if args.model == "transformer":
        model_name = "TranSDDP"
    else:
        model_name = "TranSDDP-Decoder"

    fig, ax = plt.subplots(3, 2, figsize=(16, 24), sharex='all', sharey='row')
    for stage in range((len(target_cut))//2):
        ax[int(stage % ax.shape[0]), 0].grid(color="gray", alpha=0.5, linestyle="--")
        ax[int(stage % ax.shape[0]), 0].set_title("stage{} ({})".format(stage, model_name), size=14)
        ax[int(stage % ax.shape[0]), 1].grid(color="gray", alpha=0.5, linestyle="--")
        ax[int(stage % ax.shape[0]), 1].set_title("stage{} (SDDP)".format(stage), size=14)

        for i in range(len(target_cut[stage])):
            if stage == 0 and i <= 20:
                ax[int(stage % ax.shape[0]), 1].plot(x, target_cut[stage][i][var_start_idx+var_idx] * x + target_cut[stage][i][-2], label=f"cut {i}")
            else:
                ax[int(stage % ax.shape[0]), 1].plot(x, target_cut[stage][i][var_start_idx+var_idx] * x + target_cut[stage][i][-2])

        for i in range(len(pred_cut[stage])):
            ax[int(stage % ax.shape[0]), 0].plot(x, pred_cut[stage][i][var_start_idx+var_idx] * x / (-pred_cut[stage][i][-3]) + pred_cut[stage][i][-2] / (-pred_cut[stage][i][-3]))

    # set labels
    for axs in ax[-1]:
        axs.set_xlabel(f"x: {var_name}", fontsize=14)
    for axs in ax[:, 0]:
        axs.set_ylabel("l(x): cutting plane", fontsize=14)

    ax[0, -1].legend(loc="upper left", bbox_to_anchor=(1.03, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "stage0~2_cuts_{}.png".format(var_idx)), dpi=300)
    plt.clf()

    fig, ax = plt.subplots(3, 2, figsize=(16, 24), sharex='all', sharey='row')

    for stage in range((len(target_cut))//2, len(target_cut)):
        ax[int(stage % ax.shape[0]), 0].grid(color="gray", alpha=0.5, linestyle="--")
        ax[int(stage % ax.shape[0]), 0].set_title("stage{} ({})".format(stage, model_name), size=14)
        ax[int(stage % ax.shape[0]), 1].grid(color="gray", alpha=0.5, linestyle="--")
        ax[int(stage % ax.shape[0]), 1].set_title("stage{} (SDDP)".format(stage), size=14)

        for i in range(len(target_cut[stage])):
            if stage == (len(target_cut))//2 and i <= 20:
                ax[int(stage % ax.shape[0]), 1].plot(x, target_cut[stage][i][var_start_idx+var_idx] * x + target_cut[stage][i][-2], label=f"cut {i}")
            else:
                ax[int(stage % ax.shape[0]), 1].plot(x, target_cut[stage][i][var_start_idx+var_idx] * x + target_cut[stage][i][-2])

        for i in range(len(pred_cut[stage])):
            ax[int(stage % ax.shape[0]), 0].plot(x, pred_cut[stage][i][var_start_idx+var_idx] * x / (-pred_cut[stage][i][-3]) + pred_cut[stage][i][-2] / (-pred_cut[stage][i][-3]))

    # set labels
    for axs in ax[-1]:
        axs.set_xlabel(f"x: {var_name}", fontsize=14)
    for axs in ax[:, 0]:
        axs.set_ylabel("l(x): cutting plane", fontsize=14)

    ax[0, -1].legend(loc="upper left", bbox_to_anchor=(1.03, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "stage3~5_cuts_{}.png".format(var_idx)), dpi=300)
    plt.clf()


def get_sample_scenario_cuts_graph(cuts, pred_cut, var_idx, args):
    x = np.arange(-500, 500)
    var_name = ""
    if args.prob == "ProductionPlanning":
        var_start_idx = 6
    elif args.prob == "EnergyPlanning":
        var_start_idx = 1
        var_name = "reservoir final"
    elif args.prob == "MertonsPortfolioOptimization":
        x = x/5
        var_start_idx = 0
        if var_idx == 0:
            var_name = "stock"
        else:
            var_name = "bond"
    else:
        raise NotImplementedError

    fig, ax = plt.subplots(2, 3, figsize=(21, 14))
    for stage in range(len(cuts)):
        ax[0 if stage < len(cuts)/2 else 1, int(stage % ax.shape[1])].grid(color="gray", alpha=0.5, linestyle="--")
        ax[0 if stage < len(cuts)/2 else 1, int(stage % ax.shape[1])].set_title("stage{}".format(stage))

        for i in range(cuts[f"stage{stage}"].shape[0]):
            if i == 0:
                ax[0 if stage < len(cuts)/2 else 1, int(stage % ax.shape[1])].plot(x, cuts[f"stage{stage}"][i][var_start_idx + var_idx] * x + cuts[f"stage{stage}"][i][-1], label="cut sample", color='khaki')
            else:
                ax[0 if stage < len(cuts)/2 else 1, int(stage % ax.shape[1])].plot(x, cuts[f"stage{stage}"][i][var_start_idx + var_idx] * x + cuts[f"stage{stage}"][i][-1], color='khaki')

        avg_cut = np.mean(cuts[f"stage{stage}"], axis=0)
        ax[0 if stage < len(cuts)/2 else 1, int(stage % ax.shape[1])].plot(x, avg_cut[var_start_idx + var_idx] * x + avg_cut[-1], label="cut mean", color='blue')
        ax[0 if stage < len(cuts)/2 else 1, int(stage % ax.shape[1])].plot(x, -pred_cut[f"stage{stage}"][var_start_idx + var_idx] * x / pred_cut[f"stage{stage}"][-3] + -pred_cut[f"stage{stage}"][-2] / pred_cut[f"stage{stage}"][-3], label="cut prediction", color='red')

    # set labels
    for axs in ax[1]:
        axs.set_xlabel(f"x: {var_name}", size=12)
    for axs in ax[:, 0]:
        axs.set_ylabel("l(x): cutting plane", size=12)

    ax[0, -1].legend(loc="upper left", bbox_to_anchor=(1.03, 1))
    plt.show()
