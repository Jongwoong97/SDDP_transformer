from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

parameters = {'axes.labelsize': 12,
              'axes.titlesize': 12,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'legend.fontsize': 10}

plt.rcParams.update(parameters)


def plot_head_map(mma, target_labels, source_labels):
    # sns.set_palette("pastel", 2)
    ax = sns.heatmap(mma.T, annot=True, fmt=".2f", cmap="Reds", xticklabels=True, yticklabels=True)
    ax.set_xticks(ticks=np.arange(mma.shape[0])+0.5, labels=target_labels)
    ax.set_yticks(ticks=np.arange(mma.shape[1])+0.5, labels=source_labels)
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    plt.xlim(0, mma.shape[0])
    plt.ylim(mma.shape[1], 0)
    plt.xlabel("target")
    plt.ylabel("source")
    plt.show()


def read_plot_alignment_matrices(source_labels, target_labels, alpha):
    mma = alpha.cpu().data.numpy()
    plot_head_map(mma, target_labels, source_labels)

