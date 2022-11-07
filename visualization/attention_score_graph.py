from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os


# http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
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


    # fig, ax = plt.subplots()
    # heatmap = ax.pcolor(mma.T, cmap=plt.cm.Greys).imshow(np.round(mma.T, decimals=2))
    #
    # # put the major ticks at the middle of each cell
    # ax.set_xticks(np.arange(mma.shape[0]) + 0.5, minor=False)  # mma.shape[1] = target seq 길이
    # ax.set_yticks(np.arange(mma.shape[1]) + 0.5, minor=False)  # mma.shape[0] = input seq 길이
    #
    # # without this I get some extra columns rows
    # # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    # ax.set_xlim(0, int(mma.shape[0]))
    # ax.set_ylim(0, int(mma.shape[1]))
    #
    # # want a more natural, table-like display
    # ax.invert_yaxis()
    # ax.xaxis.tick_top()
    #
    # # source words -> column labels
    # ax.set_xticklabels(target_labels, minor=False)
    # # target words -> row labels
    # ax.set_yticklabels(source_labels, minor=False)
    #
    # plt.xticks(rotation=45)
    #
    # # plt.tight_layout()
    # plt.show()


def read_plot_alignment_matrices(source_labels, target_labels, alpha):
    mma = alpha.cpu().data.numpy()

    plot_head_map(mma, target_labels, source_labels)
