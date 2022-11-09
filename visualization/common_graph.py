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


if __name__ == '__main__':
    sample = [1, 3, 4, 4.5, 4.75, 4.875, 4.9, 4.91, 4.92]
    get_2dim_graph(x=np.arange(len(sample)),
                   y=sample,
                   x_label='iteration',
                   y_label='obj value')