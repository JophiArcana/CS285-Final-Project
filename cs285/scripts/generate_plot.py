import os
import pickle
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    os.chdir('../../')
    n_seeds = 3

    x = 'Train_EnvstepsSoFar'
    y = 'Eval_AverageReturn'

    exp_name = 'sanity_invpendulum_reinforce'
    with open(f'plot_data/{exp_name}.pkl', 'rb') as fp:
        return_logs = pickle.load(fp)
    plt.plot(return_logs[x], return_logs[y], label=exp_name)

    # b = np.array([96, 232, 247], dtype=float) / 255
    # for s in range(n_seeds):
    #     exp_name = f'lunarlander_seed{s + 1}'
    #     with open(f'plot_data/{exp_name}.pkl', 'rb') as fp:
    #         return_logs = pickle.load(fp)
    #     plt.plot(return_logs[x], return_logs[y], color=b * (s + 1) / n_seeds, label=exp_name)
    #
    # r = np.array([247, 111, 96], dtype=float) / 255
    # for s in range(n_seeds):
    #     exp_name = f'lunarlander_doubleq_seed{s + 1}'
    #     with open(f'plot_data/{exp_name}.pkl', 'rb') as fp:
    #         return_logs = pickle.load(fp)
    #     plt.plot(return_logs[x], return_logs[y], color=r * (s + 1) / n_seeds, label=exp_name)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Pacman')
    plt.legend()
    plt.show()
