import numpy as np


def get_parameters(args):
    if args.prob == "ProductionPlanning":
        A = np.array([[-1, 0, 0, -1, 0, 0, 1, 0, 0],
                      [0, -1, 0, 0, -1, 0, 0, 1, 0],
                      [0, 0, -1, 0, 0, -1, 0, 0, 1],
                      [1, 2, 5, 0, 0, 0, 0, 0, 0]], dtype=float)
        B = np.array([[0, 0, 0, 0, 0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
        b = np.array([[0],
                      [0],
                      [0],
                      [-10]])
        c = np.array([0, 0, 0, 6, 12, 20, 3, 7, 10])
    elif args.prob == "EnergyPlanning":
        A = np.array([[1, 0, 0, 0],
                      [-1, 1, 1, 0],
                      [0, 0, -1, -1]], dtype=float)
        B = np.array([[0, -1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]], dtype=float)
        b = np.array([[-40],
                       [0],
                       [20]])
        c = np.array([2, 7, -0.1, 5])
    elif args.prob == "MertonsPortfolioOptimization":
        riskFree = 0.03
        delta_t = 1 / (args.num_stages - 1)
        A = np.array([[0, 0, 0, 1],
                      [1, 1, 1, -1]], dtype=float)
        B = np.array([[0, -(1+riskFree*delta_t), 0, 0],
                      [0, 0, 0, 0]], dtype=float)
        b = np.array([[-100],
                      [0]])
        c = np.array([0, 0, 1, 0])
    else:
        raise NotImplementedError
    return A, B, b, c
