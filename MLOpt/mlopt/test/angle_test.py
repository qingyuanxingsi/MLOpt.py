__author__ = 'qingyuanxingsi'

from mlopt.model.angle_measurement import AngleMeasurement
from mlopt.opt.optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt

"""
Angle-only measurement tests
"""

def test_angle():
    sample = np.array([[28976.985200060444, 12095.921247416274, -0.20460391230838582],
                       [21399.224900479396, 28005.663602853492, 1.1179398758981125],
                       [23359.266675802537, 28675.878103997853, 1.0212927844392057]])
    biased_sample = np.array([[14472.90041987611, 14852.459224022326, 0.27346669323819384],
                              [11638.620939799031, 10960.318365429848, 0.8792708847729779],
                              [12567.89408221255, 10702.533698252686, 1.0541683171468343]])
    precision = 1/(0.05**2)
    init_points = np.array([[10345.17500173, 10789.7906996],
                            [13000, 13000]])
    angle_measurement = AngleMeasurement(sample, precision)
    algorithms = [("Newton", "newton", "const"),
                  ("Damped Newton", "newton", "damped"),
                  ("Gradient Descent", "gradient", "wolfe"),
                  ("ILS", "gradient", "wolfe")
                 ]
    colors = ["green", "purple", "black", "red"]
    msizes = [6, 6, 4, 4]

    plt.figure(figsize=(12, 3))

    for i in range(len(init_points)):
        init_x = init_points[i]
        plt.subplot(1, 2, i+1)

        for j in range(len(algorithms)):
            algor = algorithms[j]
            optimizer = Optimizer(angle_measurement,
                                  init_x,
                                  direction=algor[1],
                                  step_size_chooser=algor[2],
                                  max_iteration=300)
            ret = optimizer.optimize()
            f_all = ret.f_all[0:ret.iteration]
            # f_all[np.isnan(f_all) | np.isinf(f_all)] = 50
            iteration = range(0, ret.iteration)
            plt.plot(iteration, f_all, color=colors[j],linestyle="solid",marker="o",markersize=msizes[j],label=algor[0])
            plt.ylim(0, 1.5*np.max(ret.f_all))
            # plt.ylim(0, 5)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Function Value")
        # plt.title("init = %.2f" % init_x[1])

    plt.savefig("fig/angle_measurement_test.png", transparent=True, bbox_inches="tight", pad_inches=0)

    # plot the function
    # plt.figure(figsize=(3, 4))
    # x = linspace(-15,15,100)
    # y = log(exp(x) + exp(-x))
    # plt.plot(x,y)
    # plt.title("log(exp(x)+exp(-x))")
    # plt.savefig("fig/1d-func.pdf", transparent=True, bbox_inches="tight", pad_inches=0)
    # plt.savefig("fig/1d-func.svg", transparent=True, bbox_inches="tight", pad_inches=0)

if __name__ == '__main__':
    test_angle()

