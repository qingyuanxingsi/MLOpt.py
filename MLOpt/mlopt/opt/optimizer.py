__author__ = 'qingyuanxingsi'

import numpy as np
from mlopt.model.angle_measurement import AngleMeasurement
import numpy.linalg as lg

"""
An stub implementation of a optimizer
@version:1.0
@date:2014-12-3
"""

# TODO:To be modified for quasi-newton method
def wolfe_line_search(opt_problem, point, direction, c1=0.01, c2=0.9, t0=1, tol=0.001):
    """
    Line Search using wolfe conditions
    :param opt_problem:
    :param point:
    :param direction:
    :param c1:
    :param c2:
    :param t0:
    :param tol:0.001 originally
    :return:
    """
    low = 0
    high = np.inf

    t = t0
    fx = opt_problem.obj_val(point)
    dfx = opt_problem.gradient(point)

    # TODO[1]:Understanding search strategy
    while True:
        if opt_problem.obj_val(point + t*direction) > fx + c1*t*np.dot(dfx, direction):
            high = t
            t = (low + high) / 2
        elif np.dot(opt_problem.gradient(point+t*direction), direction) < c2*np.dot(dfx, direction):
            low = t
            if np.isinf(high):
                t = 2*low
            else:
                t = (low+high)/2
        else:
            break
        if high-low < tol:
            break
    return t

# TODO:Understanding back tracking method
def backtrack_line_search(opt_problem, point, direction, alpha=0.4, beta=0.8):
    """
    Newton method is very sensitive to initial points
    Line search using back tracking method
    :param opt_problem:
    :param point:
    :param direction:
    :param alpha:
    :param beta:
    :return:
    """
    t = 1

    fx = opt_problem.obj_val(point)
    dfx = opt_problem.gradient(point)

    while opt_problem.obj_val(point + t*direction) > fx + alpha * t * np.dot(dfx, direction):
        t *= beta
    return t


# TODO:Optimize the whole Optimizer structure
class Optimizer:
    def __init__(self, opt_problem, initial_point,
                 max_iteration=500,
                 step_size_chooser='wolfe',
                 direction='quasi-newton',
                 xtol=1e-15,
                 ftol=1e-10,
                 gtol=1e-10,
                 verbose=True,
                 store_x=True,
                 store_f=True,
                 store_g=True,
                 store_d=True,
                 store_t=True):
        self.opt_problem = opt_problem
        # Initial Point
        self.init_point = initial_point
        # Max iteration
        self.max_iteration = max_iteration
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose
        self.direction_chooser = direction
        self.step_size_chooser = step_size_chooser
        # Whether to store x or not
        self.store_x = store_x
        # Whether to store function value or not
        self.store_f = store_f
        # Whether to store gradient or not
        self.store_g = store_g
        # Whether to store direction or not
        self.store_d = store_d
        # Whether to store step size or not
        self.store_t = store_t
        # Dimension
        self.dim = opt_problem.dimension()
        f_all = x_all = g_all = d_all = t_all = np.empty((0,))
        if store_f:
            f_all = np.zeros((max_iteration+1,))
            f_all[0] = opt_problem.obj_val(initial_point)
        if store_x:
            x_all = np.zeros((self.dim, max_iteration+1))
            x_all[:, 0] = initial_point
        if store_g:
            g_all = np.zeros((self.dim, max_iteration))
        if store_d:
            d_all = np.zeros((self.dim, max_iteration))
        if store_t:
            t_all = np.zeros((max_iteration,))
        x_prev = np.zeros((self.dim,))
        h = h_prev = np.identity(self.dim)
        self.ws = WorkingSet(
            initial_point,
            x_prev,
            self.opt_problem.gradient(initial_point),
            self.opt_problem.gradient(x_prev),
            h,
            h_prev,
            opt_problem.obj_val(initial_point),
            opt_problem.obj_val(x_prev),
            np.zeros((self.dim,), dtype='float'),
            0,
            0,
            f_all,
            x_all,
            g_all,
            d_all,
            t_all
        )

    def optimize(self):
        """
        Entrance point for the optimization process
        :return:
        """
        # ----------------------------------------
        # loop
        # ----------------------------------------
        while not self.stop():
            self.ws.iteration += 1
            delta_x = self.ws.x - self.ws.x_prev
            delta_gradient = self.opt_problem.gradient(self.ws.x)-self.opt_problem.gradient(self.ws.x_prev)
            self.ws.x_prev = self.ws.x
            # compute gradient
            self.ws.g_prev = self.ws.g
            self.ws.f_prev = self.ws.f
            self.ws.g = self.opt_problem.gradient(self.ws.x)
            self.ws.h_prev = self.ws.h

            # compute direction
            if self.direction_chooser == 'gradient':
                self.ws.direction = -self.ws.g
                if self.step_size_chooser == 'wolfe':
                    # line search
                    self.ws.step_size = wolfe_line_search(self.opt_problem, self.ws.x, self.ws.direction)
                else:
                    raise ValueError('Given step size chooser not supported')
                gain = self.ws.step_size*self.ws.direction
            elif self.direction_chooser == 'quasi-newton':
                self.ws.h = self.opt_problem.calc_quasi_hessian(delta_x, delta_gradient, self.ws.h_prev)
                self.ws.direction = -np.dot(self.ws.h, self.ws.g)
                if self.step_size_chooser == 'wolfe':
                    self.ws.step_size = wolfe_line_search(self.opt_problem, self.ws.x, self.ws.direction)
                else:
                    raise ValueError('Given step size chooser not supported')
                gain = self.ws.step_size*self.ws.direction
            elif self.direction_chooser == 'newton':
                self.ws.direction = self.opt_problem.solve_hessian(self.ws.x, self.ws.g)
                if self.step_size_chooser == 'damped':
                    self.ws.step_size = backtrack_line_search(self.opt_problem, self.ws.x, self.ws.direction)
                elif self.step_size_chooser == 'const':
                    self.ws.step_size = 1.0
                else:
                    raise ValueError('Given step size chooser not supported')
                gain = self.ws.step_size*self.ws.direction
            elif self.direction_chooser == 'ils':
                gain = self.opt_problem.calc_gain_ils(self.ws.x)
            else:
                raise ValueError("Given direction calculator not supported")

            # move
            self.ws.x = self.ws.x_prev + gain
            self.ws.f = self.opt_problem.obj_val(self.ws.x)

            # traces
            if self.verbose:
                print "Iteration:%4d" % self.ws.iteration
                print self.ws.x
                print "Function Value:%.6f" % self.ws.f
                print "Decreased in this iteration:%.6f" % (self.ws.f_prev-self.ws.f)

            # recording for debugging / visualization
            if self.store_x:
                self.ws.x_all[:, self.ws.iteration] = self.ws.x
            if self.store_g:
                self.ws.g_all[:, self.ws.iteration-1] = self.ws.g
            if self.store_d:
                self.ws.d_all[:, self.ws.iteration-1] = self.ws.direction
            if self.store_t:
                self.ws.t_all[self.ws.iteration-1] = self.ws.step_size
            if self.store_f:
                self.ws.f_all[self.ws.iteration] = self.ws.f
        return self.ws

    def stop(self):
        """
        Determine whether to stop or not
        :return:
        """
        stop = False
        if self.ws.iteration >= self.max_iteration:
            stop = True
        # stop condition
        if abs(self.ws.f_prev - self.ws.f) < self.ftol:
            stop = True
        if lg.norm(self.ws.x-self.ws.x_prev) < self.xtol:
            stop = True
        if self.ws.iteration > 1:
            if lg.norm(self.ws.g) < self.gtol:
                stop = True
        return stop


class WorkingSet:
    def __init__(self, x, x_prev, g, g_prev, h, h_prev, f, f_prev, direction, step_size, iteration,
                 f_all, x_all, g_all, d_all, t_all):
        """
        Working set,stores everything during optimization
        """
        # Current point
        self.x = x
        # Previous point
        self.x_prev = x_prev
        # Current gradient
        self.g = g
        # previous gradient
        self.g_prev = g_prev
        # current quasi hessian
        self.h = h
        # previous hessian
        self.h_prev = h_prev
        # current function value
        self.f = f
        # previous function value
        self.f_prev = f_prev
        # current direction
        self.direction = direction
        # step size
        self.step_size = step_size
        # Iteration number
        self.iteration = iteration
        # All function values
        self.f_all = f_all
        # All point values
        self.x_all = x_all
        # All gradient values
        self.g_all = g_all
        # All direction values
        self.d_all = d_all
        # All step size values
        self.t_all = t_all

if __name__ == '__main__':
    sample = np.array([[28976.985200060444, 12095.921247416274, -0.20460391230838582],
                       [21399.224900479396, 28005.663602853492, 1.1179398758981125],
                       [23359.266675802537, 28675.878103997853, 1.0212927844392057]])
    biased_sample = np.array([[14472.90041987611, 14852.459224022326, 0.27346669323819384],
                              [11638.620939799031, 10960.318365429848, 0.8792708847729779],
                              [12567.89408221255, 10702.533698252686, 1.0541683171468343]])
    precision = 1/(0.05**2)
    angleMeasurement = AngleMeasurement(sample, precision)
    initial_point = np.array([10345.17500173, 10789.7906996])
    newton_initial = np.array([13000, 13000])
    # TODO:Initialize the initial point using the first two measurements of the platform
    optimizer = Optimizer(angleMeasurement,
                          newton_initial,
                          direction='newton',
                          step_size_chooser='damped')
    print(optimizer.optimize().x)