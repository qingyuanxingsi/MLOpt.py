__author__ = 'qingyuanxingsi'

'''
Implement the OptProblem abstract class
Specifically
the angle-only measurement problem
@version:1.0
@date:2014-12-2
'''
from mlopt.opt.opt_problem import OptProblem
import math
import numpy as np
import numpy.linalg as lg


def error(measurement, dat):
    return measurement[2]-math.atan((measurement[1]-dat[1])/(measurement[0]-dat[0]))


class AngleMeasurement(OptProblem):
    def __init__(self, sample, precision):
        self.sample = sample
        length = len(self.sample)
        self.precision = precision
        self.precision_matrix = np.identity(length)*precision

    def dimension(self):
        """
        Return the dimension of the problem
        :return:
        """
        return 2

    def support_second_order(self):
        return True

    def obj_val(self, point):
        """
        Evaluate the function at given point
        :param point:
        :return:
        """
        return sum([error(measurement, point)**2 for measurement in self.sample])

    def gradient(self, point):
        """
        Calculate the gradient at given point
        :param point:
        :return:
        """
        gradient = np.zeros((self.dimension(),), dtype='float')
        for j in range(len(self.sample)):
            delta_y = point[1]-self.sample[j][1]
            delta_x = point[0]-self.sample[j][0]
            partial_x = -delta_y/(delta_x**2+delta_y**2)
            partial_y = delta_x/(delta_x**2+delta_y**2)
            gradient[0] -= error(self.sample[j], point)*partial_x
            gradient[1] -= error(self.sample[j], point)*partial_y
        return gradient

    # this function will only be called in 2nd order methods, it returns the solution
    # of the equation: H(x) * z = d, where H(x) is the Hessian matrix at point x
    def solve_hessian(self, point, direction):
        """
        Solve the hessian
        """
        hessian = np.zeros((self.dimension(), self.dimension()))
        for j in range(len(self.sample)):
            delta_y = point[1]-self.sample[j][1]
            delta_x = point[0]-self.sample[j][0]
            partial_x = -delta_y/(delta_x**2+delta_y**2)
            partial_y = delta_x/(delta_x**2+delta_y**2)
            partial_xx = 2*delta_x*delta_y/((delta_x**2+delta_y**2)**2)
            partial_xy = (delta_y**2-delta_x**2)/((delta_x**2+delta_y**2)**2)
            partial_yy = -partial_xx
            hessian[0][0] += partial_x**2 - error(self.sample[j], point)*partial_xx
            hessian[0][1] += partial_x*partial_y-error(self.sample[j], point)*partial_xy
            hessian[1][0] = hessian[0][1]
            hessian[1][1] += partial_y**2-error(self.sample[j], point)*partial_yy
        return -np.dot(lg.inv(hessian), direction)

    def support_exact_line_search(self):
        """
        Whether exact line search is supported
        """
        return False

    def exact_line_search(self, point, direction):
        """
        Exact line search
        """
        pass

    @property
    def support_ils(self):
        """
        Return whether iterated least square is supported
        :return:
        """
        return True

    def calc_gain_ils(self, point):
        """
        Calculate the gain for the Iterated Least Square method
        """
        jacobian_matrix = np.zeros((len(self.sample), self.dimension()))
        print(jacobian_matrix.shape)
        residual_error = np.zeros((len(self.sample,)))
        for j in range(len(self.sample)):
            delta_y = point[1]-self.sample[j][1]
            delta_x = point[0]-self.sample[j][0]
            jacobian_matrix[j][0] = -delta_y/(delta_x**2+delta_y**2)
            jacobian_matrix[j][1] = delta_x/(delta_x**2+delta_y**2)
            residual_error[j] = error(self.sample[j], point)
        print(jacobian_matrix)
        a = np.dot(np.transpose(jacobian_matrix), self.precision_matrix)
        b = lg.inv(np.dot(a, jacobian_matrix))
        return np.dot(np.dot(b, a), residual_error)

    def calc_quasi_hessian(self, delta_x, delta_gradient, prev_h):
        """
        Calculate a hessian approximation
        for the quasi-newton method
        H_{k+1} = (I-\rno_ks_ky_k')H_k(I-\rno_ky_ks_k')+\rno_ks_ks_k'
        where \rno_k = \frac{1}{y_k's_k}
        :param delta_x:
        :param delta_gradient:
        :param prev_h:
        :return:
        """
        p = 1/np.dot(delta_gradient, delta_x)
        identity = np.identity(self.dimension())
        left = identity - p*np.outer(delta_x, delta_gradient)
        middle = identity - p*np.outer(delta_gradient, delta_x)
        right = p*np.outer(delta_gradient, delta_gradient)
        return np.dot(np.dot(left, prev_h), middle)+right

if __name__ == "__main__":
    sample = np.array([[29051.299495709056, 24222.779543258082, 0.43637999711474],
                       [27508.42690581429, 11212.36692079506, -0.8636104281953914],
                       [24209.772053781227, 26696.11792040088, 1.009620767436036]])
    curr_estimation = np.array([20000.0, 20000.0])
    precision = 1/(0.05**2)
    angleMeasurement = AngleMeasurement(sample, precision)
    gradient = angleMeasurement.gradient(curr_estimation)
    print(gradient)
    print(angleMeasurement.obj_val(curr_estimation))
    print(angleMeasurement.solve_hessian(curr_estimation, gradient))
