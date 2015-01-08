__author__ = 'qingyuanxingsi'

from abc import abstractmethod

"""
An abstract Optimization Problem
"""


class OptProblem:
    def __init__(self):
        pass

    @abstractmethod
    def dimension(self):
        """
        Return the dimension of the problem
        :return:
        """
        pass

    @abstractmethod
    def obj_val(self, point):
        """
        Evaluate the function at given point
        :param point:
        :return:
        """
        pass

    @abstractmethod
    def gradient(self, point):
        """
        Calculate the gradient at given point
        :param point:
        :return:
        """
        pass

    @abstractmethod
    def support_second_order(self):
        pass

    # This function will only be called in 2nd order methods, it returns the solution
    # of the equation: H(x) * z = d, where H(x) is the Hessian matrix at point x
    @abstractmethod
    def solve_hessian(self, point, direction):
        """
        Solve the hessian
        """
        pass

    @abstractmethod
    def support_exact_line_search(self):
        """
        Whether exact line search is supported
        """
        pass

    @abstractmethod
    def exact_line_search(self, point, direction):
        """
        Exact line search
        """
        pass

    def support_ils(self):
        """
        Return whether iterated least square is supported
        :return:
        """
        return True

    @abstractmethod
    def calc_gain_ils(self, point):
        """
        Calculate the gain for the Iterated Least Square method
        """
        pass

    @abstractmethod
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
        pass
