__author__ = 'qingyuanxingsi'

import math
import random

"""
A simple tool for generating test data
for angle-only measurement scenario
@version:1.0
@date:2014-12-2
"""


class DataGenerator:
    """
    Observed data:z = tan-1((y-y0)/(x-x0))+omega
    where omega conforms to a norm distribution
    with mean zero and variance sigma^2
    """
    def __init__(self, sigma=0.05, x=15000, y=15000,
                 low=10000, high=15000):
        """
        Initialization method
        :param sigma:
        :param x:
        :param y:
        :param low:
        :param high:
        :return:
        """
        # x position of the target
        self.target_x = x
        # y position of the target
        self.target_y = y
        # variance sigma^2
        self.variance = sigma**2
        self.low = low
        self.high = high

    def generate_data(self, count=3):
        """
        Return generated test data
        :return:
        """
        result = []
        for i in range(count):
            new_x = random.uniform(self.low, self.high)
            new_y = random.uniform(self.low, self.high)
            new_measure = math.atan((new_y-self.target_y)/(new_x-self.target_x))+random.gauss(0, self.variance)
            result.append((new_x, new_y, new_measure))
        return result

if __name__ == '__main__':
    generator = DataGenerator()
    sample = generator.generate_data()
    print(sample)
