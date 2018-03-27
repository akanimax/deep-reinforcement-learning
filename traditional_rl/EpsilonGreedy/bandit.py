""" Bandit module for performing the experiment
"""

import numpy as np


class Bandit:
    """ Bandit class. This implements actually one machine arm
    """
    def __init__(self, m):
        """ initialization """
        self.m = m
        self.mean = 0  # for storing the current estimate of the mean
        self.N = 0  # number of experiments run till now.

    def pull(self):
        """ pulling the arm returns some reward """
        return np.random.randn() + self.m  # reward is a sample from a
        # unit gaussian across the mean

    def update(self, x):
        """ update the mean estimate based on the received reward"""
        self.N += 1  # increment the value of the counter
        self.mean = ((1 - (1 / self.N)) * self.mean) + ((1 / self.N) * x)
