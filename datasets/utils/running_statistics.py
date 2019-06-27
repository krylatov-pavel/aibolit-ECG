import numpy as np
import math

class RunningStatistics(object):
    def __init__(self):
        self._n = 0
        self._mean_avg = 0
        self._std_avg = 0

    def next(self, x):
        self._n += 1
        
        x = np.mean(np.array(x))

        mean_prev = self._mean_avg
        self._mean_avg = self._mean_avg + (x - self._mean_avg) / self._n

        self._std_avg = self._std_avg + (x - mean_prev)*(x - self._mean_avg)

    @property
    def mean(self):
        return self._mean_avg

    @property
    def variance(self):
        return math.sqrt(self._std_avg / (self._n - 1))