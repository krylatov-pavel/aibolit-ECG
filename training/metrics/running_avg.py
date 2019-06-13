class RunningAvg(object):
    def __init__(self, initail_val):
        self._epoch_avg = initail_val
        self._iteration = 0

    @property
    def avg(self):
        return self._epoch_avg

    def next_iteration(self, val):
        self._iteration += 1
        self._epoch_avg = self._epoch_avg + (val - self._epoch_avg) / self._iteration

    def next_epoch(self):
        self._epoch_avg = 0.0
        self._iteration = 0