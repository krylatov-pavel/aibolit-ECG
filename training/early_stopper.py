class EarlyStopper(object):
    def __init__(self, patience, threshold=1e-4):
        self._patience = patience
        self._threshold = threshold
        self._best_value = 0
        self._best_epoch = 0
        self._curr_epoch = 0

    def re_init(self, patience, threshold=1e-4):
        self._patience = patience
        self._threshold = threshold
    
    @property
    def stop(self):
        return self._curr_epoch - self._best_epoch > self._patience

    @property
    def best_epoch(self):
        return self._best_epoch

    def step(self, metric_value, epoch=None):
        self._curr_epoch = epoch or (self._curr_epoch + 1)
        if self._is_better(metric_value):
            self._best_value = metric_value
            self._best_epoch = self._curr_epoch

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    def _is_better(self, value):
        rel_epsilon = 1. + self._threshold
        return value > self._best_value * rel_epsilon
    
    