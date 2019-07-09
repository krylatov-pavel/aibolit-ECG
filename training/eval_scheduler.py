class EvalScheduler(object):
    def __init__(self, interval, decrease_step=None, decrease_timeout=None):
        self._initial_interval = interval
        self._interval = interval
        self._decrease_step = decrease_step
        self._decrease_timeout = decrease_timeout 
        self._curr_epoch = 0
        self._last_eval_epoch = 0

    def re_init(self, interval, decrease_step=None, decrease_timeout=None):
        self._initial_interval = interval
        self._interval = interval
        self._decrease_step = decrease_step
        self._decrease_timeout = decrease_timeout
        self.step(epoch=self._curr_epoch)

    @property
    def eval(self):
        return (self._curr_epoch == self._last_eval_epoch) \
            or (self._curr_epoch - self._last_eval_epoch >= self._interval)

    def step(self, epoch=None):
        self._curr_epoch = epoch or (self._curr_epoch + 1)
        
        if self._decrease_timeout and self._curr_epoch >= self._decrease_timeout:
            self._interval = max(self._initial_interval - self._decrease_step * (self._curr_epoch // self._decrease_timeout), 1)
        
        if self._curr_epoch - self._last_eval_epoch == self._interval:
            self._last_eval_epoch = self._curr_epoch

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)