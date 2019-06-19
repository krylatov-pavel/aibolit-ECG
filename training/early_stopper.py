class EarlyStopper(object):
    def __init__(self, wait_steps, metric_value=None, steps_since_last_improvemet=None):
        self._wait_steps = wait_steps
        self._best_metric_value = metric_value or 0.0
        self._steps_since_last_improvemet = steps_since_last_improvemet or 0
    
    def step(self, metric_value):
        if metric_value > self._best_metric_value:
            self._best_metric_value = metric_value
            self._steps_since_last_improvemet = 0
        else:
            self._steps_since_last_improvemet += 1
    
    @property
    def stop(self):
        return self._steps_since_last_improvemet >= self._wait_steps

    @property
    def best_metric_value(self):
        return self._best_metric_value

    @property
    def steps_since_last_improvemet(self):
        return self._steps_since_last_improvemet