class TrainSpec(object):
    def __init__(self, max_epochs, dataset, batch_size=None, optimizer_type=None, optimizer_params=None,
        max_to_keep=None, wait_improvement_n_evals=None):
        self.max_epochs = max_epochs
        self.dataset = dataset
        self.batch_size = batch_size or 32
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params
        self.wait_improvement_n_evals = wait_improvement_n_evals

class EvalSpec(object):
    def __init__(self, class_num, dataset, batch_size=None, every_n_epochs=None, class_map=None, keep_n_checkpoints=None):
        self.dataset = dataset
        self.batch_size = batch_size or 100
        self.class_num = class_num
        self.every_n_epochs = every_n_epochs
        self.class_map = class_map
        self.keep_n_checkpoints = keep_n_checkpoints