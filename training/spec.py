class TrainSpec(object):
    def __init__(self, max_epochs, dataset, batch_size=None, optimizer_type=None, optimizer_params=None,
        max_to_keep=None, early_stopper_params=None, lr_scheduler_params=None):
        self.max_epochs = max_epochs
        self.dataset = dataset
        self.batch_size = batch_size or 32
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params
        self.early_stopper_params = early_stopper_params
        self.lr_scheduler_params = lr_scheduler_params

class EvalSpec(object):
    def __init__(self, class_num, dataset, batch_size=None, eval_scheduler_params=None, class_map=None, keep_n_checkpoints=None):
        self.dataset = dataset
        self.batch_size = batch_size or 100
        self.class_num = class_num
        self.class_map = class_map
        self.eval_scheduler_params = eval_scheduler_params
        self.keep_n_checkpoints = keep_n_checkpoints