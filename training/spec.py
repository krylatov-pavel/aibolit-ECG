class TrainSpec(object):
    def __init__(self, max_epochs, dataset, batch_size=None, optimizer_type=None, optimizer_params=None):
        self.max_epochs = max_epochs
        self.dataset = dataset
        self.batch_size = batch_size or 32
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params

class EvalSpec(object):
    def __init__(self, class_num, dataset, batch_size=None, every_n_epochs=None, class_map=None):
        self.dataset = dataset
        self.batch_size = batch_size or 100
        self.class_num = class_num
        self.every_n_epochs = every_n_epochs or 5
        self.class_map = class_map