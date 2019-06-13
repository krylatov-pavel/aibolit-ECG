import os
from utils.dirs import create_dirs
from utils.helpers import get_class
from training.model import Model
from datasets.base.dataset import Dataset

TRAIN = "train"
EVAL = "eval"

class Experiment():
    def __init__(self, config, model_dir):
        self._config = config
        self._model_dir = model_dir
        self._k = len(config.dataset.params.split_ratio)
        self._examples_provider = get_class(config.dataset.provider)(config.dataset.params)

    def run(self):
        self._num_epochs = self._config.model.hparams["num_epochs"]
        self._learning_rate = self._config.model.hparams["learning_rate"]
        self._class_num = len(self._config.dataset.params["label_map"])

        #regular experiment
        if self._k == 2:
            self._train_model(self._model_dir)

        #k-fold crossvalidation 
        elif self._k > 2:
            for i in range(self._k):
                directory = os.path.join(self._model_dir, "fold_{}".format(i))
                self._train_model(directory, i)

    def plot_metrics(self):
        pass

    def _train_model(self, model_dir, fold_num=None):
        create_dirs([model_dir])

        net = get_class(self._config.model.name)(self._config)
        train_set = Dataset(self._examples_provider, self._get_fold_nums(TRAIN, fold_num))
        eval_set = Dataset(self._examples_provider, self._get_fold_nums(EVAL, fold_num))

        optimizer_params = {
            "lr": self._learning_rate
        }

        model = Model(net, model_dir, self._class_num)
        model.train_and_evaluate(self._num_epochs, train_set, optimizer_params, eval_set)

    def _get_fold_nums(self, mode, fold_num=None):
        folds = list(range(self._k))
        eval_fold = fold_num if fold_num != None else 1

        if mode == TRAIN:
            return [f for f in folds if  f != eval_fold]
        else:
            return [eval_fold]