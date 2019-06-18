import os
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import training.checkpoint as checkpoint
from training.spec import TrainSpec, EvalSpec
from training.model import Model
from training.metrics.logger import Logger
from utils.dirs import create_dirs
from utils.helpers import get_class
from models.MIT.ensemble import Ensemble
from datasets.base.dataset import Dataset

TRAIN = "train"
EVAL = "eval"

class Experiment():
    def __init__(self, config, model_dir):
        self._config = config
        self._model_dir = model_dir
        self._name = config.model.experiment
        self._k = len(config.dataset.params.split_ratio)
        self._examples_provider = get_class(config.dataset.provider)(config.dataset.params)
        self._iteration = self._config["iteration"]
        self._num_epochs = self._config.model.hparams["num_epochs"]
        self._learning_rate = self._config.model.hparams["learning_rate"]
        self._class_num = len(self._config.dataset.params["label_map"])
        self._label_map = self._config.dataset.params["label_map"]

    def run(self):
        #regular experiment
        if self._k == 2:
            self._train_model(self._model_dir)

        #k-fold crossvalidation 
        elif self._k > 2:
            for i in range(self._k):
                directory = os.path.join(self._model_dir, "fold_{}".format(i))
                self._train_model(directory, i)

    def plot_metrics(self):
        if self._k == 2:
            logs = Logger(self._model_dir).read()
        elif self._k > 2:
            logs = pd.DataFrame(data=None, index=None, columns=None)
            for i in range(self._k):
                log_dir = os.path.join(self._model_dir, "fold_{}".format(i))
                log = Logger(log_dir).read(fold_num=i)
                logs = logs.append(log, ignore_index=True)

        Logger.plot(logs, os.path.join(self._model_dir, "plot.png"))

    def export(self, checkpoint=None, use_best=False):
        if self._k == 2:
            net = self._load_net(self._model_dir)
        elif self._k > 2:
            models = {}
            for i in range(self._k):
                model_dir = os.path.join(self._model_dir, "fold_{}".format(i))
                model_name = "model{}".format(i)
                models[model_name] = self._load_net(model_dir, checkpoint_index=checkpoint, use_best=use_best)

            net = Ensemble(**models)
        else:
            raise ValueError("invalid k")

        dataset = Dataset(self._examples_provider, self._get_fold_nums(EVAL, 1))
        data_loader = torch.utils.data.DataLoader(dataset)
        x, _ = iter(data_loader).next()

        net.eval()
        net.to("cpu")
        with torch.no_grad():
            torch.onnx.export(net, x, os.path.join(self._model_dir, "model.onnx"))

    def _train_model(self, model_dir, fold_num=None):
        create_dirs([model_dir])

        net = get_class(self._config.model.name)(self._config)

        train_spec = TrainSpec(
            max_epochs=self._num_epochs,
            dataset=Dataset(self._examples_provider, self._get_fold_nums(TRAIN, fold_num)),
            batch_size=32,
            optimizer_type="adam",
            optimizer_params={
                "lr": self._learning_rate
            }
        )

        eval_spec = EvalSpec(
            class_num=self._class_num,
            dataset=Dataset(self._examples_provider, self._get_fold_nums(EVAL, fold_num)),
            batch_size=100,
            every_n_epochs=5,
            class_map={value: key for key, value in self._label_map.items()}
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        last_checkpoint = checkpoint.last(model_dir)
        if last_checkpoint:
            model = Model.restore(net, model_dir, last_checkpoint, device=device)
        else:
            model = Model(net, model_dir, device=device)

        model.train_and_evaluate(train_spec, eval_spec)

    def _load_net(self, model_dir, checkpoint_index=None, use_best=False):
        net = get_class(self._config.model.name)(self._config)

        if use_best:
            raise NotImplementedError()
        else:
            checkpoint_index = checkpoint_index or checkpoint.last(model_dir)
            _, model_state, _, _ = checkpoint.load(model_dir, checkpoint)
            net.load_state_dict(model_state)

        net.eval()

        return net

    def _get_fold_nums(self, mode, fold_num=None):
        folds = list(range(self._k))
        eval_fold = fold_num if fold_num != None else 1

        if mode == TRAIN:
            return [f for f in folds if  f != eval_fold]
        else:
            return [eval_fold]