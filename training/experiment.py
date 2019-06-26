import os
import torch
import torch.utils.data
import training.checkpoint as checkpoint
import training.metrics.stats as stats
from training.spec import TrainSpec, EvalSpec
from training.model import Model
from utils.dirs import create_dirs
from utils.helpers import get_class
from models.MIT.ensemble import Ensemble
from datasets.common.dataset import Dataset
from datasets.common.examples_provider import ExamplesProvider

TRAIN = "train"
EVAL = "eval"

class Experiment():
    def __init__(self, config, model_dir):
        self._dataset_provider = get_class(config.dataset.dataset_provider)(config.dataset.params)
        self._file_provider = get_class(config.dataset.file_provider)()

        self._config = config
        self._model_dir = model_dir
        self._name = config.model.experiment
        self._k = len(config.dataset.params.split_ratio)
        self._iteration = config["iteration"]
        self._max_epochs = config.model["max_epochs"]
        self._learning_rate = config.model.hparams["learning_rate"]
        self._class_num = len(config.dataset.params["label_map"])
        self._label_map = config.dataset.params["label_map"]
        self._max_to_keep = config["max_to_keep"]

    def run(self):
        #regular experiment
        if self._k == 2:
            self._train_model(self._model_dir)

        #k-fold crossvalidation 
        elif self._k > 2:
            for i in range(self._k):
                directory = os.path.join(self._model_dir, "fold_{}".format(i))
                self._train_model(directory, i)

    def export(self, checkpoint=None):
        if self._k == 2:
            net = self._load_net(self._model_dir, checkpoint_index=checkpoint[0])
        elif self._k > 2:
            models = {}
            for i in range(self._k):
                model_dir = os.path.join(self._model_dir, "fold_{}".format(i))
                model_name = "model{}".format(i)
                checkpoint_index = checkpoint[i] if checkpoint else None
                models[model_name] = self._load_net(model_dir, checkpoint_index=checkpoint_index)

            net = Ensemble(**models)
        else:
            raise ValueError("invalid k")

        examples = ExamplesProvider(
            folders=self._dataset_provider.test_set_path(),
            file_reader=self._file_provider,
            label_map=self._label_map
        )

        dataset = Dataset(examples)
        data_loader = torch.utils.data.DataLoader(dataset)
        x, _ = iter(data_loader).next()

        net.eval()
        net.to("cpu")
        with torch.no_grad():
            torch.onnx.export(net, x, os.path.join(self._model_dir, "model.onnx"))

    def plot_metrics(self):
        stats.plot_metrics(self._model_dir, self._k)

    def _train_model(self, model_dir, fold_num=None):
        create_dirs([model_dir])

        net = get_class(self._config.model.name)(self._config)

        train_examples = ExamplesProvider(
            folders=self._dataset_provider.train_set_path(fold_num),
            file_reader=self._file_provider,
            label_map=self._label_map,
        )

        train_spec = TrainSpec(
            max_epochs=self._max_epochs,
            dataset=Dataset(train_examples),
            batch_size=32,
            optimizer_type="adam",
            optimizer_params={
                "lr": self._learning_rate
            },
            max_to_keep=self._max_to_keep
        )

        eval_examples = ExamplesProvider(
            folders=self._dataset_provider.eval_set_path(fold_num),
            file_reader=self._file_provider,
            label_map=self._label_map,
        )

        eval_spec = EvalSpec(
            class_num=self._class_num,
            dataset=Dataset(eval_examples),
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

    def _load_net(self, model_dir, checkpoint_index=None):
        net = get_class(self._config.model.name)(self._config)

        checkpoint_index = checkpoint_index or checkpoint.last(model_dir)
        _, model_state, _, _ = checkpoint.load(model_dir, checkpoint_index)
        net.load_state_dict(model_state)

        net.eval()

        return net