import os
import torch
import torch.utils.data
import time
from torchvision import transforms
import training.checkpoint as checkpoint
import training.metrics.stats as stats
from training.spec import TrainSpec, EvalSpec
from training.model import Model
from utils.dirs import create_dirs
from utils.helpers import get_class
from models.common.ensemble import Ensemble
from datasets.common.dataset import Dataset

TRAIN = "train"
EVAL = "eval"

def squeeze(x):
    return torch.squeeze(x, dim=0)

class Experiment():
    def __init__(self, config, model_dir):
        self._dataset_generator = get_class(config.dataset.dataset_generator)(config.dataset.params, config.dataset.sources)
        self._examples_provider = get_class(config.dataset.examples_provider)

        self._config = config
        self._model_dir = model_dir
        self._k = len(config.dataset.params.split_ratio)
        self._seed = config.dataset.params.get("seed") or 0

        self._optimizer = config.model.hparams.optimizer.type
        self._optimizer_params = config.model.hparams.optimizer.params

        self._class_num = len(config.dataset.params.class_settings)
        self._label_map = { lbl: c.label_map for lbl, c in config.dataset.params.class_settings.items() }
        self._normalize_input = config.dataset.params.normalize_input

        self._iteration = config.iteration
        self._max_epochs = config.max_epochs
        self._keep_n_checkpoints = config.keep_n_checkpoints
        self._early_stopper_params = config.early_stopper_params
        self._eval_scheduler_params = config.eval_scheduler_params
        self._lr_scheduler_params = config.lr_scheduler_params

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
            net = self._load_net(self._model_dir, checkpoint_index=checkpoint)
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

        examples = self._examples_provider(
            folders=self._dataset_generator.train_set_path(0),
            label_map=self._label_map,
            seed=self._seed
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

        if self._normalize_input:
            mean, std = self._dataset_generator.stats
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean], std=[std]),
                transforms.Lambda(squeeze)
            ])
        else:
            transform = None

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        last_checkpoint = checkpoint.last(model_dir)
        if last_checkpoint:
            model = Model.restore(net, model_dir, last_checkpoint, device=device)
        else:
            model = Model(net, model_dir, device=device)

        train_examples = self._examples_provider(
            folders=self._dataset_generator.train_set_path(fold_num),
            label_map=self._label_map,
            seed=self._seed
        )

        train_spec = TrainSpec(
            max_epochs=self._max_epochs,
            dataset=Dataset(train_examples, transform=transform),
            batch_size=32,
            optimizer_type= self._optimizer,
            optimizer_params=self._optimizer_params,
            lr_scheduler_params=self._lr_scheduler_params,
            early_stopper_params=self._early_stopper_params
        )

        eval_examples = self._examples_provider(
            folders=self._dataset_generator.eval_set_path(fold_num),
            label_map=self._label_map,
            equalize_labels=True,
            seed=self._seed
        )

        eval_spec = EvalSpec(
            class_num=self._class_num,
            dataset=Dataset(eval_examples, transform=transform),
            batch_size=100,
            eval_scheduler_params=self._eval_scheduler_params,
            class_map={value: key for key, value in self._label_map.items()},
            keep_n_checkpoints=self._keep_n_checkpoints
        )
        
        start = time.perf_counter()
        try:
            model.train_and_evaluate(train_spec, eval_spec)
        finally:
            train_examples.close()
            eval_examples.close()
        end = time.perf_counter()
        print("execution time: {}s".format(end - start))

    def _load_net(self, model_dir, checkpoint_index=None):
        net = get_class(self._config.model.name)(self._config)

        checkpoint_index = checkpoint_index or checkpoint.last(model_dir)
        _, model_state, _, _ = checkpoint.load(model_dir, checkpoint_index)
        net.load_state_dict(model_state)

        net.eval()

        return net