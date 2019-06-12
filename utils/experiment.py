import os
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from utils.helpers import get_class
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
        self._train_batch_size = self._config.model.hparams["train_batch_size"]
        self._eval_batch_size = self._config.model.hparams["eval_batch_size"]
        self._num_epochs = self._config.model.hparams["num_epochs"]
        self._learning_rate = self._config.model.hparams["learning_rate"]

        #regular experiment
        if self._k == 2:
            self._train_model(self._model_dir)

        #k-fold crossvalidation 
        elif self._k > 2:
            for i in range(self._k):
                directory = os.path.join(self._model_dir, "fold_{}".format(i))
                self._train_model(directory, i)
            #plot_metrics(self._model_dir)

    def evaluate_accuracy(self, checkpoint_num=None):
        cm = self.confusion_matrix(checkpoint_num)

        tp = cm.diagonal().sum()
        total = cm.sum()
        accuracy = tp / total

        print("Evaluated accuracy: ", accuracy)

    def confusion_matrix(self, checkpoint_num=None):
        if self._k == 2:
            cm = self._evaluate_model(self._config, self._model_dir, checkpoint_num=checkpoint_num)
        if self._k > 2:
            cm = np.zeros((self._config.model.hparams.class_num, self._config.model.hparams.class_num))
            for i in range(self._k):
                directory = os.path.join(self._model_dir, "fold_{}".format(i))
                cm += self._evaluate_model(self._config, directory, i, checkpoint_num=checkpoint_num)
        
        return cm

    def _evaluate_model(self, config, model_dir, fold_num=None, checkpoint_num=None):
            model = get_class(config.model.name)(config.model.hparams, config.dataset.params)
            
            dataset = get_class(config.dataset.name)(config.dataset.params)

            x, labels = dataset.get_eval_examples(fold_num)
            y = [config.dataset.params.label_map[label] for label in labels]

            input_fn = tf.estimator.inputs.numpy_input_fn(np.array(x, dtype="float32") , shuffle=False)

            estimator = tf.estimator.Estimator(
                model_fn=model.build_model_fn(), 
                model_dir=model_dir
            )

            checkpoint_path = os.path.join(model_dir, "model.ckpt-{}".format(checkpoint_num)) if checkpoint_num else None 
            predictions = list(estimator.predict(input_fn, checkpoint_path=checkpoint_path))
            predictions = [p["class_ids"][0] for p in predictions]

            cm = np.zeros((config.model.hparams.class_num, config.model.hparams.class_num))

            for predicted, actual  in zip(predictions, y):
                cm[predicted][actual] += 1

            return cm

    def _train_model(self, model_dir, fold_num=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model = get_class(self._config.model.name)(self._config)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = self._learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        train_set = Dataset(self._examples_provider, self._get_fold_nums(TRAIN, fold_num))
        train_loader = data.DataLoader(train_set, batch_size=self._train_batch_size, shuffle=True)

        for epoch in range(self._num_epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                inputs, labels = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()
                predictions = model(inputs)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss

                if i % 20 == 19:
                    print("epoch {}\titeration {}\tloss {:.3f}".format(epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0

        print("training complete\nevaluating accuracy...")

        model.to("cpu")
        model.eval()

        eval_set = Dataset(self._examples_provider, self._get_fold_nums(EVAL, fold_num))
        eval_loader = data.DataLoader(eval_set, batch_size=self._eval_batch_size, shuffle=True, num_workers=2)

        tp = 0
        total = 0
        for inputs, labels in eval_loader:
            with torch.no_grad():
                predictions = model(inputs)
                _, predictions = torch.max(predictions, 1)
                total += len(labels)
                tp += (predictions == labels).sum().item()

        print("accuracy {:.3f}".format((tp / total) * 100.0))
        return

    def _get_fold_nums(self, mode, fold_num=None):
        folds = list(range(self._k))
        eval_fold = fold_num if fold_num != None else 1

        if mode == TRAIN:
            return [f for f in folds if  f != eval_fold]
        else:
            return [eval_fold]