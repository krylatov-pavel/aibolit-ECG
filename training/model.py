import os
import re
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from training.metrics.running_avg import RunningAvg
from training.metrics.confusion_matrix import ConfusionMatrix
from training.metrics.logger import Logger
import training.checkpoint as checkpoint
from training.early_stopper import EarlyStopper

def create_optimizer(optimizer_type, net_parameters, optimizer_params):
    if optimizer_type == "adam":
        return optim.Adam(net_parameters, **optimizer_params)
    else:
        raise ValueError("Unknown optimizer type")

class Model(object):
    def __init__(self, net, model_dir, device=None, curr_epoch=None, optimizer=None, early_stopper=None):
        self._net = net
        self._model_dir = model_dir
        self._device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._curr_epoch = curr_epoch or 0
        self._optimizer = optimizer
        self._early_stopper = early_stopper

    @staticmethod
    def restore(net, model_dir, checkpoint_index, device=None):
        epoch, model_state, optimizer_state, params = checkpoint.load(model_dir, checkpoint_index)

        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net.load_state_dict(model_state)
        net.to(device)

        optimizer_type = params["optimizer_type"]
        optimizer_params = params["optimizer_params"]
        optimizer = create_optimizer(optimizer_type, net.parameters(), optimizer_params)
        optimizer.load_state_dict(optimizer_state)

        best_metric_value = params.get("best_metric_value")
        steps_since_last_improvemet = params.get("steps_since_last_improvemet")
        wait_improvement_n_evals = params.get("wait_improvement_n_evals")
        early_stopper = EarlyStopper(wait_improvement_n_evals, metric_value=best_metric_value, steps_since_last_improvemet=steps_since_last_improvemet)

        return Model(net, model_dir, device=device, optimizer=optimizer, curr_epoch=epoch, early_stopper=early_stopper)

    @property
    def net(self):
        return self._net

    def train_and_evaluate(self, train_spec, eval_spec):
        if self._curr_epoch < train_spec.max_epochs:
            if not self._optimizer:
                self._optimizer = create_optimizer(
                    optimizer_type=train_spec.optimizer_type,
                    optimizer_params=train_spec.optimizer_params,
                    net_parameters = self._net.parameters()
                )
            
            if not self._early_stopper:
                self._early_stopper = EarlyStopper(train_spec.wait_improvement_n_evals)

            loss_fn = nn.CrossEntropyLoss()
            loss_acm = RunningAvg(0.0)

            self._net.to(self._device)

            file_writer = Logger(log_dir=self._model_dir)
            tensorboard_writer = SummaryWriter(log_dir=os.path.join(self._model_dir, "tbruns"))
            inputs = next(iter(train_spec.dataset))[0].unsqueeze(0).to(self._device)
            tensorboard_writer.add_graph(self._net, input_to_model=inputs)

            #evaluate initial accuracy
            metrics = self.evaluate(eval_spec)
            for metric, scalar in metrics.items():
                file_writer.add_scalar(metric, scalar, 0)
                tensorboard_writer.add_scalar(metric, scalar, global_step=0)

            train_loader = data.DataLoader(train_spec.dataset, batch_size=train_spec.batch_size, shuffle=True)
            while self._curr_epoch < train_spec.max_epochs and not self._early_stopper.stop:
                self._curr_epoch += 1
                self._net.train()
                loss_acm.next_epoch()
                for batch in train_loader:
                    inputs, labels = batch[0].to(self._device), batch[1].to(self._device)

                    self._optimizer.zero_grad()
                    predictions = self._net(inputs)
                    loss = loss_fn(predictions, labels)
                    loss.backward()
                    self._optimizer.step()

                    loss_acm.next_iteration(loss.item())

                tensorboard_writer.add_scalar("traing_loss", loss_acm.avg, global_step=self._curr_epoch)
                
                if self._curr_epoch % eval_spec.every_n_epochs == 0 or self._curr_epoch == train_spec.max_epochs:
                    metrics = self.evaluate(eval_spec)
                    for metric, scalar in metrics.items():
                        file_writer.add_scalar(metric, scalar, self._curr_epoch)
                        tensorboard_writer.add_scalar(metric, scalar, global_step=self._curr_epoch)
                    self._early_stopper.step(metrics.get("accuracy"))
                    self._save_checkpoint(
                        optimizer_type=train_spec.optimizer_type, 
                        optimizer_params=train_spec.optimizer_params,
                        best_metric_value=self._early_stopper.best_metric_value,
                        steps_since_last_improvemet=self._early_stopper.steps_since_last_improvemet,
                        wait_improvement_n_evals=train_spec.wait_improvement_n_evals
                    )
                    self._clear_checkpoints(
                        best_checkpoint=self._curr_epoch - self._early_stopper.steps_since_last_improvemet * eval_spec.every_n_epochs,
                        keep_n_last=eval_spec.keep_n_checkpoints
                    )

            tensorboard_writer.close()
            
            if self._early_stopper.stop:
                print("accuracy didn't improve for last {} eval probs, interrupted on epoch {}".format(train_spec.wait_improvement_n_evals, self._curr_epoch))
            elif self._curr_epoch >= train_spec.max_epochs:
                print("reached max_epoch steps: {}".format(train_spec.max_epochs))
        else:
            print("model have already trained for max_epochs steps: {}".format(train_spec.max_epochs))

    def evaluate(self, eval_spec):
        self._net.eval()

        cm = ConfusionMatrix([], [], eval_spec.class_num)

        eval_loader = data.DataLoader(eval_spec.dataset, batch_size=eval_spec.batch_size, shuffle=True, num_workers=2)
        for batch in eval_loader:
            inputs, labels = batch[0].to(self._device), batch[1]
            with torch.no_grad():
                predictions = self._net(inputs)
                _, predictions = torch.max(predictions, 1)
                cm.append(predictions.cpu().numpy(), labels.numpy())

        metrics = { "accuracy": cm.accuracy() }
        for i, acc in enumerate(cm.class_accuracy()):
            metrics["accuracy_{}".format(eval_spec.class_map.get(i))] = acc
        
        return metrics

    def _save_checkpoint(self, optimizer_type, optimizer_params, best_metric_value, steps_since_last_improvemet, wait_improvement_n_evals):
        params = {
            "optimizer_type": optimizer_type,
            "optimizer_params": optimizer_params,
            "best_metric_value": best_metric_value,
            "steps_since_last_improvemet": steps_since_last_improvemet,
            "wait_improvement_n_evals": wait_improvement_n_evals
        }

        checkpoint.save(
            model_dir=self._model_dir,
            epoch=self._curr_epoch,
            net=self._net,
            optimizer=self._optimizer,
            params=params
        )

    def _clear_checkpoints(self, best_checkpoint, keep_n_last):
        ckpts = checkpoint.all(self._model_dir)
        if len(ckpts) > keep_n_last:
            checkpoints_to_remove = list(set(ckpts[0:-1 * keep_n_last]) - set([best_checkpoint]))
            checkpoint.remove(self._model_dir, checkpoints_to_remove)