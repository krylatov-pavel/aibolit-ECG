import os
import re
import torch
import adabound as adabound
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from training.metrics.running_avg import RunningAvg
from training.metrics.confusion_matrix import ConfusionMatrix
from training.metrics.logger import Logger
import training.checkpoint as checkpoint
from training.early_stopper import EarlyStopper
from training.eval_scheduler import EvalScheduler
from training.lr_scheduler import CustomReduceLROnPlateau

def create_optimizer(optimizer_type, net_parameters, optimizer_params):
    if optimizer_type == "adam":
        return optim.Adam(net_parameters, **optimizer_params)
    elif optimizer_type == "adabound":
        return adabound.AdaBound(net_parameters, **optimizer_params)
    else:
        raise ValueError("Unknown optimizer type")

def create_lr_scheduler(optimizer, lr_scheduler_params):
    return CustomReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        verbose=True,
        **lr_scheduler_params
    )

class Model(object):
    def __init__(self, net, model_dir, device=None, curr_epoch=None, optimizer=None, lr_scheduler=None,
        early_stopper=None, eval_scheduler=None):
        self._net = net
        self._model_dir = model_dir
        self._device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._curr_epoch = curr_epoch or 0
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._early_stopper = early_stopper
        self._eval_scheduler = eval_scheduler

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

        lr_scheduler_params = params.get("lr_scheduler_params")
        lr_scheduler_state = params.get("lr_scheduler_state")
        lr_scheduler = create_lr_scheduler(optimizer, lr_scheduler_params)
        lr_scheduler.load_state_dict(lr_scheduler_state)
        
        early_stopper_params = params.get("early_stopper_params")
        early_stopper_state = params.get("early_stopper_state")
        early_stopper = EarlyStopper(**early_stopper_params)
        early_stopper.load_state_dict(early_stopper_state)

        eval_scheduler_params = params.get("eval_scheduler_params")
        eval_scheduler_state = params.get("eval_scheduler_state")
        eval_scheduler = EvalScheduler(**eval_scheduler_params)
        eval_scheduler.load_state_dict(eval_scheduler_state)

        return Model(
            net=net,
            model_dir=model_dir,
            device=device,
            curr_epoch=epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            early_stopper=early_stopper,
            eval_scheduler=eval_scheduler
        )

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
            
            if not self._lr_scheduler:
                self._lr_scheduler = create_lr_scheduler(self._optimizer, train_spec.lr_scheduler_params)
            
            if not self._early_stopper:
                self._early_stopper = EarlyStopper(**train_spec.early_stopper_params)
            else:
                self._early_stopper.re_init(**train_spec.early_stopper_params)

            if not self._eval_scheduler:
                self._eval_scheduler = EvalScheduler(**eval_spec.eval_scheduler_params)
            else:
                self._eval_scheduler.re_init(**eval_spec.eval_scheduler_params)
            
            loss_fn = nn.CrossEntropyLoss()
            loss_acm = RunningAvg(0.0)

            self._net.to(self._device)

            file_writer = Logger(log_dir=self._model_dir)
            tb_writer_train = SummaryWriter(log_dir=os.path.join(self._model_dir, "tbruns/train"))
            tb_writer_eval = SummaryWriter(log_dir=os.path.join(self._model_dir, "tbruns/eval"))
            inputs = next(iter(train_spec.dataset))[0].unsqueeze(0).to(self._device)
            tb_writer_train.add_graph(self._net, input_to_model=inputs)

            #evaluate initial accuracy
            metrics, _ = self.evaluate(eval_spec)
            for metric, scalar in metrics.items():
                file_writer.add_scalar(metric, scalar, 0)
                tb_writer_train.add_scalar(metric, scalar, global_step=0)

            train_loader = data.DataLoader(train_spec.dataset, batch_size=train_spec.batch_size, shuffle=True, num_workers=0)

            while self._curr_epoch < train_spec.max_epochs and not self._early_stopper.stop:
                self._curr_epoch += 1
                self._net.train()
                loss_acm.next_epoch()
                for batch in train_loader:
                    inputs, y = batch[0].to(self._device), batch[1].to(self._device)
                    if len(y) > 1:
                        self._optimizer.zero_grad()
                        predictions = self._net(inputs)
                        loss = loss_fn(predictions, y)
                        loss.backward()
                        self._optimizer.step()

                        loss_acm.next_iteration(loss.item())

                tb_writer_train.add_scalar("loss", loss_acm.avg, global_step=self._curr_epoch)
                
                self._eval_scheduler.step(self._curr_epoch)
                if self._eval_scheduler.eval or self._curr_epoch == train_spec.max_epochs:
                    metrics, _ = self.evaluate(eval_spec)
                    for metric, scalar in metrics.items():
                        file_writer.add_scalar(metric, scalar, self._curr_epoch)
                        tb_writer_eval.add_scalar(metric, scalar, global_step=self._curr_epoch)
                    
                    self._lr_scheduler.step(metrics.get("accuracy"), epoch=self._curr_epoch)
                    self._early_stopper.step(metrics.get("accuracy"), epoch=self._curr_epoch)

                    self._save_checkpoint(
                        optimizer_type=train_spec.optimizer_type, 
                        optimizer_params=train_spec.optimizer_params,
                        lr_scheduler_params=train_spec.lr_scheduler_params,
                        early_stopper_params=train_spec.early_stopper_params,
                        eval_scheduler_params=eval_spec.eval_scheduler_params
                    )
                    self._clear_checkpoints(
                        best_checkpoint=self._early_stopper.best_epoch,
                        keep_n_last=eval_spec.keep_n_checkpoints
                    )

            tb_writer_train.close()
            tb_writer_eval.close()

            if self._early_stopper.stop:
                print("accuracy didn't improve for last {} eval probs, interrupted on epoch {}".format(train_spec.early_stopper_params.patience, self._curr_epoch))
            elif self._curr_epoch >= train_spec.max_epochs:
                print("reached max_epoch steps: {}".format(train_spec.max_epochs))
        else:
            print("model have already trained for max_epochs steps: {}".format(train_spec.max_epochs))

    def evaluate(self, eval_spec):
        self._net.eval()

        loss_fn = nn.CrossEntropyLoss()
        loss_acm = RunningAvg(0.0)

        cm = ConfusionMatrix([], [], eval_spec.class_num)

        eval_loader = data.DataLoader(eval_spec.dataset, batch_size=eval_spec.batch_size, shuffle=True, num_workers=0)
        for batch in eval_loader:
            inputs, y = batch[0].to(self._device), batch[1].to(self._device)
            with torch.no_grad():
                predictions = self._net(inputs)
                loss = loss_fn(predictions, y)
                loss_acm.next_iteration(loss.item())
                
                _, predictions = torch.max(predictions, 1)
                cm.append(predictions.cpu().numpy(), y.cpu().numpy())

        metrics = { 
            "accuracy": cm.accuracy(),
            "loss": loss_acm.avg
        }
        
        for i, acc in enumerate(cm.class_accuracy()):
            metrics["accuracy_{}".format(eval_spec.class_map.get(i))] = acc
        
        return metrics, cm

    def _save_checkpoint(self, optimizer_type, optimizer_params, lr_scheduler_params,
        early_stopper_params, eval_scheduler_params):
        params = {
            "optimizer_type": optimizer_type,
            "optimizer_params": optimizer_params,
            "lr_scheduler_params": lr_scheduler_params,
            "lr_scheduler_state": self._lr_scheduler.state_dict(),
            "early_stopper_params": early_stopper_params,
            "early_stopper_state": self._early_stopper.state_dict(),
            "eval_scheduler_params": eval_scheduler_params,
            "eval_scheduler_state": self._eval_scheduler.state_dict()
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