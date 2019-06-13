import os
import re
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from training.metrics.running_avg import RunningAvg
from training.metrics.file_logger import FileLogger
from training.metrics.confusion_matrix import ConfusionMatrix

class Model(object):
    def __init__(self, net, model_dir, class_num):
        self._net = net
        self._model_dir = model_dir
        self._class_num = class_num

        self._ckpt_extension = ".tar"
        self._ckpt_name_tmpl = "model.ckpt-{}" + self._ckpt_extension

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def train_and_evaluate(self, num_epochs, train_set, optimizer_params, eval_set):
        if self.last_checkpoint < num_epochs:
            optimizer = optim.Adam(self._net.parameters(), **optimizer_params)
            loss_fn = nn.CrossEntropyLoss()
            loss_acm = RunningAvg(0.0)

            if self.last_checkpoint:
                optimizer, last_epoch = self._load_checkpoint(self.last_checkpoint, optimizer)
                curr_epoch = last_epoch + 1
            else:
                curr_epoch = 0

            self._net.to(self._device)

            train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True)

            for epoch in range(curr_epoch, num_epochs):
                self._net.train()
                loss_acm.next_epoch()
                for batch in train_loader:
                    inputs, labels = batch[0].to(self._device), batch[1].to(self._device)

                    optimizer.zero_grad()
                    predictions = self._net(inputs)
                    loss = loss_fn(predictions, labels)
                    loss.backward()
                    optimizer.step()

                    loss_acm.next_iteration(loss)
                
                if epoch % 5 == 4:
                    self._save_checkpoint(optimizer, epoch)
                    self._evaluate(eval_set, epoch)

            if epoch % 5 != 4:
                self._save_checkpoint(optimizer, epoch)
                self._evaluate(eval_set, epoch)

            print("training complete")
        else:
            print("model have already trained for num_epochs parameter")

    def _evaluate(self, eval_set, step):
        self._net.eval()

        eval_loader = data.DataLoader(eval_set, batch_size=100, shuffle=True, num_workers=2)
        
        cm = ConfusionMatrix([], [], self._class_num)
        for batch in eval_loader:
            inputs, labels = batch[0].to(self._device), batch[1]
            with torch.no_grad():
                predictions = self._net(inputs)
                _, predictions = torch.max(predictions, 1)
                cm.append(predictions.cpu().numpy(), labels.numpy())

        metrics = { "accuracy": cm.accuracy() }
        for i, acc in enumerate(cm.class_accuracy()):
            metrics[str(i)] = acc

        file_logger = FileLogger(self._model_dir, "accuracy", list(metrics.keys()))
        file_logger.log(metrics, step)

    def _save_checkpoint(self, optimizer, epoch):
        fname = self._ckpt_name_tmpl.format(epoch + 1)
        fpath = os.path.join(self._model_dir, fname)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self._net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, fpath)

    def _load_checkpoint(self, checkpoint, optimizer):
        fname = self._ckpt_name_tmpl.format(checkpoint)
        fpath = os.path.join(self._model_dir, fname)

        checkpoint = torch.load(fpath)
        
        self._net.load_state_dict(checkpoint['model_state_dict'])
        self._net.to(self._device)
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return optimizer, epoch

    @property
    def last_checkpoint(self):
        regex = "^model.ckpt-(?P<checkpoint>[\d]+)\.tar" 
        
        ckpts_names = (f for f in os.listdir(self._model_dir) if f.endswith(self._ckpt_extension))
        ckpt_matches = (re.match(regex, f) for f in ckpts_names)
        ckpt_nums = [int(m.group("checkpoint")) for m in ckpt_matches if m]

        return max(ckpt_nums) if len(ckpt_nums) else 0