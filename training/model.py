import os
import re
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from training.metrics.running_avg import RunningAvg

class Model(object):
    def __init__(self, net, model_dir):
        self._net = net
        self._model_dir = model_dir

        self._ckpt_extension = ".tar"
        self._ckpt_name_tmpl = "model.ckpt-{}" + self._ckpt_extension
        
    def train(self, num_epochs, train_set, optimizer_params):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.last_checkpoint < num_epochs:
            optimizer = optim.Adam(self._net.parameters(), **optimizer_params)
            loss_fn = nn.CrossEntropyLoss()
            loss_acm = RunningAvg(0.0)

            if self.last_checkpoint:
                optimizer, last_epoch = self._load_checkpoint(self.last_checkpoint, optimizer)
                curr_epoch = last_epoch + 1
            else:
                curr_epoch = 0

            self._net.to(device)

            train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True)

            for epoch in range(curr_epoch, num_epochs):
                loss_acm.next_epoch()
                for batch in train_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)

                    optimizer.zero_grad()
                    predictions = self._net(inputs)
                    loss = loss_fn(predictions, labels)
                    loss.backward()
                    optimizer.step()

                    loss_acm.next_iteration(loss)
                
                print("epoch {}\tloss {:.3f}".format(epoch + 1, loss_acm.avg))
                self._save_checkpoint(optimizer, epoch)

            print("training complete")
        else:
            print("model have already trained for num_epochs parameter")

    def evaluate(self, eval_set, checkpoint=None):
        if self.last_checkpoint > 0:
            self._load_checkpoint(checkpoint or self.last_checkpoint, None)
            self._net.to("cpu")
            self._net.eval()

            eval_loader = data.DataLoader(eval_set, batch_size=100, shuffle=True, num_workers=2)

            tp = 0
            total = 0
            for inputs, labels in eval_loader:
                with torch.no_grad():
                    predictions = self._net(inputs)
                    _, predictions = torch.max(predictions, 1)
                    total += len(labels)
                    tp += (predictions == labels).sum().item()

            print("accuracy {:.3f}".format((tp / total) * 100.0))
        else:
            print("model is not trained")

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