import os
import numpy as np
import pandas as pd
from training.metrics.logger import Logger

def plot_metrics(model_dir, k):
        if k == 2:
            logs = Logger(model_dir).read()
        elif k > 2:
            logs = pd.DataFrame(data=None, index=None, columns=None)
            for i in range(k):
                log_dir = os.path.join(model_dir, "fold_{}".format(i))
                log = Logger(log_dir).read(fold_num=i)
                logs = logs.append(log, ignore_index=True)

        Logger.plot(logs, os.path.join(model_dir, "plot.png"))

def max_accuracy(model_dir, k):
        if k == 2:
            logs = Logger(model_dir).read()
            acc_per_checkpoint, _ = Logger.max_accuracy(logs)
            acc_combination = acc_per_checkpoint
        elif k > 2:
            logs = pd.DataFrame(data=None, index=None, columns=None)
            fold_accs = [None] * k
            for i in range(k):
                log_dir = os.path.join(model_dir, "fold_{}".format(i))
                log = Logger(log_dir).read(fold_num=i)
                fold_accs[i], _ = Logger.max_accuracy(log)
                logs = logs.append(log, ignore_index=True)
            acc_per_checkpoint, _ = Logger.max_accuracy(logs)
            acc_combination = np.mean(fold_accs)

        return acc_per_checkpoint, acc_combination