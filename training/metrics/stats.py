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

    Logger.plot_accuracy(logs, os.path.join(model_dir, "accuracy.png"))
    Logger.plot_loss(logs, os.path.join(model_dir, "loss.png"))

def max_accuracy(model_dir, k):
    if k == 2:
        log = Logger(model_dir).read()
        accuracy, step = Logger.max_accuracy(log)
    elif k > 2:
        accuracy = [None] * k
        step = [None] * k
        for i in range(k):
            log_dir = os.path.join(model_dir, "fold_{}".format(i))
            log = Logger(log_dir).read(fold_num=i)
            accuracy[i], step[i] = Logger.max_accuracy(log)
        accuracy = np.mean(accuracy)

    return accuracy, step

def accuracy_stats(model_dir, k):
    if k > 2:
        accuracy = [None] * k
        for i in range(k):
            log_dir = os.path.join(model_dir, "fold_{}".format(i))
            log = Logger(log_dir).read(fold_num=i)
            accuracy[i], _ = Logger.max_accuracy(log)

        mean = np.mean(accuracy)
        std = np.std(accuracy)
        min = np.min(accuracy)
        max = np.max(accuracy)
    else:
        log = Logger(model_dir).read()
        accuracy, _ = Logger.max_accuracy(log)
        mean = min = max = accuracy
        std = 0
        
    return mean, std, min, max