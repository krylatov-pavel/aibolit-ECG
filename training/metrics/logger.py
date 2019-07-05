import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.helpers import unzip_list
from utils.dirs import create_dirs

class Logger(object):
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._fpath = os.path.join(log_dir, "log.csv")

    def add_scalar(self, tag, value, step):
        is_newfile = not os.path.exists(self._fpath)
        if is_newfile:
            create_dirs([self._log_dir])

        with open(self._fpath, "a", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            if is_newfile:
                writer.writerow(["tag", "value", "step"])
            writer.writerow([tag, value, step])

    def read(self, fold_num=None):
        log = pd.read_csv(self._fpath, delimiter=",")
        
        data = []
        for step, group in log.groupby("step"):
            data_row = {r.tag: r.value for r in group.itertuples()}
            data_row["step"] = step
            if fold_num is not None:
                data_row["fold_num"] = fold_num
            data.append(data_row)

        df = pd.DataFrame(data)
        return df   

    @staticmethod
    def plot(logs, fpath):
        steps = logs.groupby("step")

        metrics = list(set(logs.columns) - set(["step", "fold_num"]))
        plots = []
        legends = []

        fig = plt.figure()
        plt.xlabel("steps")
        plt.ylabel("accuracy")

        plt.ylim(0.0, 1.0)
        y_pos = np.arange(11) * 0.1
        plt.yticks(y_pos, [str(int(percent * 100)) for percent in y_pos])
        x_pos = sorted(logs.step.unique())
        plt.xticks(x_pos, x_pos)
        
        for metric in metrics:
            m_mean = steps[metric].agg(np.mean)
            x, y = unzip_list(m_mean.iteritems())
            alpha = 0.5

            if metric == "accuracy":
                idx_max = np.argmax(y)
                step = x[idx_max]
                max_accuracy = y[idx_max]
                alpha = 1
                color = "black"
                plt.text(min(steps)[0], 0.05, "max accuracy {:.3f} on step {}".format(max_accuracy, step))

            label = metric
            plot, = plt.plot(x, y, alpha=alpha, label=label)
            plots.append(plot)
            legends.append(label)

        plt.legend(plots, legends)
        plt.legend(loc="upper left")
        plt.grid(axis="y")

        fig.savefig(fpath)
        plt.close(fig)

    @staticmethod
    def max_accuracy(logs):
        steps = logs.groupby("step")
        
        accuracy = steps.accuracy.agg(np.mean)
        steps, accuracy = unzip_list(accuracy.iteritems())
        
        best_idx = np.argmax(accuracy)

        return accuracy[best_idx], steps[best_idx]