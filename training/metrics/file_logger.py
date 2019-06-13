import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dirs import create_dirs
from utils.helpers import unzip_list

class FileLogger(object):
    def __init__(self, fpath, name, metrics):
        self._fpath = fpath
        self._name = name
        self._metrics = metrics
        self._columns = ["name", "step"] + metrics

        self._logs = self._load_logs()
        self._writable = True

    @property
    def columns(self):
        return self._columns

    @property
    def logs(self):
        return self._logs

    def log(self, values, step):
        if self._writable:
            data_row = {
                "name": [self._name],
                "step": [step]
            }
            
            for metric in self._metrics:
                data_row[metric] = [values[metric]]
            new_row = pd.DataFrame(data_row, columns=self._columns)

            self._logs = self._logs.append(new_row)
            self._logs.to_csv(self._fpath, index=False)
        else:
            raise ValueError("log is not in writable state")

    def add(self, file_logger):
        if isinstance(file_logger, FileLogger):
            if set(self._columns) == set(file_logger.columns):
                self._logs = self._logs.append(file_logger._logs)
                self._writable = False
            else:
                raise ValueError("log columns mismatch")    
        else:
            raise ValueError("file_logger must be an instance of FileLogger class")
    
    def plot(self, fpath, class_map):
        steps = self._logs.groupby("step")

        colors = ["b-", "y-", "-g", "-c", "-m", "-r"]
        colors = colors * math.ceil(len(self._metrics) / len(colors))
        metrics = [(m, c) for m, c in zip(self._metrics, colors)]

        plots = []

        fig = plt.figure()
        plt.xlabel("steps")
        plt.ylabel("accuracy")

        plt.ylim(0.0, 1.0)
        y_pos = np.arange(11) * 0.1
        plt.yticks(y_pos, [str(int(percent * 100)) for percent in y_pos])
        
        legend = []
        for metric, color in metrics:
            m_mean = steps[metric].agg(np.mean)
            x, y = unzip_list(m_mean.iteritems())
            alpha = 0.5

            if metric == "accuracy":
                idx_max = np.argmax(y)
                step = x[idx_max]
                max_accuracy = y[idx_max]
                alpha = 1
                plt.text(0.05, 0.05, "max accuracy {:.3f} on step {}".format(max_accuracy, step))
                legend.append(metric)
            else:
                legend.append(class_map[int(metric)])

            plot, = plt.plot(x, y, color, alpha=alpha, label=metric)
            plots.append(plot)

        plt.legend(plots, legend)
        plt.legend(loc="upper left")
        plt.grid(axis="y")

        fig.savefig(fpath)
        plt.close(fig)

    def _load_logs(self):
        if os.path.exists(self._fpath):
            return pd.read_csv(self._fpath)
        else:
            create_dirs([os.path.dirname(self._fpath)])
            return pd.DataFrame(columns=self._columns)