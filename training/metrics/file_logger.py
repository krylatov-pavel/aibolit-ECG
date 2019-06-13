import os
import numpy as np
import pandas as pd
from utils.dirs import create_dirs

class FileLogger(object):
    def __init__(self, model_dir, name, metrics):
        self._model_dir = model_dir
        self._fpath = os.path.join(model_dir, "{}.csv".format(name))
        self._metrics = metrics
        self._columns = ["step"] + metrics

    def log(self, values, step):
        create_dirs([self._model_dir])

        data_row = {
            "step": [step]
        }
        
        for metric in self._metrics:
            data_row[metric] = [values[metric]]

        df = pd.DataFrame(data_row, columns=self._columns)

        stats = self._load_logs()
        stats = stats.append(df)

        stats.to_csv(self._fpath, index=False)
    
    def plot(self):
        pass

    def _load_logs(self):
        if os.path.exists(self._fpath):
            return pd.read_csv(self._fpath)
        else:
            return pd.DataFrame(columns=self._columns)