import csv
import os
import numpy as np
from datasets.base.base_file_provider import BaseFileProvider

class WavedataProvider(BaseFileProvider):
    def __init__(self):
        super(WavedataProvider, self).__init__(".csv")
        self.AUGMENTED_DIR = "augmented"

    def _read_file(self, fpath):
        with open(fpath, "r", newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            return list(reader)

    def _build_save_file_fn(self, directory, params):
        def save_file_fn(signal, fname):
            fpath = os.path.join(directory, fname)
            with open(fpath, "w", newline='') as f:
                wr = csv.writer(f)
                wr.writerows(np.expand_dims(signal, axis=1))

        def dispose_fn():
            return None

        return save_file_fn, dispose_fn