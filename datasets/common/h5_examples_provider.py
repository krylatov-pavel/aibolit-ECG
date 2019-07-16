import h5py
import numpy as np
import os
from datasets.base.base_examples_provider import BaseExamplesProvider

class H5ExamplesProvider(BaseExamplesProvider):
    def __init__(self, folders, label_map, equalize_labels=False):
        self._files = { f: h5py.File("{}.hdf5".format(f), "r") for f in folders }
        
        super(H5ExamplesProvider, self).__init__(folders, label_map, equalize_labels)
    
    def _read_data(self, folder, key):
        dset = self._files[folder][key][:]
        return np.expand_dims(dset, axis=0)

    def _get_examples_metadate(self, folder):
        keys = self._files[folder].keys()
        return [(folder, key, self._files[folder][key].attrs["label"].decode("utf-8")) for key in keys]

    def close(self):
        for _, f in self._files.items():
            f.close()