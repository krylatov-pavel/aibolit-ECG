import os
import h5py
import numpy as np
from datasets.utils.name_generator import NameGenerator
from datasets.base.base_dataset_generator import BaseDatasetGenerator

class H5DatasetGenerator(BaseDatasetGenerator):
    def __init__(self, params, sources):
        super(H5DatasetGenerator, self).__init__(params, sources)

        self._name_generator = NameGenerator(file_extension="")

        self._example_duration = params.example_duration
        self._fs = params.example_fs
        self._seed = params.get("seed") or 0

    @property
    def examples_exists(self):
        for i in range(self._k):
            if not os.path.exists(os.path.join(self.examples_dir, "{}.hdf5".format(i))):
                return False
        if self._test_set_size:    
            return os.path.exists(os.path.join(self.examples_dir, "TEST.hdf5"))
        
    def _save(self, examples, fold):
         with h5py.File(os.path.join(self.examples_dir, "{}.hdf5".format(fold)), "a") as f:
            for e in examples:
                fname = self._name_generator.generate_name(
                    label=e.metadata.label,
                    source_id=e.metadata.source_id,
                    start=e.metadata.start,
                    end=e.metadata.end
                )

                dset = f.create_dataset(fname, data=e.data.astype(np.float32), dtype=np.float32)
                dset.attrs["label"] = np.string_(e.metadata.label)
                dset.attrs["source_id"] = np.string_(e.metadata.source_id)
                dset.attrs["source_type"] = np.string_(e.metadata.source_type)

    def _dataset_flavor(self):
        classes = ",".join(["{}{}".format("" if (not eq) or eq == 1 else eq, c)
            for c, eq in self._class_distribution.items()])
        return "{}fold_{}s_({})_{}hz_{}h5".format(
            self._k,
            self._example_duration,
            classes,
            self._fs,
            "seed{}_".format(self._seed) if self._seed else ""
        )