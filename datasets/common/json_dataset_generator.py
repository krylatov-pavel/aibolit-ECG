import os
import json
import numpy as np
import utils.dirs as dirs
from datasets.utils.name_generator import NameGenerator
from datasets.base.base_dataset_generator import BaseDatasetGenerator

class JsonDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, params, sources):
        super(JsonDatasetGenerator, self).__init__(params, sources)

        self._name_generator = NameGenerator(file_extension=".json")

        self._example_duration = params.example_duration
        self._fs = params.example_fs
        self._seed = params.get("seed") or 0

    @property
    def examples_exists(self):
        for i in range(self._k):
            if not os.path.exists(os.path.join(self.examples_dir, str(i))):
                return False
        if self._test_set_size:    
            return os.path.exists(os.path.join(self.examples_dir, "TEST"))
        
    def _save(self, examples, fold):
        fold_path = os.path.join(self.examples_dir, str(fold))
        dirs.create_dirs([fold_path])

        for e in examples:
            fname = self._name_generator.generate_name(
                label=e.metadata.label,
                source_id=e.metadata.source_id,
                start=e.metadata.start,
                end=e.metadata.end
            )

            with open(os.path.join(fold_path, fname), "w") as file:
                json.dump(list(e.data), file, sort_keys=False, indent=4)

    def _dataset_flavor(self):
        classes = ",".join(["{}{}".format("" if (not eq) or eq == 1 else eq, c)
            for c, eq in self._class_distribution.items()])
        return "{}fold_{}s_({})_{}hz_{}json".format(
            self._k,
            self._example_duration,
            classes,
            self._fs,
            "seed{}_".format(self._seed) if self._seed else ""
        )