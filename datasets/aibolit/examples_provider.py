import os
import json
import random
import itertools
from datasets.base.base_examples_provider import BaseExamplesProvider
from datasets.MIT.common.wavedata_provider import WavedataProvider
from datasets.utils.ecg import ECG
from utils.helpers import flatten_list
from utils.dirs import create_dirs

class ExamplesProvider(BaseExamplesProvider):
    def __init__(self, params):
        super(ExamplesProvider, self).__init__("wave", params)

        self.normalize = params["normalize"]
        self.fs = params["fs"]
        self.resample_fs = params["resample_fs"]

    def _build_examples(self):
        """process records, creates labeled examples and saves them to disk
        returns: None
        """
        ecgs = self.__ecg_generator()

        slices = [e.get_slices(self.example_duration, self.rhythm_filter, resample_fs=self.resample_fs) for e in ecgs]
        slices = flatten_list(slices)

        splits = self.__split_slices(slices)

        wp = WavedataProvider()

        for i, examples in enumerate(splits):
            directory = os.path.join(self.examples_dir, str(i))
            wp.save(examples , directory)

            #TO DO: add actual augmentation
            aug_directory = os.path.join(self.examples_dir, str(i), wp.AUGMENTED_DIR)
            create_dirs([aug_directory])

    def _load_examples(self):
        example_splits = {}
        wp = WavedataProvider()

        for i in range(len(self.split_ratio)):
            directory = os.path.join(self.examples_dir, str(i))
            examples = wp.load(directory, include_augmented=True)

            random.shuffle(examples[0])
            random.shuffle(examples[1])

            example_splits[i] = {
                "original": examples[0],
                "augmented": examples[1]
            }

        if self.normalize:
            _, _, mean, std = self._calc_stats(example_splits)
            for key in example_splits:
                example_splits[key] = {
                    "original": self._normalize(example_splits[key]["original"], mean, std),
                    "augmented": self._normalize(example_splits[key]["augmented"], mean, std)
                }
        
        return example_splits

    def __ecg_generator(self):
        path = os.path.join("data", "database", self.db_name) 

        dirs = (os.path.join(path, d) for d in os.listdir(path))
        dirs = (d for d in dirs if os.path.isdir(d))

        for d in dirs:
            label = os.path.basename(d)

            files = (os.path.join(d, f) for f in os.listdir(d))
            files = (f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() == ".json")
            for f in files:
                with open(f, "r") as json_file:
                    signal = json.load(json_file)
                    ecg = ECG(
                        name=os.path.splitext(os.path.basename(f))[0],
                        signal=signal,
                        labels=[label],
                        timecodes=[0],
                        fs=self.fs
                    )
                    yield ecg

    def __split_slices(self, slices):
        splits = [[] for i in range(len(self.split_ratio))] 
        
        base_class = [f.name for f in self.rhythm_filter if f.distribution == 1][0]
        base_qty = sum(1 for s in slices if s.rhythm == base_class)

        random.shuffle(slices)
        for f in self.rhythm_filter:
            class_slices = (s for s in slices if s.rhythm == f.rhythm)
            class_slices = itertools.islice(class_slices, int(base_qty * f.distribution))
            class_slices = list(class_slices)

            split_points = [0]
            for i, ratio in enumerate(self.split_ratio):
                if i != len(self.split_ratio) - 1:
                    split_point = split_points[-1] + int(ratio * len(class_slices))
                else:
                    split_point = len(class_slices)
                split_points.append(split_point)

            i = 0
            for start, end in zip(split_points[:-1], split_points[1:]):
                splits[i].extend(class_slices[start:end])
                i += 1
        
        return splits