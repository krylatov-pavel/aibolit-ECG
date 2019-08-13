import os
import random
import json
import scipy.signal
import utils.helpers as helpers
import utils.dirs as dirs
import shutil 
import numpy as np
from datasets.utils.ecg import ECG
from datasets.utils.data_structures import ExampleMetadata, Example
from datasets.base.base_examples_generator import BaseExamplesGenerator

class CardianExamplesGenerator(BaseExamplesGenerator):
    def __init__(self, common_params, source_params):
        self._example_duration = common_params.example_duration
        self._resample_fs = common_params.example_fs
        self._seed = common_params.get("seed") or 0

        self._fs = source_params.fs
        self._source_name = source_params.name
        self._class_settings = source_params.class_settings
        self._rhythm_filter = {}
        for name, c in source_params.class_settings.items():
            c.update({"name": name})
            self._rhythm_filter[c.rhythm] = c

        self._median_filter = source_params.get("median_filter") or 0

    def get_examples_meta(self):
        ecgs = self.__ecg_generator()

        metadata = [e.get_examples_metadata(self._example_duration, self._class_settings) for e in ecgs]
        metadata = helpers.flatten_list(metadata)
        
        return metadata

    def get_examples(self, source_id, metadata):
        examples = []

        fname = "{}.json".format(source_id)
        for label in set(m.label for m in metadata):
            fpath = os.path.join("data", "database", self._source_name, label, fname) 

            with open(fpath, "r") as json_file:
                signal = json.load(json_file)
                if self._median_filter:
                    signal = scipy.signal.medfilt(signal, self._median_filter)

                ecg = ECG(
                    source_type=self._source_name,
                    name=source_id,
                    labels=[label],
                    timecodes=[0],
                    fs=self._fs,
                    signal = signal
                )

            label_metadata = [m for m in metadata if m.label == label]
            examples.extend(ecg.get_examples(label_metadata, resample_fs=self._resample_fs))
        
        return examples

    def split_examples(self, metadata, first_fraction):
        first_group = []
        second_group = []

        labels = set(m.label for m in metadata)
        for label in labels:
            class_metadata = [m for m in metadata if m.label == label]

            source_ids = sorted(list(set(m.source_id for m in class_metadata)))
            random.Random(self._seed).shuffle(source_ids)
            class_metadata = [[m for m in metadata if m.source_id == source_id] for source_id in source_ids]
            class_metadata = helpers.flatten_list(class_metadata)

            split_point = int(len(class_metadata) * first_fraction)
            split_point, found = self.__find_best_split_point(class_metadata, split_point)

            if not found:
                print("Warning: couldn't find valid split, class {}, ratio {}:{2}".format(label, first_fraction, 1-first_fraction))

            first_group.extend(class_metadata[:split_point])
            second_group.extend(class_metadata[split_point:])

        return first_group, second_group

    def __find_best_split_point(self, metadata, split_point):
        def is_valid_split(metadata, split_point):
           return metadata[split_point].source_id != metadata[split_point - 1].source_id 

        if is_valid_split(metadata, split_point):
            return split_point, True
        else:
            found = False
            for distance in range(1, max(split_point, len(metadata) - split_point - 1)):
                split_point_left = split_point - distance
                if split_point_left > 0 and is_valid_split(metadata, split_point_left):
                    split_point = split_point_left
                    found = True
                    break

                split_point_right = split_point + distance
                if split_point_right < (len(metadata) - 1) and is_valid_split(metadata, split_point_right):
                    split_point = split_point_right
                    found = True
                    break

            return split_point, found
    
    ###
    def __ecg_generator(self):
        path = os.path.join("data", "database", self._source_name) 

        dirs = (os.path.join(path, d) for d in os.listdir(path))
        dirs = (d for d in dirs if os.path.isdir(d))

        for d in dirs:
            rhythm = os.path.basename(d)

            if rhythm in self._rhythm_filter:
                label = self._rhythm_filter[rhythm].name

                files = (os.path.join(d, f) for f in os.listdir(d))
                files = (f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() == ".json")
                for f in files:
                    with open(f, "r") as json_file:
                        signal = json.load(json_file)
                        ecg = ECG(
                            source_type=self._source_name,
                            name=os.path.splitext(os.path.basename(f))[0],
                            labels=[label],
                            timecodes=[0],
                            fs=self._fs,
                            signal_len = len(signal)
                        )
                        yield ecg