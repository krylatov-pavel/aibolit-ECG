import os
import numpy as np
import pandas as pd
import datasets.utils.equalizer as eq
import utils.dirs as dirs

class BaseDatasetProvider(object):
    def __init__(self, params, file_provider):
        self._file_provider = file_provider

        self._source_name = params.source_name
        self._k = len(params.split_ratio) 
        self._split_ratio = params.split_ratio
        self._test_set_size = params.test_set_size
        self._class_settings = { c.name: c for c in params.class_settings }
        self._class_distribution = { c.name: c.equalize_distribution for c in params.class_settings } 
        #save read config params here, only those that used in this base class

    @property
    def examples_dir(self):
        return os.path.join("data", "examples", self._source_name, self._dataset_flavor())

    @property
    def examples_exists(self):
        for i in range(self._k):
            if not os.path.exists(os.path.join(self.examples_dir, str(i))):
                return False
        if self._test_set_size:    
            return os.path.exists(os.path.join(self.examples_dir, "TEST"))

    @property
    def stats(self):
        # calc mean, std using rolling avg,
        # print in file, read from file if file exists
        # read   file with stats, return 
        raise NotImplementedError()

    def generate(self):
        if not self.examples_exists:
            print("generating examples...")

            dirs.clear_dir(self.examples_dir)

            #get examples metadata
            examples_meta = self._get_examples_meta()
            
            #split into k folds and TEST set
            folders = {}
            if self._test_set_size:
                folders["TEST"], examples_meta = self._split_examples(examples_meta, first_fraction=self._test_set_size)
                folders["TEST"] = eq.equalize(folders["TEST"], key_fn=lambda m: m.label, distribution={key: 1 for key in self._class_distribution})

            examples_meta = eq.equalize(examples_meta, key_fn=lambda m: m.label, distribution=self._class_distribution)

            for i in range(self._k - 1):
                first_fraction = self._split_ratio[i] / sum(self._split_ratio[i:])
                folders[str(i)], examples_meta = self._split_examples(examples_meta, first_fraction=first_fraction)

            folders[str(i+1)] = examples_meta

            #get examples data and serialize to disk
            for folder, examples_meta in folders.items():
                source_ids = set(m.source_id for m in examples_meta)
                for source_id in source_ids:
                    metadata_group = [m for m in examples_meta if m.source_id == source_id]
                    examples = self._get_examples(source_id, metadata_group)
                    path = os.path.join(self.examples_dir, folder)
                    self._file_provider.save(examples, path)
            
            print("generating examples complete")
        else:
            print("examples for current configuration already exist")

    def train_set_path(self, eval_fold_number=None):
        if self._k == 2:
            return [os.path.join(self.examples_dir, "0")]
        elif self._k > 2:
            folders = set(range(self._k)) - set((eval_fold_number,))
            return [os.path.join(self.examples_dir, str(f)) for f in folders]
        else:
            raise ValueError("Invalid k")

    def eval_set_path(self, eval_fold_number=None):
        if self._k == 2:
            return [os.path.join(self.examples_dir, "1")]
        elif self._k > 2:
            return [os.path.join(self.examples_dir, str(eval_fold_number))]
        else:
            raise ValueError("Invalid k")

    def test_set_path(self):
        return [os.path.join(self.examples_dir, "TEST")]

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

    def _split_examples(self, metadata, first_fraction):
        first_group = []
        second_group = []

        labels = set(m.label for m in metadata)
        for label in labels:
            class_metadata = [m for m in metadata if m.label == label]
            class_metadata = sorted(class_metadata, key=lambda m: m.source_id)

            split_point = int(len(class_metadata) * first_fraction)
            split_point, found = self.__find_best_split_point(class_metadata, split_point)

            if not found:
                print("Warning: couldn't find valid split, class {}, ratio {}:{2}".format(label, first_fraction, 1-first_fraction))

            first_group.extend(class_metadata[:split_point])
            second_group.extend(class_metadata[split_point:])

        return first_group, second_group

    #abstract members

    def _get_examples_meta(self):
        #returns list of { Example_Metadata }
        #implement in derived class
        raise NotImplementedError()

    def _dataset_flavor(self):
        raise NotImplementedError()

    def _get_examples(self, source_id, metadata):
        #returns list of { Example }
        #implement in derived class
        raise NotImplementedError()    