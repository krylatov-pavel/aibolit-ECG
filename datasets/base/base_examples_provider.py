import os
import numpy as np
import pandas as pd
from datasets.utils.combinator import Combinator
from datasets.utils.data_structures import Example
from utils.helpers import flatten_list
from utils.dirs import is_empty
import utils.helpers as helpers

#TODO: rewrite examples provider, keep just most generic methods and basic structure, use pytorch transform for normalization
class BaseExamplesProvider(object):
    def __init__(self, name, params):
        self.name = name

        self.db_name = params["db_name"]
        self.experiment = params["experiment"]
        self.example_duration = params["example_duration"]
        self.rhythm_map = params["rhythm_map"]
        self.rhythm_filter = params["rhythm_filter"]
        self.split_ratio = params["split_ratio"]
        self.label_map = params["label_map"]
        self.normalize = params["normalize"]

        self.__examples = None

    #abstract members
    def _build_examples(self):
        """process records, creates labeled examples and saves them to disk
        returns: None
        """
        raise NotImplementedError()

    def _load_examples(self, path):
        """load examples from disk
        returns: dictionary of Example namedtupe 
        {
            "original": [Example, ...]
            "augmented": [Example, ...]
        }
        """
        raise NotImplementedError()

    #base members
    def generate(self):
        if not self.examples_exists:
            print("generating examples...")
            self._build_examples()
            print("generating examples complete")
        else:
            print("examples for current configuration already exist")

    def len(self, fold_nums, is_train=False):
        num = 0
        examples = self.__get_examples()
        groups = [0] + [1] * is_train
        for i in fold_nums:
            for g in groups:
                num += len(examples[i][g])
        return num

    def get_example(self, index, fold_nums, is_train=False):
        examples = self.__get_examples()
        groups = [0] + [1] * is_train
        for i in fold_nums:
            for g in groups:
                group_length = len(examples[i][g])
                if index <= group_length - 1:
                    return (examples[i][g][index].x, self.label_map[examples[i][g][index].y])
                else:
                    index -= group_length
        raise ValueError("Index is out of range")

    @property
    def examples_dir(self):
        return os.path.join("data", "examples", self.db_name, self.name, self.experiment, str(self.example_duration))

    @property
    def examples_exists(self):
        if os.path.exists(self.examples_dir) and is_empty(self.examples_dir):
            return False
        else:
            for i in range(len(self.split_ratio)):
                if not os.path.exists(os.path.join(self.examples_dir, str(i))):
                    return False
        return True

    def _load_dataset(self):
        example_splits = {}
        
        for i in range(len(self.split_ratio)):
            directory = os.path.join(self.examples_dir, str(i))
            example_splits[i] = self._load_examples(directory)

        if self.normalize:
            _, _, mean, std = self._calc_stats(example_splits)
            for key in example_splits:
                example_splits[key] = self._normalize(example_splits[key], mean, std)

        for key in example_splits:
            example_splits[key] = self._equalize(example_splits[key])
        
        return example_splits

    def _equalize(self, examples):
        equalized = []
        remaining = []
        classes_sr = pd.Series([e.y for e in examples])
        count_min = classes_sr.value_counts()[-1]
        for y in classes_sr.unique():
            class_examples = [e for e in examples if e.y == y]
            equalized.extend(class_examples[:count_min])
            remaining.extend(class_examples[count_min:])

        return equalized, remaining

    def _calc_stats(self, splits):
        data = [s for key, s in splits.items()]
        data = flatten_list(data)
        data = [e.x for e in data]
        data = np.array(data)

        min = np.min(data)
        max = np.max(data)
        mean = np.mean(data)
        std = np.std(data)

        return min, max, mean, std

    def _normalize(self, examples, mean, std):
        return [Example(x=helpers.normalize(e.x, mean, std), y=e.y, name=e.name) for e in examples]

    def _split_slices(self, slices):
        """Split slices with according to slpit_ratio distribution.
        Args:
            slices: list of Slice namedtuple
        In default case, [0.8, 0.2], 80% for TRAIN set and 20% for EVAL
        In case k-fold validation, [0.2, 0.2, 0.2, 0.2, 0.2] each fold has 20% of examples
        Returns:
            list of shape [slice_num, n_slices]
        """
        slices_df = pd.DataFrame(slices, columns=["record", "rhythm", "start", "end", "signal"])

        split_map = self.__build_split_map(slices_df)

        splits_list = []
        for k in range(len(self.split_ratio)):
            split = []
            for rhythm, group in slices_df.groupby("rhythm"):
                include_records = group.record.isin(split_map[rhythm][k])
                rhythm_slices = [s for s in group[include_records].itertuples()]
                
                split.extend(rhythm_slices)
            
            splits_list.append(split)
        
        return splits_list

    def __build_split_map(self, df):
        """
        Returns: dictionary with rhythm type keys, and k-length 2d list values, e.g:
        {
            "(N": [["418", "419"], ["500"], ...],
            ...
        }  
        """
        split_map = {}
        combinator = Combinator()

        for rhythm, rhythm_group in df.groupby("rhythm"):
            slices = [(record, len(record_group)) for record, record_group in rhythm_group.groupby("record")]
            slices_splitted = combinator.split(slices, self.split_ratio)
            split_map[rhythm] = [[s[0] for s in subgroup] for subgroup in slices_splitted]
        
        return split_map

    def __get_examples(self):
        if not self.__examples:
            if self.examples_exists:
                print("loading examples...")
                self.__examples = self._load_dataset()
                print("loading examples complete")
            else:
                raise ValueError("Examples not exist")

        return self.__examples