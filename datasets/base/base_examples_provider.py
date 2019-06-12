import os
import numpy as np
import pandas as pd
from datasets.utils.combinator import Combinator
from utils.helpers import flatten_list
from utils.dirs import is_empty

class BaseExamplesProvider(object):
    def __init__(self, name, params):
        self.name = name

        self.db_name = params["db_name"]
        self.experiment = params["experiment"]
        self.slice_window = params["slice_window"]
        self.rhythm_map = params["rhythm_map"]
        self.rhythm_filter = params["rhythm_filter"]
        self.split_ratio = params["split_ratio"]

        self._examples = None

    #abstract members
    def _build_examples(self):
        """process records, creates labeled examples and saves them to disk
        returns: None
        """
        raise NotImplementedError()

    def _load_examples(self):
        """load examples from disk
        returns: dictionary of Example namedtupe 
        {
            {split_number}: {
                "original": [Example, ...]
                "augmented": [Example, ...]
            },
            ...
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

    def get(self, fold_nums, include_augmented):
        """Gets examples set from splits numbers list
        Args:
            fold_nums: list of int, numbers of fold_nums.
            In default scenario [0] for TRAIN set, [1] for EVAL
            In k-split validation scenario, for example, for 5-split validation it could be [0, 1, 2, 3] for TRAIN
            and [4] for EVAL
            include_augmented: bool. True for TRAIN, False for EVAL
        Returns:
            list of Example named tuple (x, y)
        """
        if not self._examples:
            if self.examples_exists:
                print("loading examples...")
                self._examples = self._load_examples()
                print("loading examples complete")
            else:
                raise ValueError("Examples not exist")

        examples = [self._examples[i]["original"] for i in fold_nums]
        if include_augmented:
            examples_aug = [self._examples[i]["augmented"] for i in fold_nums]
            examples += examples_aug
        
        return flatten_list(examples)

    @property
    def examples_dir(self):
        return os.path.join("data", "examples", self.db_name, self.name, self.experiment, str(self.slice_window))

    @property
    def examples_exists(self):
        if os.path.exists(self.examples_dir) and is_empty(self.examples_dir):
            return False
        else:
            for i in range(len(self.split_ratio)):
                if not os.path.exists(os.path.join(self.examples_dir, str(i))):
                    return False
        return True

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

    def _split_aug_slices(self, slices, original_splits):
        """Split augmented slices according distribution of original slices.
        Args:
            slices: list of Slice namedtuple
            original_splits: k-length list of splits, each split contains slices
        Returns:
            list of shape [k, n_slices]
        """
        aug_splits = [None] * len(original_splits)

        df_columns = ["record", "rhythm", "start", "end", "signal"]
        aug_slices_df = pd.DataFrame(slices, columns=df_columns)

        for i, s in enumerate(original_splits):
            slices_df = pd.DataFrame(s, columns=["index"] + df_columns)

            aug_split_slices = []
            aug_splits[i] = aug_split_slices

            for rhythm, group in slices_df.groupby("rhythm"):
                records = group["record"].unique()

                include = aug_slices_df[(aug_slices_df["rhythm"] == rhythm) & (aug_slices_df["record"].isin(records))]

                aug_split_slices.extend(include.itertuples())


            #equalize distribution of classes
            """
            orig_distribution = slices_df["rhythm"].value_counts().sort_values().iteritems()
            orig_distribution = list(orig_distribution)

            min_class, min_count = orig_distribution[0]

            min_class_slices = [s for s in aug_split_slices if s.rhythm == min_class]
            aug_splits[i].extend(min_class_slices)

            min_class_slices_num = min_count + len(min_class_slices) #original + augmented

            for j in range(1, len(orig_distribution)):
                class_name, class_count = orig_distribution[j]

                take = min_class_slices_num - class_count
                if take > 0:
                    class_slices = [s for s in aug_split_slices if s.rhythm == class_name]
                    random.shuffle(class_slices)

                    take = min_class_slices_num - class_count

                    aug_splits[i].extend(class_slices[:take])
            """

        return aug_splits

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