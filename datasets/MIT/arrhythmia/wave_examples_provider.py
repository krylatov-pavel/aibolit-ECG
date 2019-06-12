import os
import math
import numpy as np
import pandas as pd
import random
from datasets.MIT.base.base_examples_provider import BaseExamplesProvider
from datasets.MIT.providers.database_provider import DatabaseProvider
from datasets.MIT.providers.wavedata_provider import WavedataProvider
from datasets.Arythmia.arythmia_ecg import ArythmiaECG
from datasets.MIT.utils.data_structures import Example, Slice
from utils.helpers import flatten_list, unzip_list, rescale, normalize
from utils.dirs import create_dirs

class WaveExamplesProvider(BaseExamplesProvider):
    def __init__(self, params):
        super(WaveExamplesProvider, self).__init__("wave", params)

        self.equalize = params["equalize_classes"]
        self.rescale = params.rescale if hasattr(params, "rescale") else False
        self.normalize = params.normalize
        self.slice_overlap = params["slice_overlap"]

    def _build_examples(self):
        ecgs = self._get_ECGs()

        slices = [e.get_slices(self.slice_window, self.rythm_filter, self.slice_overlap) for e in ecgs]
        slices = flatten_list(slices)

        for f in self.rythm_filter:
            if hasattr(f, "allow_spread") and f.allow_spread:
                not_class = [s for s in slices if s.rythm != f.name]
                spreaded_class = self._spread_slices([s for s in slices if s.rythm == f.name], len(self.split_ratio))
                slices = not_class + spreaded_class

        splits = self._split_slices(slices)

        wp = WavedataProvider()

        for i in range(len(splits)):
            examples = splits[i]
            
            if self.equalize:
                examples, _ = self._equalize_examples(examples, [], equalizer=self.equalize)

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

        if self.rescale:
            min, max, _, _ = self._calc_stats(example_splits)
            for key in example_splits:
                example_splits[key] = {
                    "original": self._rescale_examples(example_splits[key]["original"], min, max, self.rescale),
                    "augmented": self._rescale_examples(example_splits[key]["augmented"], min, max, self.rescale)
                }

        if self.normalize:
            _, _, mean, std = self._calc_stats(example_splits)
            for key in example_splits:
                example_splits[key] = {
                    "original": self._normalize(example_splits[key]["original"], mean, std),
                    "augmented": self._normalize(example_splits[key]["augmented"], mean, std)
                }
        
        return example_splits

    def _get_ECGs(self):
        """Reads records from database and converts them to ECG objects
        Returns:
            list of ECG objects
        """
        if not self._ecgs:
            records = DatabaseProvider(self.db_name).get_records()

            self._ecgs = [ArythmiaECG(name=r.signal.record_name,
                signal=np.reshape(r.signal.p_signal, [-1]),
                labels=r.annotation.aux_note,
                beats=r.annotation.symbol,
                timecodes=r.annotation.sample) for r in records]

        return self._ecgs

    def _equalize_examples(self, examples, aug_examples, equalizer=None):
        examples_eq = []
        aug_examples_eq = []
        
        df = pd.DataFrame(examples)  

        orig_distribution = df.rythm.value_counts().sort_values().iteritems()
        orig_distribution = list(orig_distribution)

        if equalizer:
            min_class, min_count = [d for d in orig_distribution if not equalizer[d[0]]][-1]
        else:
            min_class, min_count = orig_distribution[0]

        min_count += len([e for e in aug_examples if e.rythm == min_class])

        for class_name, class_count in orig_distribution:
            class_examples = [e for e in examples if e.rythm == class_name]
            aug_class_examples = [e for e in aug_examples if e.rythm == class_name]

            if equalizer and not equalizer[class_name]:
                take = len(class_examples)
                take_aug = len(aug_class_examples)
            elif class_count <= min_count:
                take = class_count
                take_aug = min_count - class_count
                random.shuffle(aug_class_examples)
            else:
                take = min_count
                take_aug = 0
                random.shuffle(class_examples)

            examples_eq.extend(class_examples[:take])
            aug_examples_eq.extend(aug_class_examples[:take_aug])

        return examples_eq, aug_examples_eq

    def _calc_stats(self, splits):
        data = [s["original"] + s["augmented"] for key, s in splits.items()]
        data = flatten_list(data)
        data = [e.x for e in data]
        data = np.array(data)

        min = np.min(data)
        max = np.max(data)
        mean = np.mean(data)
        std = np.std(data)

        return min, max, mean, std

    def _rescale_examples(self, examples, min, max, scale):
        return [Example(x=rescale(e.x, min, max, scale.min, scale.max), y=e.y, name=e.name) for e in examples]

    def _normalize(self, examples, mean, std):
        return [Example(x=normalize(e.x, mean, std), y=e.y, name=e.name) for e in examples]

    def _spread_slices(self, slices, k):
        spread = []
        df = pd.DataFrame(slices)

        for record, group in df.groupby("record"):
            slices_in_fold = math.ceil(len(group) / k)
            group_slices = list(group.itertuples())
            for i in range(k):
                start = i * slices_in_fold
                end = (i + 1) * slices_in_fold
                if end > len(group):
                    end = len(group)
                spreaded_slices = [Slice("{}.{}".format(record, i), s.rythm, s.start, s.end, s.signal) for s in group_slices[start:end]]
                spread.extend(spreaded_slices)

        return spread