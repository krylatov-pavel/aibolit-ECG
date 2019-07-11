import numpy as np
import pandas as pd
from datasets.base.base_dataset_provider import BaseDatasetProvider
from datasets.common.wavedata_provider import WavedataProvider
from datasets.MIT.common.database_provider import DatabaseProvider
from datasets.utils.ecg import ECG
from datasets.utils.combinator import Combinator
import utils.helpers as helpers

class FileDatasetProvider(BaseDatasetProvider):
    def __init__(self, params):
        super(FileDatasetProvider, self).__init__(params, WavedataProvider())

        self._rhythm_filter = { c.rhythm: c for _, c in self._class_settings.items() }
        self._rhythm_map = params.rhythm_map

        self._fs = 250
        self._example_duration = params.example_duration

    ###abstract methods implementation

    def _dataset_flavor(self):
        classes = ",".join(["{}{}".format("" if (not s.equalize_distribution) or s.equalize_distribution == 1 else s.equalize_distribution, c)
            for c, s in self._class_settings.items()])
        return "{}fold_{}s_({})".format(
            self._k,
            self._example_duration,
            classes
        )

    def _get_examples_meta(self):
        ecgs = self.__ecg_generator()

        metadata = [e.get_examples_metadata(self._example_duration, self._rhythm_filter, rhythm_map=self._rhythm_map) for e in ecgs]
        metadata = helpers.flatten_list(metadata)

        return metadata

    def _get_examples(self, source_id, metadata):
        database = DatabaseProvider(self._source_name)
        signal, annotation = database.get_record(source_id)
        ecg = ECG(
            name=source_id,
            labels=annotation.aux_note,
            timecodes=annotation.sample,
            fs=self._fs,
            signal = np.squeeze(signal.p_signal)
        )

        examples = ecg.get_examples(metadata)
        return examples
    ###

    def _split_examples(self, metadata, first_fraction):
        first_group = []
        second_group = []
        combinator = Combinator()

        labels = set(m.label for m in metadata)
        for label in labels:
            class_metadata = [m for m in metadata if m.label == label]
            
            groups = list(pd.Series(m.source_id for m in class_metadata).value_counts().items())
            splits = combinator.split(groups, [first_fraction, 1 - first_fraction])
            splits = [[g[0] for g in s] for s in splits]

            first = [m for m in class_metadata if m.source_id in splits[0]]
            second = [m for m in class_metadata if m.source_id in splits[1]]
            
            first_group.extend(first)
            second_group.extend(second)

        return first_group, second_group

    def __ecg_generator(self):
        database = DatabaseProvider(self._source_name)
        records = database.get_records()

        for signal, annotation in records:
            ecg = ECG(
                name=annotation.record_name,
                labels=annotation.aux_note,
                timecodes=annotation.sample,
                fs=self._fs,
                signal_len = signal.sig_len
            )
            yield ecg