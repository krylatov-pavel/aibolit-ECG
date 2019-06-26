import os
import json
from datasets.utils.ecg import ECG
from datasets.base.base_dataset_provider import BaseDatasetProvider
from datasets.common.wavedata_provider import WavedataProvider
import utils.helpers as helpers

class FileDatasetProvider(BaseDatasetProvider):
    def __init__(self, params):
        super(FileDatasetProvider, self).__init__(params, WavedataProvider())

        self._fs = params.fs
        self._resample_fs = params.resample_fs
        self._example_duration = params.example_duration

    ###abstract methods implementation

    def _dataset_flavor(self):
        classes = ",".join(["{}{}".format("" if s.equalize_distribution == 1 else s.equalize_distribution, c) for c, s in self._class_settings.items()])
        return "{}fold_{}s_{}_{}hz".format(
            self._k,
            self._example_duration,
            classes,
            self._resample_fs or self._fs
        )

    def _get_examples_meta(self):
        ecgs = self.__ecg_generator()

        metadata = [e.get_examples_metadata(self._example_duration, self._class_settings) for e in ecgs]
        metadata = helpers.flatten_list(metadata)

        return metadata

    def _get_examples(self, source_id, metadata):
        fname = "{}.json".format(source_id)
        fpath = os.path.join("data", "database", self._source_name, metadata.label, fname) 

        with open(fpath, "r") as json_file:
            signal = json.load(json_file)
            ecg = ECG(
                name=source_id,
                labels=[metadata.label],
                timecodes=[0],
                fs=self._fs,
                signal = signal
            )

        return ecg.get_examples(metadata, resample_fs=self._resample_fs)
    
    ###

    def __ecg_generator(self):
        path = os.path.join("data", "database", self._source_name) 

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
                        labels=[label],
                        timecodes=[0],
                        fs=self._fs,
                        signal_len = len(signal)
                    )
                    yield ecg