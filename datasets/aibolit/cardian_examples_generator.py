import os
import json
import utils.helpers as helpers
from datasets.utils.ecg import ECG
from datasets.utils.data_structures import ExampleMetadata, Example
from datasets.base.base_examples_generator import BaseExamplesGenerator

class CardianExamplesGenerator(BaseExamplesGenerator):
    def __init__(self, common_params, source_params):
        self._example_duration = common_params.example_duration
        self._resample_fs = common_params.example_fs

        self._fs = source_params.fs
        self._source_name = source_params.name
        self._class_settings = source_params.class_settings
        self._rhythm_filter = {}
        for name, c in source_params.class_settings.items():
            c.update({"name": name})
            self._rhythm_filter[c.rhythm] = c
    
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