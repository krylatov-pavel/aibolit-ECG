import datasets.utils.equalizer as eq
import random

class ExamplesProvider(object):
    def __init__(self, folders, file_reader, label_map, equalize_labels=False):
        self._file_reader = file_reader
        self._label_map = label_map

        examples = file_reader.list(folders)
        if equalize_labels:
            examples = eq.equalize(examples, key_fn=lambda e: e[1].label, distribution={ key: 1 for key in label_map })
        random.shuffle(examples)
        self._examples = examples

    @property
    def count(self):
        return len(self._examples)

    def get(self, index):
        x = self._file_reader.read(self._examples[index][0])   
        y = self._label_map[self._examples[index][1].label]
        return x, y