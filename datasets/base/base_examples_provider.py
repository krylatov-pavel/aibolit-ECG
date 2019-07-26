import random
import datasets.utils.equalizer as eq
import utils.helpers as helpers

class BaseExamplesProvider(object):
    def __init__(self, folders, label_map, equalize_labels=False, seed=0):
        self._folders = folders
        self._label_map = label_map

        examples = helpers.flatten_list([self._get_examples_metadate(f) for f in folders])
        if equalize_labels:
            examples = eq.equalize(examples, key_fn=lambda e: e[2], distribution={ key: 1 for key in label_map }, seed=0)
        random.Random(0).shuffle(examples)
        self._examples = examples

    @property
    def count(self):
        return len(self._examples)

    def get(self, index):
        example = self._examples[index]
        x = self._read_data(example[0], example[1])   
        y = self._label_map[example[2]]
        return x, y

    #abstract members
    def close(self):
        raise NotImplementedError()

    def _read_data(self, folder, key):
        raise NotImplementedError()

    def _get_examples_metadate(self, folder):
        #returns list of tuple (folder, example key, label)
        raise NotImplementedError()