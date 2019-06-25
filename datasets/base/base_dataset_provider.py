import os
import datasets.utils.equalizer as eq

class BaseDatasetProvider(object):
    def __init__(self, params, file_provider):
        self._file_provider = file_provider

        self._k = len(params.split_ratio) 
        self._split_ratio = params.split_ratio
        self._test_set_size = params.test_set_size
        self._class_settings = { c.name: c for c in params.class_settings }
        self._class_distribution = { c.name: c.distribution for c in params.class_settings } 
        #save read config params here, only those that used in this base class

    @property
    def examples_dir(self):
        raise NotImplementedError()

    @property
    def examples_exists(self):
        raise NotImplementedError()

    @property
    def stats(self):
        # calc mean, std using rolling avg,
        # print in file, read from file if file exists
        # read   file with stats, return 
        raise NotImplementedError()

    def generate(self):
        if not self.examples_exists:
            print("generating examples...")

            #get examples metadata
            examples_meta = self.__get_examples_meta()
            examples_meta = eq.equalize(examples_meta, key=lambda m: m.label, distribution=self._class_distribution)

            #split into k folds and TEST set
            folders = {}
            if self._test_set_size:
                folders["TEST"], examples_meta = self.__split_examples(examples_meta, self._test_set_size)

            for i in range(self._k - 1):
                first_group = self._split_ratio[i] / sum(self._split_ratio[i:])
                folders[str(i)], examples_meta = self.__split_examples(examples_meta, first_group)

            folders[i+1] = examples_meta

            #get examples data and serialize to disk
            for folder, examples_meta in folders.items():
                source_ids = set(m.source_id for m in examples_meta)
                for source_id in source_ids:
                    metadata_group = [m for m in examples_meta if m.source_id == source_id]
                    examples = self.__get_examples(source_id, metadata_group)
                    path = os.path.join(self.examples_dir, folder)
                    self._file_provider.save(examples , path)
            
            print("generating examples complete")
        else:
            print("examples for current configuration already exist")

    def train_set_path(self, eval_fold_number=None):
        raise NotImplementedError()

    def eval_set_path(self, eval_fold_number=None):
        raise NotImplementedError()

    def test_set_path(self):
        raise NotImplementedError()

    def __get_examples_meta(self):
        #returns list of { Example_Metadata }
        #implement in derived class
        raise NotImplementedError()

    def __get_examples(self, source_id, metadata):
        #returns list of { Example }
        #implement in derived class
        raise NotImplementedError()

    def __split_examples(self, metadata, first_group):
        """
        returns first_group, second_group:
            [{Example_Metadata}, ...]
        """
        raise NotImplementedError()