import os
import utils.helpers as helpers
import utils.dirs as dirs
import datasets.utils.equalizer as eq

class BaseDatasetGenerator(object):
    def __init__(self, params, sources):
        self._split_ratio = params.split_ratio
        self._k = len(params.split_ratio) 
        self._test_set_size = params.test_set_size

        self._generators = { src.name: helpers.get_class(src.examples_generator)(params, src) for src in sources }
        self._class_distribution = { lbl: settings.equalize_distribution for lbl, settings in params.class_settings.items() }
        self._label_map =  { lbl: settings.label_map for lbl, settings in params.class_settings.items() }

    @property
    def examples_dir(self):
        return os.path.join("data", "examples", "+".join(self._generators.keys()), self._dataset_flavor())

    @property
    def examples_exists(self):
        for i in range(self._k):
            if not os.path.exists(os.path.join(self.examples_dir, str(i))):
                return False
        if self._test_set_size:    
            return os.path.exists(os.path.join(self.examples_dir, "TEST"))

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

    def generate(self):
        if not self.examples_exists:
            print("generating examples...")
            dirs.clear_dir(self.examples_dir)
            dirs.create_dirs([self.examples_dir])
            
            #get examples metadata
            examples_meta = [gen.get_examples_meta() for _, gen in self._generators.items()]
            examples_meta = helpers.flatten_list(examples_meta)

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
                for generator_name, generator in self._generators.items():
                    source_ids = set(m.source_id for m in examples_meta)
                    for source_id in source_ids:
                        metadata_group = [m for m in examples_meta if m.source_id == source_id and m.source_type == generator_name]
                        examples = generator.get_examples(source_id, metadata_group)
                        self._save(examples, folder)

            print("generating examples complete")
        else:
            print("examples for current configuration already exist")

    def _split_examples(self, examples_meta, first_fraction):
        first_folder = []
        second_folder = []

        for generator_name, generator in self._generators.items():
            group = [e for e in examples_meta if e.source_type == generator_name]
            first_group, second_group = generator.split_examples(group, first_fraction) 
            
            first_folder.extend(first_group)
            second_folder.extend(second_group)

        return first_folder, second_folder

    #abstract members

    def _save(self, examples, fold):
        raise NotImplementedError()

    def _dataset_flavor(self):
        raise NotImplementedError()