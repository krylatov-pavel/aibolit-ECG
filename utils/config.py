import os
import json
import utils.dictionary as dictionary
from utils.dirs import create_dirs

class Config(object):
    def __init__(self, name):
        if os.path.exists(name) and os.path.isfile(name):
            directory = os.path.dirname(name)
            fname, _ = os.path.splitext(os.path.basename(name))
        else:
            directory = "configs"
            fname = name
        self.settings = self._get_config_from_json(directory, fname)

    @property
    def model_dir(self):
        iteration = self.settings.iteration if "iteration" in self.settings else "default"

        return os.path.join(
            "data\\experiments",
            self.settings.model.name.split(".")[-1],
            self.settings.model.experiment if hasattr(self.settings.model, "experiment") else self.settings.experiment,
            iteration
        )

    @property
    def k(self):
        return len(self.settings.dataset.params.split_ratio)

    def save(self, model_dir):
        create_dirs([model_dir])
        fpath = os.path.join(model_dir, "config.json")
        with open(fpath,'w') as file:
            json.dump(self.settings, file, sort_keys=False, indent=4)

    def update(self, settings, iteration):
        settings = dictionary.merge(self.settings, settings)
        self.settings = dictionary.to_bunch(settings)
        self.settings.iteration = iteration

    def _get_config_from_json(self, directory, name):
        config_dict = {}
        parts = name.split(".")
        
        for i in range(1, len(parts) + 1):
            fname = ".".join(parts[:i]) + ".json"
            fpath = os.path.join(directory, fname)

            if os.path.lexists(fpath):
                # parse the configurations from the config json file provided
                with open(fpath, 'r') as config_file:
                    config_dict = dictionary.merge(config_dict, json.load(config_file))

        # convert the dictionary to a namespace using bunch lib
        #config = Bunch(config_dict)
        config = dictionary.to_bunch(config_dict)

        return config