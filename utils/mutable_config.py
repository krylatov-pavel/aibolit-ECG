import copy
import random
import math
from utils.config import Config

class MutableConfig(Config):
    def __init__(self, name):
        super(MutableConfig, self).__init__(name)

        self._original_settings = copy.deepcopy(self.settings)

        mutations_config = self._get_config_from_json("configs\\mutations", name)
        self._model_mutation_rules = \
            [self._create_mutatation_rule(key, value) for key, value in mutations_config.hparams.items()]
        self._data_mutation_rules = \
            [self._create_mutatation_rule(key, value) for key, value in mutations_config.params.items()]

    def mutate(self, iteration):
        self.settings = copy.deepcopy(self._original_settings)

        for rule in self._model_mutation_rules:
            self._apply_rule(self.settings.model.hparams, rule)
        
        for rule in self._data_mutation_rules:
            self._apply_rule(self.settings.dataset.params, rule)

        self.settings.iteration = iteration

        return self.settings

    def _apply_rule(self, obj, rule):
        name = rule["hparam"]
        value = self._generate_value(rule)

        if name in obj:
            obj[name] = value
        else:
            raise ValueError("theres' no '{}' parameter in configuration object. make sure that name is correct".format(name))

    def _create_mutatation_rule(self, name, config):
        rule = {
            "hparam": name
        }

        if "options" in config:
            rule["options"] = config.options

        if "range" in config:
            rule["range"] = config.range
            rule["logscale"] = config.logscale if "logscale" in config else False

        return rule

    def _generate_value(self, rule):
        if "options" in rule:
            return random.choice(rule["options"])
        elif "range" in rule:
            a, b = rule["range"]
            if rule["logscale"]:
                return math.exp(random.uniform(math.log(a), math.log(b)))
            else:
                return random.uniform(a, b)
        else:
            raise ValueError("Rule must specify either mutation 'options' or mutation 'range'")