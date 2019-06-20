import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "learning_rate": hp.loguniform("learning_rate", math.log(0.001), math.log(0.1))
        }

    def _dataset_params_space(self):
        return {}