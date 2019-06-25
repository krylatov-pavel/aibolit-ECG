import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "learning_rate": hp.loguniform("learning_rate", math.log(0.0001), math.log(0.1)),
            "dropout_rate": hp.uniform("dropout_rate", 0.3, 0.7),
            "filters_num": hp.choice("filters_num", [32, 48, 64]),
            "filters_step": hp.choice("filters_step", [32, 64]),
            "conv_kernel_size": hp.choice("conv_kernel_size", [3, 5, 7]),
            "conv_layer_num": hp.choice("conv_layer_num", [3, 4, 5, 6, 7]),
            "fc_units": hp.choice("fc_units", [40, 60, 100, 120])
        }

    def _dataset_params_space(self):
        return {}