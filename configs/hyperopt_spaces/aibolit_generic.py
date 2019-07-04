import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "learning_rate": hp.loguniform("learning_rate", math.log(0.0001), math.log(0.1)),
            "dropout_rate": hp.uniform("dropout_rate", 0.3, 0.7),
            "filters_num": hp.choice("filters_num", [48, 64, 90]),
            "filters_step": hp.choice("filters_step", [32, 64]),
            "conv_kernel_size": hp.choice("conv_kernel_size", [5, 7, 9]),
            "conv_layer_num": hp.choice("conv_layer_num", [5, 6, 7, 8]),
            "fc_units": hp.choice("fc_units", [40, 60, 100, 120]),
            "fc_layers": hp.choice("fc_layers", [2, 3, 4])
        }

    def _dataset_params_space(self):
        return {}