import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.0006), math.log(0.009)),
            "optimizer.params.weight_decay": hp.loguniform("optimizer.params.weight_decay", math.log(0.00001), math.log(0.0001)),
            "dropout": hp.uniform("dropout", 0.3, 0.7)#,
            #"filters_num": hp.quniform("filters_num", 48, 66, q=6),
            #"fc_units": hp.quniform("fc_units", 35, 45, q=5)
        }

    def _dataset_params_space(self):
        return {}