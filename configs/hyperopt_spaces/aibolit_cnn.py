import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.004), math.log(0.006)),
            "optimizer.params.weight_decay": hp.loguniform("optimizer.params.weight_decay", math.log(0.00001), math.log(0.00003)),
            "dropout": hp.uniform("dropout", 0.38, 0.45),
            "filters_num": hp.quniform("filters_num", 32, 48, q=6),
            "fc_units": hp.quniform("fc_units", 40, 50, q=5)
        }

    def _dataset_params_space(self):
        return {}