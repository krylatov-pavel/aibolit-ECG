import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.0003), math.log(0.006)),
            "optimizer.params.weight_decay": hp.loguniform("optimizer.params.weight_decay", math.log(0.0001), math.log(0.0004)),
            "dropout": hp.uniform("dropout", 0.4, 0.6),
            "filters_num": hp.quniform("filters_num", 36, 48, q=2)
        }

    def _dataset_params_space(self):
        return {}