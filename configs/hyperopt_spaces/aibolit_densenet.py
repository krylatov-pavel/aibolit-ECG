import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.0004), math.log(0.0006)),
            "optimizer.params.weight_decay": hp.loguniform("optimizer.params.weight_decay", math.log(0.0001), math.log(0.0002)),
            "growth_rate": hp.quniform("growth_rate", 8, 14, q=2),
            "compression": hp.quniform("compression", 0.26, 0.34, q=0.02),
            "dropout": hp.quniform("dropout", 0.05, 0.15, q=0.05),
            "depth": hp.quniform("depth", 7, 12, q=1)
        }

    def _dataset_params_space(self):
        return {}