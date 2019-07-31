import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.0005), math.log(0.001)),
            "optimizer.params.weight_decay": hp.loguniform("optimizer.params.weight_decay", math.log(0.00008), math.log(0.0002)),
            "growth_rate": hp.quniform("growth_rate", 6, 12, q=2),
            "compression": hp.quniform("compression", 0.3, 0.36, q=0.02),
            "dropout": hp.quniform("dropout", 0.1, 0.2, q=0.05),
            "depth": hp.quniform("depth", 7, 12, q=1)
        }

    def _dataset_params_space(self):
        return {}