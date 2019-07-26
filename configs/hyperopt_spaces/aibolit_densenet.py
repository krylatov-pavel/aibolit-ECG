import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.0001), math.log(0.005)),
            "optimizer.params.weight_decay": hp.loguniform("optimizer.params.weight_decay", math.log(0.00001), math.log(0.0004)),
            "growth_rate": hp.quniform("growth_rate", 4, 12, q=2),
            "kernel_size": hp.choice("kernel_size", [5, 7, 9, 11]),
            "compression": hp.quniform("compression", 0.3, 0.6, q=0.1)
        }

    def _dataset_params_space(self):
        return {}