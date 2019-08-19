import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.0003), math.log(0.0009)),
            "optimizer.params.weight_decay": hp.loguniform("optimizer.params.weight_decay", math.log(0.00009), math.log(0.0002)),
            "growth_rate": hp.quniform("growth_rate", 6, 14, q=2),
            "compression": hp.quniform("compression", 0.2, 0.4, q=0.02),
            "dropout": hp.quniform("dropout", 0.05, 0.15, q=0.05),
            "depth": hp.quniform("depth", 4, 11, q=1),
            "kernel_size": hp.choice("kernel_size", [3, 5, 7, 9, 11]),
        }

    def _dataset_params_space(self):
        return {}