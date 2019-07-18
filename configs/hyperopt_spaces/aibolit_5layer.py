import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.004), math.log(0.006)),
            "optimizer.params.weight_decay": hp.loguniform("optimizer.params.weight_decay", math.log(0.00001), math.log(0.0003)),
            "dropout": hp.uniform("dropout", 0.35, 0.45)#,
            #"filters_num": hp.quniform("filters_num", 54, 72, q=6),
            #"fc_units": hp.quniform("fc_units", 30, 50, q=5),
            #"fc_layers": hp.quniform("fc_layers", 2, 3, q=1),
        }

    def _dataset_params_space(self):
        return {}