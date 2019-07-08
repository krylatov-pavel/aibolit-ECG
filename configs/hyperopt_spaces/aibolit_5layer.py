import math
from hyperopt import hp
from configs.hyperopt_spaces.base_space import BaseSpace

class Space(BaseSpace):
    def _model_hparams_space(self):
        return {
            "optimizer.params.lr": hp.loguniform("optimizer.params.lr", math.log(0.003), math.log(0.009)),
            "dropout": hp.uniform("dropout", 0.3, 0.4),
            "filters_num": hp.quniform("filters_num", 48, 72, q=6),
            "fc_units": hp.quniform("fc_units", 30, 50, q=5),
            "fc_layers": hp.quniform("fc_layers", 2, 3, q=1)#,
            #"activation_fn": hp.choice("activation_fn", ["torch.nn.ReLU", "torch.nn.ELU", "models.extensions.swish.Swish"]),
        }

    def _dataset_params_space(self):
        return {}