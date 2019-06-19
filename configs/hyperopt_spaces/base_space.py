class BaseSpace(object):
    def space(self):
        model_hparams = {"model.hparams.{}".format(key): value for key, value in self._model_hparams_space().items()}
        dataset_hparams = {"dataset.params.{}".format(key): value for key, value in self._dataset_params_space().items()}
        space = dict(model_hparams, **dataset_hparams)

        return space

    def _model_hparams_space(self):
        raise NotImplementedError()

    def _dataset_params_space(self):
        raise NotImplementedError()