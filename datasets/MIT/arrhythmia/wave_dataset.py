from datasets.MIT.wave_dataset import WaveDataset
from datasets.Arythmia.wave_examples_provider import WaveExamplesProvider

class ArythmiaWaveDataset(WaveDataset):
    def __init__(self, params):
        super(ArythmiaWaveDataset, self).__init__(params)
        self.examples_provider = WaveExamplesProvider(params)