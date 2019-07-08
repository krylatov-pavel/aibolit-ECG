import argparse
import os
import torch.utils.data
import onnxruntime as rt
import numpy as np
from datasets.common.dataset import Dataset
from datasets.common.examples_provider import ExamplesProvider
from training.metrics.confusion_matrix import ConfusionMatrix
from utils.helpers import get_class
from utils.config import Config 
from datasets.MIT.common.wavedata_provider import WavedataProvider
from torch.utils import data
from torchvision import transforms

def squeeze(x):
    return torch.squeeze(x, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)

    args = parser.parse_args()
    if args.config:
        config = Config(args.config)

        dataset_provider = get_class(config.settings.dataset.dataset_provider)(config.settings.dataset.params)
        file_reader = get_class(config.settings.dataset.file_provider)()
        batch_size = config.settings.model.hparams.eval_batch_size
        
        examples = ExamplesProvider(
            folders=dataset_provider.test_set_path(),
            file_reader=file_reader,
            label_map=config.settings.dataset.params.label_map,
            equalize_labels=True
        )

        if config.settings.dataset.params.normalize_input:
            mean, std = dataset_provider.stats
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean], std=[std]),
                transforms.Lambda(squeeze)
            ])
        else:
            transform = None 

        dataset = Dataset(examples, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        cm = ConfusionMatrix([], [], len(config.settings.dataset.params.label_map))

        fpath = os.path.join(config.model_dir, "model.onnx")
        sess = rt.InferenceSession(fpath)
        input_name = sess.get_inputs()[0].name

        for i, data in enumerate(data_loader):
            x, labels = data
            print("examples ", (i + 1) * batch_size)
            predictions = sess.run(None, {input_name: x.numpy()})
            predictions = predictions[0]
            predictions = np.argmax(predictions, axis=1)
            cm.append(predictions, labels)

        print("accuracy: ", cm.accuracy())
        class_map = {value: key for key, value in config.settings.dataset.params.label_map.items()}
        class_accuracy = cm.class_accuracy()
        for i, acc in enumerate(class_accuracy):
            print("accuracy\t{}\t{:.3f}".format(class_map[i], acc))
        cm.plot(os.path.join(config.model_dir, "TEST_confusion_matrix.png"), class_map=class_map)
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()