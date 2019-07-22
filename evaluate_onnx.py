import argparse
import os
import torch.utils.data
import onnxruntime as rt
import numpy as np
from datasets.common.dataset import Dataset
from training.metrics.confusion_matrix import ConfusionMatrix
from utils.helpers import get_class
from utils.config import Config 
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

        dataset_generator = get_class(config.settings.dataset.dataset_generator)(config.settings.dataset.params, config.settings.dataset.sources)
        examples_provider = get_class(config.settings.dataset.examples_provider)
        batch_size = config.settings.model.hparams.eval_batch_size
        label_map = { lbl: c.label_map for lbl, c in config.settings.dataset.params.class_settings.items() }
        
        examples = examples_provider(
            folders=dataset_generator.test_set_path(),
            label_map=label_map,
            equalize_labels=True
        )

        if config.settings.dataset.params.normalize_input:
            mean, std = dataset_generator.stats
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean], std=[std]),
                transforms.Lambda(squeeze)
            ])
        else:
            transform = None 

        dataset = Dataset(examples, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        cm = ConfusionMatrix([], [], len(label_map))

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
        class_map = {value: key for key, value in label_map.items()}
        class_accuracy = cm.class_accuracy()
        for i, acc in enumerate(class_accuracy):
            print("accuracy\t{}\t{:.3f}".format(class_map[i], acc))
        cm.plot(os.path.join(config.model_dir, "TEST_confusion_matrix.png"), class_map=class_map)
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()