import argparse
import os
import torch.utils.data
import onnxruntime as rt
import numpy as np
from datasets.base.dataset import Dataset
from training.metrics.confusion_matrix import ConfusionMatrix
from utils.helpers import get_class
from utils.config import Config 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)

    args = parser.parse_args()
    if args.config:
        config = Config(args.config)

        batch_size = 32
        examples_provider = get_class(config.settings.dataset.provider)(config.settings.dataset.params)
        dataset = Dataset(examples_provider, list(range(len(config.settings.dataset.params.split_ratio))))
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
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()