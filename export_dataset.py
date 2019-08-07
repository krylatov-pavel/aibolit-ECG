import argparse
import os
import numpy as np
import torch
import utils.transforms as transforms
from datasets.common.dataset import Dataset
from utils.config import Config
from utils.helpers import get_class
import codecs, json 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)
    parser.add_argument("--path", "-p", help="file path", type=str)

    args = parser.parse_args()
    if args.config:
        config = Config(args.config)

        dataset_generator = get_class(config.settings.dataset.dataset_generator)(config.settings.dataset.params, config.settings.dataset.sources)
        examples_provider = get_class(config.settings.dataset.examples_provider)
        batch_size = config.settings.model.hparams.eval_batch_size
        seed = config.settings.dataset.params.get("seed") or 0
        label_map = { lbl: c.label_map for lbl, c in config.settings.dataset.params.class_settings.items() }
        
        examples = examples_provider(
            folders=dataset_generator.eval_set_path(),
            label_map=label_map,
            equalize_labels=True,
            seed=seed
        )

        if config.settings.dataset.params.normalize_input:
            transform = transforms.get_transform()
        else:
            transform = None

        dataset = Dataset(examples, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        X = []
        Y = []
        for i, data in enumerate(data_loader):
            x, labels = data
            X.extend(x.numpy().tolist())
            Y.extend(labels.numpy().tolist())

        file_path = os.path.join(args.path, "x.json")   ## your path variable
        json.dump(X, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        file_path = os.path.join(args.path, "y.json")   ## your path variable
        json.dump(Y, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        file_path = os.path.join(args.path, "x_single.json")   ## your path variable
        json.dump(X[0: 1], codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        file_path = os.path.join(args.path, "y_single.json")   ## your path variable
        json.dump(Y[0: 1], codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()