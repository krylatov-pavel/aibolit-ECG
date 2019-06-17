import argparse
import os
import numpy as np
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

        examples_provider = get_class(config.settings.dataset.provider)(config.settings.dataset.params)

        X = []
        Y = []
        all_folds = list(range(len(config.settings.dataset.params.split_ratio)))
        for i in range(examples_provider.len(all_folds)):
            x, y = examples_provider.get_example(i, all_folds, False)
            X.append(x.tolist())
            Y.append(y)

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