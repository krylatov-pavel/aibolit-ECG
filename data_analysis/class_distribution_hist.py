import argparse
import os
import json
import csv
import numpy as np
import pandas as pd
import utils.helpers as helpers
from utils.config import Config
from datasets.utils.name_generator import NameGenerator
from datasets.common.wavedata_provider import WavedataProvider

def database_data(path, fs, exclude=None):
    data = []
    exclude = exclude or []

    dirs = (os.path.join(path, d) for d in os.listdir(path))
    dirs = (d for d in dirs if os.path.isdir(d))
    dirs = (d for d in dirs if not(os.path.basename(d) in exclude))

    for class_dir in dirs:
        fnames = (os.path.join(class_dir, f) for f in os.listdir(class_dir))
        fnames = [f for f in fnames if os.path.isfile(f) and f.endswith(".json")]

        class_data = [None] * len(fnames)
        for i, f in enumerate(fnames):
            with open(f, "r") as json_file:
                signal = json.load(json_file)
                class_data[i] = {
                    "label": os.path.basename(class_dir).upper(),
                    "duration": len(signal) / fs
                }
        data.extend(class_data)

    return pd.DataFrame(data)

def dataset_data(root, folds, fs):
    data = []

    name = NameGenerator(".csv")
    wave = WavedataProvider()

    dirs = (os.path.join(root, f) for f in folds)
    for dir in dirs:
        fnames = (os.path.join(dir, f) for f in os.listdir(dir))
        fnames = [f for f in fnames if os.path.isfile(f) and f.endswith(".csv")]

        fold_data = [None] * len(fnames)
        for i, fname in enumerate(fnames):
            label = name.get_metadata(os.path.basename(fname)).label

            signal = wave.read(fname)
            duration = len(signal[0]) / fs
            
            fold_data[i] = {
                "label": label,
                "duration": duration
            }
        
        data.extend(fold_data)

    return pd.DataFrame(data)

def relative_duration(db):
    db = db.groupby("label").duration.sum().to_frame()
    db["percentage"] = db.duration / db.duration.sum() * 100
    db = db.sort_values("duration")
    return db

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)
    parser.add_argument("--db", "-b", dest="db", action="store_true")
    parser.add_argument("--ds", "-s", dest="ds", action="store_true")
    parser.set_defaults(db=False)
    parser.set_defaults(ds=False)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)

        dataset_provider = helpers.get_class(config.settings.dataset.dataset_provider)(config.settings.dataset.params)
        
        if args.db:
            db = database_data(
                "D:\\Study\\Aibolit-ECG\\data\\database\\aibolit",
                config.settings.dataset.params.get("fs"),
                ["N", "n", "ISH"]
            )
            db = relative_duration(db)
            print(db)
        
        if args.ds:
            ds = dataset_data(
                dataset_provider.examples_dir,
                [str(i) for i in range(config.k)],
                config.settings.dataset.params.get("resample_fs")
            )
            ds = relative_duration(ds)
            print(ds)
        
    else:
        print("configuration file name is required. use -h for help")

if __name__=="__main__":
    main()