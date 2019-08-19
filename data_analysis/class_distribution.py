import argparse
import os
import json
import h5py
import numpy as np
import pandas as pd
import utils.helpers as helpers
from utils.config import Config

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

    fnames = (os.path.join(root, "{}.hdf5".format(f)) for f in folds)
    for fname in fnames:
        with h5py.File(fname, "r") as f:
            fold_data = [None] * len(f.keys())
            for i, example in enumerate(f.keys()):
                label = f[example].attrs["label"].decode("utf-8")
                signal = f[example][:]
                duration = len(signal) / fs
                duration_clipped = len(list(filter(lambda s: s > 5, signal))) / fs
                duration_low = len(list(filter(lambda s: s < -4, signal))) / fs
                
                fold_data[i] = {
                    "label": label,
                    "duration": duration,
                    "duration_clipped": duration_clipped,
                    "duration_low": duration_low
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

        dataset_generator = helpers.get_class(config.settings.dataset.dataset_generator)(config.settings.dataset.params, config.settings.dataset.sources)
        
        if args.db:
            db = database_data(
                os.path.join("data", "database", config.settings.dataset.sources[0].name),
                1000,
                ["N", "n"]
            )
            db = relative_duration(db)
            print(db)
        
        if args.ds:
            ds = dataset_data(
                dataset_generator.examples_dir,
                [str(i) for i in range(config.k)],
                config.settings.dataset.params.get("example_fs")
            )

            duration_total = ds.duration.sum() 
            duration_clip =  ds.duration_clipped.sum() 
            duration_low = ds.duration_low.sum()
            print("above 20 sigma: {:.3f}".format((duration_clip / duration_total) * 100))
            print("below -20 sigma: {:.3f}".format((duration_low / duration_total) * 100))

            ds = relative_duration(ds)
            print(ds)


        
    else:
        print("configuration file name is required. use -h for help")

if __name__=="__main__":
    main()