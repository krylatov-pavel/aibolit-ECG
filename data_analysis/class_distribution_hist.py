import os
import json
import csv
import numpy as np
import pandas as pd
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

def dataset_data(dirs, fs):
    data = []

    name = NameGenerator(".csv")
    wave = WavedataProvider()
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
    return db

def main():
    db = database_data("C:\\Study\\aibolit-ECG\\data\\database\\aibolit", 1000, ["N", "n"])
    db = relative_duration(db)
    print(db)
    
    ds = dataset_data([
        "C:\\Study\\aibolit-ECG\\data\\examples\\aibolit\\2fold_3s_(2N,R,AV)_250hz\\0",
        "C:\\Study\\aibolit-ECG\\data\\examples\\aibolit\\2fold_3s_(2N,R,AV)_250hz\\1"
    ], 250)
    ds = relative_duration(ds)
    print(ds)

if __name__=="__main__":
    main()