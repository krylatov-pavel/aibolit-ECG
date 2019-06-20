import argparse
import os
import json
import numpy as np
import pandas as pd
import training.metrics.stats as stats
from utils.config import Config

def get_search_params(model_dir):
    fpath = os.path.join(model_dir, "params.json")
    if os.path.isfile(fpath):
        with open(fpath, "r") as file:
            params = json.load(file)
            params = {key.split(".")[-1]: value for key, value in params.items() }
            return params
    else:
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config file name (wihout extension)", type=str)
    parser.add_argument("--path", "-p", help="Path to model experiments directory", type=str)

    args = parser.parse_args()

    if args.path:
        path = args.path
    elif args.config:
        path = os.path.dirname(Config(args.config).model_dir) 
    else:
        print("--config or --path parameter required")
        return

    dirs = (os.path.join(path, d) for d in os.listdir(path))
    confs = (os.path.join(d, "config.json") for d in dirs if os.path.isdir(d))
    confs = (c for c in confs if os.path.isfile(c))
    confs = (Config(c) for c in confs)

    reports = []
    for c in confs:
        accuracy, step = stats.max_accuracy(c.model_dir, c.k)
        params = get_search_params(c.model_dir)

        report = {
            "iteration": c.settings.iteration,
            "accuracy": accuracy,
            "step": str(step)
        }
        report = dict(report, **params)
        reports.append(report)

    reports = pd.DataFrame(reports)
    reports = reports.sort_values("accuracy", ascending=False)

    print(reports)

if __name__ == "__main__":
    main()