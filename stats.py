import argparse
import os
import numpy as np
import pandas as pd
import training.metrics.stats as stats
from utils.config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config file name (wihout extension)", type=str)
    parser.add_argument("--path", "-p", help="Path to model experiments directory", type=str)

    args = parser.parse_args()

    if args.path:
        dirs = (os.path.join(args.path, d) for d in os.listdir(args.path))
        confs = (os.path.join(d, "config.json") for d in dirs if os.path.isdir(d))
        confs = (c for c in confs if os.path.isfile(c))
        confs = (Config(c) for c in confs)

        report = []
        for c in confs:
            accuracy = stats.max_accuracy(c.model_dir, c.k)
            report.append({
                "accuracy": accuracy,
                "iteration": c.settings.iteration
                #add config parameters here if needed
            })

        report = pd.DataFrame(report)
        report = report.sort_values("checkpoint_accuracy", ascending=False)

        print(report)
    elif args.config:
        raise NotImplementedError()
    else:
        print("--config or --path parameter required")

if __name__ == "__main__":
    main()
