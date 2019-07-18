import argparse
import os
import numpy as np
import training.metrics.stats as stats
from training.experiment import Experiment
from utils.config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config file name (wihout extension)", type=str)
    parser.add_argument("--iterations", "-i", help="Number of search iterations", type=int)
    
    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        base_iteration = config.settings.iteration
        base_accuracy, _ = stats.max_accuracy(config.model_dir, config.k)
        
        accuracy = [None] * args.iterations
        for i in range(args.iterations):
            iteration = "{}.{}".format(base_iteration, i + 1)
            config.update({}, iteration)
            if not os.path.exists(config.model_dir):
                config.save(config.model_dir)
                experiment = Experiment(config.settings, config.model_dir)
                experiment.run()
                experiment.plot_metrics()

            accuracy[i], _ = stats.max_accuracy(config.model_dir, config.k)

        accuracy.append(base_accuracy)

        print("max:\t{:.3f}".format(np.max(accuracy)))
        print("min:\t{:.3f}".format(np.min(accuracy)))
        print("mean:\t{:.3f}".format(np.mean(accuracy)))
        print("std:\t{:.3f}".format(np.std(accuracy)))
    else:
        print("configuration file name is required. use -h for help")
if __name__ == "__main__":
    main()