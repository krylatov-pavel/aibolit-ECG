import os
import argparse
from training.experiment import Experiment
from utils.mutable_config import MutableConfig
from utils.dirs import create_dirs

def iteration_name_generator(num, directory):
    if os.path.exists(directory):
        existing_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    else:
        existing_names = []

    i = 0
    while num > 0:
        if not str(i) in existing_names:
            num -= 1
            yield str(i)
        i +=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config file name (wihout extension)", type=str)
    parser.add_argument("--iterations", "-i", help="Number of search iterations", type=int)

    args = parser.parse_args()

    if args.config:
        config = MutableConfig(args.config)

        for iteration in iteration_name_generator(args.iterations, os.path.dirname(config.model_dir)):
            print("search iteration ", iteration)

            settings = config.mutate(iteration)
            experiment = Experiment(settings, config.model_dir)

            create_dirs([config.model_dir])
            config.save(config.model_dir)

            experiment.run()
            experiment.plot_metrics()
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()