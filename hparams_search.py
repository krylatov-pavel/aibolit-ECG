import os
import argparse
from hyperopt import fmin, tpe
from training.experiment import Experiment
import training.metrics.stats as stats
from utils.config import Config
from utils.helpers import get_class
import utils.dictionary as dictionary

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

def build_objective_fn(config, name_generator):
    def objective_fn(params):
        iteration = next(name_generator)

        settings = {}
        for key, value in params.items():
            settings = dictionary.unroll(settings, key, value)

        config.update(settings, iteration)
        config.save(config.model_dir)

        experiment = Experiment(config.settings, config.model_dir)
        experiment.run()

        stats.plot_metrics(config.model_dir, config.k)
        acc = stats.max_accuracy(config.model_dir, config.k)

        return -acc

    return objective_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config file name (wihout extension)", type=str)
    parser.add_argument("--iterations", "-i", help="Number of search iterations", type=int)
    parser.add_argument("--space", "-s", help="Name of hparams search space file", type=str)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        name_generator = iteration_name_generator(args.iterations, os.path.dirname(config.model_dir))
        objective = build_objective_fn(config, name_generator)
        space = get_class(args.space)().space()
        
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=args.iterations
        )

        print(best)
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()