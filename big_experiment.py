
import os
import json
from hyperopt import fmin, tpe
import training.metrics.stats as stats
import utils.helpers as helpers
import utils.dictionary as dictionary
from utils.config import Config
from training.experiment import Experiment

def save_json(model_dir, fname, params):
    fpath = os.path.join(model_dir, fname)
    with open(fpath, "w") as file:
        json.dump(params, file, sort_keys=False, indent=4)

def iteration_name_generator(num, directory, fs, duration):
    prefix = "{}s_{}hz_".format(fs, duration)
    if os.path.exists(directory):
        existing_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        existing_names = [d for d in existing_names if d.startswith(prefix)]
    else:
        existing_names = []

    i = 0
    while num > 0:
        name = prefix + str(i)
        if not name in existing_names:
            num -= 1
            yield name
        i +=1

def build_objective_fn(config, name_generator):
    def objective_fn(params):
        iteration = next(name_generator)

        settings = {}
        for key, value in params.items():
            settings = dictionary.unroll(settings, key, value)

        config.update(settings, iteration)
        config.save(config.model_dir)
        save_json(config.model_dir, "params.json", params)

        print("iteration {}".format(iteration))
        experiment = Experiment(config.settings, config.model_dir)
        try:
            experiment.run()
        except Exception as e:
            print(e)

        stats.plot_metrics(config.model_dir, config.k)
        mean, std, min, max = stats.accuracy_stats(config.model_dir, config.k)
        save_json(config.model_dir, "accuracy.json", {
            "mean": mean,
            "std": std,
            "min": min,
            "max": max
        })

        return -mean

    return objective_fn

def main():
    config_tlmp = "aibolit_{}s"
    fs_options = [250, 500, 1000]
    duration_options = [3, 3.2, 3.4, 3.6, 3.8, 4]
    iterations = 20

    for duration in duration_options:
        config = Config(config_tlmp.format(duration))
        params_space = helpers.get_class(config.settings["params_space"])().space()
        for fs in fs_options:
            print("\n\n{}s {}hz".format(duration, fs))
            config.settings.dataset.params.example_fs = fs

            #generate dataset
            dataset_provider = helpers.get_class(config.settings.dataset.dataset_generator)(config.settings.dataset.params, config.settings.dataset.sources)
            dataset_provider.generate()

            #hp search
            name_generator = iteration_name_generator(iterations, config.model_dir, fs, duration)
            objective = build_objective_fn(config, name_generator)
        
            best = fmin(
                fn=objective,
                space=params_space,
                algo=tpe.suggest,
                max_evals=iterations
            )

            print(best)

if __name__=="__main__":
    main()