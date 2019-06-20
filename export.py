import argparse
import training.metrics.stats as stats
from training.experiment import Experiment
from utils.config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)
    parser.add_argument("--checkpoint", help="Checkpoint number", type=int)
    parser.add_argument("--best", dest="use_best", action="store_true")
    parser.set_defaults(use_best=False)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        
        experiment = Experiment(config.settings, config.model_dir)

        if args.use_best:
            print("exporting best checkpoint")
            _, checkpoints = stats.max_accuracy(config.model_dir, config.k)
            experiment.export(checkpoint=checkpoints)
        elif args.checkpoint:
            print("exporting checkpoint ", args.checkpoint)
            checkpoints = [args.checkpoint] * config.k
            experiment.export(checkpoint=checkpoints)
        else:
            print("exporting latest checkpoint")
            experiment.export()

    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()