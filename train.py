import argparse
from utils.config import Config
from training.experiment import Experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        config.save(config.model_dir)
        
        experiment = Experiment(config.settings, config.model_dir)
        experiment.run()
        experiment.plot_metrics()
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()