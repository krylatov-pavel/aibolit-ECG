import argparse
from utils.config import Config
from utils.experiment import Experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        
        experiment = Experiment(config.settings, config.model_dir)
        experiment.run()
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()