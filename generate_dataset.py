import argparse
from utils.config import Config
from utils.helpers import get_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)

        examples_provider = get_class(config.settings.dataset.provider)(config.settings.dataset.params)
        examples_provider.generate()
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()