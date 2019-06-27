import argparse
import utils.helpers as helpers
from data_analysis.find_set_overlap import find_set_overlap
from utils.config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)
    parser.add_argument("--overlap", "-o", dest="check_overlap", action="store_true")
    parser.set_defaults(check_overlap=False)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        dataset_provider = helpers.get_class(config.settings.dataset.dataset_provider)(config.settings.dataset.params)

        if args.check_overlap:
            find_set_overlap(dataset_provider.examples_dir)
    else:
        print("configuration file name is required. use -h for help")

if __name__=="__main__":
    main()
