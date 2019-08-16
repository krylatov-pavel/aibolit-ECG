import argparse
import os
import utils.helpers as helpers
import torch
import utils.transforms as transforms
from datasets.common.dataset import Dataset
from utils.config import Config
from training.spec import EvalSpec
from training.model import Model
import training.metrics.stats as stats
from training.metrics.confusion_matrix import ConfusionMatrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)

        net = helpers.get_class(config.settings.model.name)(config.settings)
        dataset_generator = helpers.get_class(config.settings.dataset.dataset_generator)(config.settings.dataset.params, config.settings.dataset.sources)
        examples_provider_ctor = helpers.get_class(config.settings.dataset.examples_provider)
        seed = config.settings.dataset.params.get("seed") or 0
        
        class_map = { val.label_map: key for key, val in config.settings.dataset.params.class_settings.items() }
        label_map = { val: key for key, val in class_map.items() }
        loss_fn = config.settings.get("loss_fn") or {}
        loss_fn_type = loss_fn.get("type")
        loss_fn_params = loss_fn.get("params")
        classifier_type = config.settings.get("classifier")
        eval_spec = EvalSpec(
            dataset=None,
            class_num=config.class_num,
            batch_size=50,
            class_map=class_map,
            loss_fn_type=loss_fn_type,
            loss_fn_params=loss_fn_params,
            classifier_type=classifier_type,
            classifier_params={
                "label_map": label_map 
            }
        )

        _, checkpoints = stats.max_accuracy(config.model_dir, config.k)

        if config.settings.dataset.params.normalize_input:
            transform = transforms.get_transform()
        else:
            transform = None
        
        confusion_mtrx = ConfusionMatrix([], [], config.class_num)
        if config.k == 2:
            examples = examples_provider_ctor(
                folders=dataset_generator.eval_set_path(),
                label_map=label_map,
                equalize_labels=True,
                seed=seed
            )
            eval_spec.dataset = Dataset(examples, transform=transform)
            model = Model.restore(net, config.model_dir, checkpoints)
            _, cm = model.evaluate(eval_spec)
            confusion_mtrx.add(cm)
        else:
            for i in range(config.k):
                examples = examples_provider_ctor(
                    folders=dataset_generator.eval_set_path(eval_fold_number=i),
                    label_map=label_map,
                    equalize_labels=True,
                    seed=seed
                )
                eval_spec.dataset = Dataset(examples, transform=transform)
                model = Model.restore(net, os.path.join(config.model_dir, "fold_{}".format(i)), checkpoints[i])
                _, cm = model.evaluate(eval_spec)
                confusion_mtrx.add(cm)

        confusion_mtrx.plot(os.path.join(config.model_dir, "confusion_matrix.png"), class_map)

    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()