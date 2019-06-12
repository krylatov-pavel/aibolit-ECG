import numpy as np

def get_class(name):
    parts = name.split(".")
    module_name = ".".join(parts[:-1])

    module = __import__(module_name)

    for constructor in parts[1:]:
        module = getattr(module, constructor)
    
    return module

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def unzip_list(l):
    return map(list, zip(*l))

def rescale(x, old_min, old_max, new_min, new_max):
    x_arr = np.array(x)
    rescaled_x = (new_max - new_min) * (x_arr - old_min) / (old_max - old_min) + new_min
    return rescaled_x

def normalize(x, mean, std):
    x_arr = np.array(x)
    normalized_x = (x_arr - mean) / std
    return normalized_x

def avg_f1_score(labels, predictions):
    epsilon = 1e-10

    def f1_score(prec, recall):
        return 2 * prec * recall / (prec + recall + epsilon)

    stats = {class_name: {} for class_name in set(labels)}

    for class_name in stats.keys():
        tp = len([True for label, pred in zip(labels, predictions) if pred == class_name and pred == label])
        fp = len([True for label, pred in zip(labels, predictions) if pred == class_name and pred != label])
        fn = len([True for label, pred in zip(labels, predictions) if label == class_name and pred != label])
        
        stats[class_name]["tp"] = tp
        stats[class_name]["fp"] = fp
        stats[class_name]["fn"] = fn

        prec = tp / (tp + fp + epsilon)
        rec= tp / (tp + fn + epsilon)

        print("class :", class_name)
        print("precision: ", prec)
        print("recall: ", rec)
        print("f1 score: ", f1_score(prec, rec))

    print("\n\n")
    for class_name, val in stats.items():
        print("{} fp: {}  fn: {}".format(class_name, val["fp"], val["fn"]))
    print("\n\n")

    print("fp sum: {}".format(sum(val["fp"] for val in stats.values())))
    print("fn sum: {}".format(sum(val["fn"] for val in stats.values())))
                     
    micro_avg_prec = sum(val["tp"] for val in stats.values()) / sum(val["tp"] + val["fp"] + epsilon for val in stats.values())
    micro_avg_recall = sum(val["tp"] for val in stats.values()) / sum(val["tp"] + val["fn"] + epsilon for val in stats.values())
    micro_avg_f1 = f1_score(micro_avg_prec, micro_avg_recall)

    print("micro precision:", micro_avg_prec)
    print("micro recall:", micro_avg_recall)

    macro_avg_prec = sum(val["tp"] / (val["tp"] + val["fp"] + epsilon) for val in stats.values()) / len(stats)
    macro_avg_recall = sum(val["tp"] / (val["tp"] + val["fn"] + epsilon) for val in stats.values()) / len(stats)
    macro_avg_f1 = f1_score(macro_avg_prec, macro_avg_recall)

    print("\nmacro precision:", macro_avg_prec)
    print("macro recall:", macro_avg_recall)

    print("\nmicro f1 score: ", micro_avg_f1)
    print("macro f1 score: ", macro_avg_f1)

    return micro_avg_f1, macro_avg_f1