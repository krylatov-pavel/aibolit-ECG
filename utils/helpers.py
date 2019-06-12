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