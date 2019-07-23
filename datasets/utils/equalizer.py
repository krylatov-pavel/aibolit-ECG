import numpy as np
import pandas as pd
import random

def equalize(data, key_fn, distribution):
    """
    distribution: { 
        class1: equalize_distribution1,
        class2: equalize_distribution2,
        ...
    }
    consider class with equalize_distribution = 1 as base and caclulate distribution of other classes relative to this one.
    if multiple classes have equalize_distribution = 1 then consider the smalest one as base class
    if all classes have equalize_distribution = 1 then euqlize all classes equally with base class as the smallest one
    if equalize_distribution is None then keep class distribution as is
    """
    equalized = []

    keys = (key_fn(item) for item in data)
    class_counts = { key: count for key, count in pd.Series(keys).value_counts().items() }
    base_count = min(class_counts[key] for key, distr in distribution.items() if distr == 1)

    for key, distr in distribution.items():
        class_items = [item for item in data if key_fn(item) == key]

        if distr:
            take = min(int(base_count * distr), len(class_items))
        else:
            take = len(class_items)

        random.Random(0).shuffle(class_items)
        equalized.extend(class_items[:take])

    return equalized