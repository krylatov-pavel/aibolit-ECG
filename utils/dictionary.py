from bunch import Bunch

def merge(a, b):
    for key in b.keys():
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                a[key] = merge(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]     
    
    return a

def to_bunch(dictionary):
    if isinstance(dictionary, dict):
        bunch = Bunch(dictionary)

        for key in bunch.keys():
            if isinstance(bunch[key], dict):
                bunch[key] = to_bunch(bunch[key])
            elif isinstance(bunch[key], list):
                bunch[key] = [to_bunch(d) for d in bunch[key]]
    else:
        bunch = dictionary

    return bunch

def unroll(obj, name, value):
    props = name.split(".")
    if len(props) == 1:
        obj[name] = value
    else:
        if not props[0] in obj:
            obj[props[0]] = {}
        unroll(obj[props[0]], ".".join(props[1:]), value)
    return obj