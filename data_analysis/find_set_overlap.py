import os
import h5py
from datasets.utils.name_generator import NameGenerator

def get_records(path):
    with h5py.File(path, "r") as f:
        records = [f[key].attrs["source_id"].decode("utf-8") + f[key].attrs["label"].decode("utf-8") for  key in f.keys()]
    return records

def find_set_overlap(base_path):
    set_dirs = (os.path.join(base_path, d) for d in os.listdir(base_path)) 
    set_dirs = [d for d in set_dirs if d.endswith(".hdf5")]

    has_overlap = False

    for i in range(len(set_dirs) - 1):
        for j in range(i + 1, len(set_dirs)):
            fold1 = get_records(set_dirs[i])
            fold2 = get_records(set_dirs[j])

            intersection = set(fold1) & set(fold2)

            if len(intersection) > 0:
                has_overlap = True
                
                fold_1_count = len([k for k in fold1 if k in intersection])
                fold_2_count = len([k for k in fold2 if k in intersection])

                print(", ".join(intersection))
                print("{}: {:.2f}".format(os.path.basename(set_dirs[i]), fold_1_count / len(fold1)))
                print("{}: {:.2f}\n".format(os.path.basename(set_dirs[j]), fold_2_count / len(fold2)))

    if not has_overlap:
        print("no overlap")