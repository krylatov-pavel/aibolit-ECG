import os
from datasets.utils.name_generator import NameGenerator

def get_records(path):
    fnames = (os.path.join(path, f) for f in os.listdir(path))
    fnames = (os.path.basename(f) for f in fnames if os.path.isfile(f))

    names = NameGenerator(".csv")
    meta = (names.get_metadata(f) for f in fnames)
    records = [r.source_id + r.label for r in meta]
    return records

def find_set_overlap(base_path):
    set_dirs = (os.path.join(base_path, d) for d in os.listdir(base_path)) 
    set_dirs = [d for d in set_dirs if os.path.isdir(d)]

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