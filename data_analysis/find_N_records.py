import argparse
import os
import onnxruntime as rt
import numpy as np
from utils.helpers import get_class
from utils.config import Config 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Config name (wihout extension) or full path", type=str)

    args = parser.parse_args()
    if args.config:
        config = Config(args.config)

        file_reader = get_class(config.settings.dataset.file_provider)()
        
        folders = ["D:\\Study\\Aibolit-ECG\data\\examples\\aibolit\\2fold_3.3s_(CLV)_250hz\\0"]
        examples = file_reader.list(folders)

        fpath = os.path.join(config.model_dir, "model.onnx")
        sess = rt.InferenceSession(fpath)
        input_name = sess.get_inputs()[0].name

        n_index = config.settings.dataset.params.label_map["N"] 
        N_records = []

        for fname, metadata in examples:
            x = np.asarray(file_reader.read(fname), dtype=np.float32)
            x = np.expand_dims(x, axis=0)

            if config.settings.dataset.params.normalize_input:
                pass
                #TO DO: normalize
            
            predictions = sess.run(None, {input_name: x})
            predictions = predictions[0]
            predictions = np.argmax(predictions, axis=1)

            if predictions[0] == n_index:
                N_records.append(metadata.source_id)

        print("N records: ", len(set(N_records)))
        with open("N.txt", "w") as f:
            for record in sorted(list(set(N_records))):
                f.write("{}\n".format(record))
        
    else:
        print("configuration file name is required. use -h for help")

if __name__ == "__main__":
    main()