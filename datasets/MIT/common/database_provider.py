import wfdb
import os
import pickle
from utils.dirs import create_dirs, is_empty, clear_dir
from datasets.utils.data_structures import Record

class DatabaseProvider(object):
    def __init__(self, db_name):
        self.db_name = db_name
        self._SIGNAL_FNAME = "signal.pkl"
        self._ANNT_FNAME = "annotation.pkl"

    @property
    def db_dir(self):
        return os.path.join("data", "database", self.db_name, "records")

    def get_records(self, bypass_cache=False):
        """Get a list of database ECG records

        Args:
            bypass_cache: if True force download record files from remote database,
            otherwise use local saved files

        Returns:
            list of Records, where each record is a named tuple. E.g:
            [(signal, annotation), ...]
        """

        db_dir = self.db_dir

        if not os.path.exists(db_dir):
            create_dirs([db_dir])

        if is_empty(db_dir) or bypass_cache:
            clear_dir(db_dir)
            self._save_records(db_dir)
        
        return self._load_records(db_dir)

    def get_record(self, record_name):
        record_dir = os.path.join(self.db_dir, record_name)
        return self.__load_record(record_dir)

    def _save_records(self, db_dir):
        record_names = wfdb.io.get_record_list(self.db_name, records="all")

        record_data = [self.__fetch_record(name) for name in record_names]
        
        for name, data in zip(record_names, record_data):
            if data != None:
                record_dir = os.path.join(db_dir, name)
                create_dirs([record_dir])

                #serialize signal object to /{database_name}/{record_name}/signal.pkl
                signal_fpath = os.path.join(record_dir, self._SIGNAL_FNAME)
                with open(signal_fpath, "wb") as f:
                    pickle.dump(data[0], f, pickle.DEFAULT_PROTOCOL)

                #serialize signal annotation object to /{database_name}/{record_name}/annotation.pkl
                ann_fpath = os.path.join(record_dir, self._ANNT_FNAME)
                with open(ann_fpath, "wb") as f:
                    pickle.dump(data[1], f, pickle.DEFAULT_PROTOCOL)

    def _load_records(self, db_dir):
        """ Loads list of records data from files in db_dir directory
        """
        
        record_dirs = [os.path.join(db_dir, r_dir) for r_dir in os.listdir(db_dir)]
        record_dirs = [r_dir for r_dir in record_dirs if os.path.isdir(r_dir)]
        records = [self.__load_record(r_dir) for r_dir in record_dirs]

        return records
    
    def __load_record(self, record_dir):
        signal_fpath = os.path.join(record_dir, self._SIGNAL_FNAME)
        with open(signal_fpath, "rb") as f:
            signal = pickle.load(f)

        annt_fpath = os.path.join(record_dir, self._ANNT_FNAME)
        with open(annt_fpath, "rb") as f:
            annotation = pickle.load(f)

        return Record(signal, annotation)

    def __fetch_record(self, record_name):
        try:
            rec_data = wfdb.rdrecord(record_name, pb_dir=self.db_name, channels=[0], physical=True)
            rec_annotation = wfdb.io.rdann(record_name, extension = "atr", pb_dir=self.db_name)
            
            print("downloaded record {} data".format(record_name))
            return rec_data, rec_annotation
        except:
            print("error downloading record {} data".format(record_name))
            return None        