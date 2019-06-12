import os
from utils.dirs import clear_dir, create_dirs
from datasets.MIT.utils.name_generator import NameGenerator
from datasets.MIT.utils.data_structures import Example

class BaseFileProvider(object):
    def __init__(self, extension):
        self.AUGMENTED_DIR = "augmented"
        self.TEST_DIR = "testset"
        
        self.FILE_EXTENSION = extension
        self.name_generator = NameGenerator(extension)

    def load(self, directory, include_augmented=False):
        """Loads examples from disk
        Args:
            directory: target directory
            include_augmented: if True, return augmented examples as secod element of returned list
        Returns:
            ([regular_examples], <[augmented_examples]>), elemets are Example namedpuples
        """
        examples = self._load_dir(directory)
        examples_aug = []
        if include_augmented:
            examples_aug = self._load_dir(os.path.join(directory, self.AUGMENTED_DIR))

        return (examples, examples_aug)

    def save(self, slices, directory, params=None):
        """Converts slices to proper file format and saves them to disc
        Args:
            slices: 2d list of slices,
            elements are namedtuples, (Index, rythm, start, end, signal), e.g:
            [[(rythm="(N", start=10, end=760, signal=[0.222, 0.225, ...]), (...)], ...]
            directory: directory to save files
            params: dictionalry, passed to _build_save_file_fn method
        Returns:
            None
        """
        if os.path.exists(directory):
            clear_dir(directory)
        else:
            create_dirs([directory])

        save_file_fn, dispose_fn = self._build_save_file_fn(directory, params)

        for s in slices:
            fname = self.name_generator.generate_name(
                index=s.Index,
                rythm=s.rythm,
                record=s.record,
                start=s.start,
                end=s.end
            )
            
            save_file_fn(s.signal, fname)
        
        dispose_fn()

    def _load_dir(self, directory):
        """Loads examples from directory
        Returns:
            list of Example namedtuples (data, label, name),
            label denotes rythm type, eg "(N"
            name denotes file name
        """
        if not os.path.exists(directory):
            return []

        fnames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        data = [None] * len(fnames)
        labels = [None] * len(fnames)

        for i, fname in enumerate(fnames):
            try:
                fpath = os.path.join(directory, fname)
                x = self._read_file(fpath)
                metadata = self.name_generator.get_metadata(fname)
                if metadata:
                    labels[i] = metadata.rythm
                    data[i] = x
                else:
                    print("Skipped file {}, can't parse name".format(fpath))
            except Exception as e:
                print("Skipped file {}, see error details:".format(fpath))
                if hasattr(e, 'message'):
                    print(e.message)
                else:
                    print(e)

        filtered = [Example(x, lbl, f) for x, lbl, f in zip(data, labels, fnames) if not (x is None) and bool(lbl)]

        return filtered

    def _read_file(self, fpath):
        """Read and return file data
        Implement in an inherited class
        """
        raise NotImplementedError()

    def _build_save_file_fn(self, directory, params):
        """Returns tuple (save_file_fn, dispose_fn) function
        save_file_fn(signal, file_name)
        """
        raise NotImplementedError()   