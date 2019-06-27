import os
from utils.dirs import clear_dir, create_dirs
from datasets.utils.name_generator import NameGenerator

class BaseFileProvider(object):
    def __init__(self, extension):
        self.FILE_EXTENSION = extension
        self.name_generator = NameGenerator(extension)

    def list(self, directories):
        """Loads examples
        Returns:
            list of file paths along with metadata
        """
        def is_metadata_valid(name, metadata):
            if not metadata:
                print("Skipped file {}, can't parse name".format(name))
            return bool(metadata)

        result = []

        for directory in directories:
            fnames = (os.path.join(directory, f) for f in os.listdir(directory))
            fnames = (f for f in fnames if os.path.isfile(f))
            files = ((f, self.name_generator.get_metadata(os.path.basename(f))) for f in fnames)
            files = [f for f in files if is_metadata_valid(*f)]
            
            result.extend(files)

        return result

    def save(self, examples, directory, params=None):
        """Converts slices to proper file format and saves them to disc
        Args:
            slices: 2d list of slices,
            elements are namedtuples, (Index, rhythm, start, end, signal), e.g:
            [[(rhythm="(N", start=10, end=760, signal=[0.222, 0.225, ...]), (...)], ...]
            directory: directory to save files
            params: dictionalry, passed to _build_save_file_fn method
        Returns:
            None
        """
        if not os.path.exists(directory):
            create_dirs([directory])

        save_file_fn, dispose_fn = self._build_save_file_fn(directory, params)

        for e in examples:
            fname = self.name_generator.generate_name(
                label=e.metadata.label,
                source_id=e.metadata.source_id,
                start=e.metadata.start,
                end=e.metadata.end
            )
            
            save_file_fn(e.data, fname)
        
        dispose_fn()

    def read(self, fpath):
        try:
            x = self._read_file(fpath)
            return x
        except Exception as e:
            print("Skipped file {}, see error details:".format(fpath))
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
            return []

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