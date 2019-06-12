import re
from datasets.MIT.utils.data_structures import SliceMeta

class NameGenerator(object):
    def __init__(self, file_extension):
        self.file_extension = file_extension

    def generate_name(self, index, rythm, record, start, end):
        template = "{}_{}_{}_{}-{}{}"
        return template.format(index, rythm, record, start, end, self.file_extension)

    def generate_aug_name(self, original, aug_name):
        return "{}_{}{}".format(
            original.rstrip(self.file_extension),
            aug_name,
            self.file_extension
        )

    def get_metadata(self, fname):
        regex = "^\d+_(?P<rythm>[(\w]+)_(?P<record>[\.\d]+)_(?P<start>\d+)-(?P<end>\d+)"
        m = re.match(regex, fname)
        if m:
            return SliceMeta(
                record=m.group("record"),
                rythm=m.group("rythm"),
                start=m.group('start'),
                end=m.group('end')
            )
        else:
            return None