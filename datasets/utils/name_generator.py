import re
from datasets.utils.example_metadata import ExampleMetadata

class NameGenerator(object):
    def __init__(self, file_extension):
        self.file_extension = file_extension

    def generate_name(self, label, source_id, start, end):
        template = "{}_{}_{}-{}{}"
        return template.format(label, source_id, start, end, self.file_extension)

    def generate_aug_name(self, original, aug_name):
        return "{}_{}{}".format(
            original.rstrip(self.file_extension),
            aug_name,
            self.file_extension
        )

    def get_metadata(self, fname):
        regex = "^(?P<label>[(\w]+)_(?P<source_id>[\.\d]+)_(?P<start>\d+)-(?P<end>\d+)"
        m = re.match(regex, fname)
        if m:
            return ExampleMetadata(
                source_id=m.group("source_id"),
                label=m.group("label"),
                start=m.group('start'),
                end=m.group('end')
            )
        else:
            return None