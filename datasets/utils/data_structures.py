from collections import namedtuple

Record = namedtuple("Record", ["signal", "annotation"])

class ExampleMetadata(object):
    def __init__(self, label, source_id, start, end):
        self.label = label
        self.source_id = source_id
        self.start = start
        self.end = end

class Example(object):
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata