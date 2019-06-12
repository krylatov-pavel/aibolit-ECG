from collections import namedtuple

Record = namedtuple("Record", ["signal", "annotation"])

_slice_meta = ["record", "rythm", "start", "end"]
SliceMeta = namedtuple("SliceMeta", _slice_meta)
Slice = namedtuple("Slice", _slice_meta + ["signal"])

Scale = namedtuple("Scale", ["min", "max"])
CropMode = namedtuple("CropMode", ["vertical", "horizontal"])
Example = namedtuple("Example", ["x", "y", "name"])

class Crop:
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"