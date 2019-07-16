import numpy as np
import random
import scipy.interpolate
from datasets.utils.data_structures import ExampleMetadata
from datasets.utils.data_structures import Example

class ECG(object):
    def __init__(self, source_type, name, labels, timecodes, fs, signal=None, signal_len=None):
        """
        Args:
            name: record name
            signal: 1d list of physical signal values, e.g [0.233, 0.217, ...]
            labels: 1d list of heart rhythm labels, e.g ["(N\x00", "(VT\x00", "(SIN\x00", ...]
            timecodes: 1d list of timecodes corresponding to heart rhythm labels, each time code
            denotes starting point of related heart rhythm sequence in frames, e.g [34, 2300, 7500, ...]     
        """
        self.source_type = source_type
        self.name = name
        self.signal = signal if signal is not None else []
        self.signal_len = signal_len or len(signal)
        self.labels = [l.rstrip("\x00") for l in labels] 
        self.timecodes = np.array(timecodes)
        self.fs = fs

    def get_examples_metadata(self, duration, rhythm_filter, rhythm_map=None):
        metadata = []
        rhythm_map = rhythm_map or {}

        for label, start, end in zip(self.labels, self.timecodes, np.append(self.timecodes[1:], self.signal_len)):
            if label in rhythm_map:
                label = rhythm_map[label]

            match = self.__match(label, rhythm_filter)
            if match:
                metadata.extend(self.__cut_examples_metadata(
                    example_duration=duration,
                    label=match.name,
                    start=start,
                    end=end,
                    overlap=match.overlap
                ))

        return metadata

    def get_examples(self, metadata, resample_fs=None):
        examples = [None] * len(metadata)

        resample_rate = resample_fs / self.fs if resample_fs else 1
        for i, meta in enumerate(metadata):
            signal = list(self.signal[meta.start:meta.end])
            
            if resample_rate != 1:
                resample_fn = scipy.interpolate.interp1d(np.arange(len(signal)), signal)
                signal_len = int(len(signal) * resample_rate)
                signal = resample_fn(np.linspace(0, len(signal) - 1, signal_len))

            examples[i] = Example(signal, meta)

        return examples
    
    def __cut_examples_metadata(self, example_duration, label, start, end, overlap=0):
        """ Cust single heart rhythm sequence into fixed-length slices
        Args:
            start: sequence start position, inclusive
            end: sequence end position, exclusive
        """
        slice_window = int(example_duration * self.fs)
        length = end - start

        slice_num = int(length >= slice_window) + np.maximum((length - slice_window) // (slice_window - int(slice_window * overlap)), 0)
        slices = [None] * slice_num

        for i in range(slice_num):
            if i == slice_num - 1:
                start_pos = self.signal_len - slice_window
                end_pos = self.signal_len
            else:
                start_pos = start + np.maximum(i * slice_window - i * int(slice_window * overlap), 0) 
                end_pos = start_pos + slice_window

            slices[i] = ExampleMetadata(
                label=label,
                source_type=self.source_type,
                source_id=self.name,
                start=start_pos,
                end=end_pos
            )

        return slices
        
    def __match(self, label, rhythm_filter):
        return rhythm_filter.get(label)