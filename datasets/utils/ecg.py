import numpy as np
import random
import scipy.interpolate
from datasets.utils.data_structures import Slice

class ECG(object):
    def __init__(self, name, signal, labels, timecodes, fs):
        """
        Args:
            name: record name
            signal: 1d list of physical signal values, e.g [0.233, 0.217, ...]
            labels: 1d list of heart rhythm labels, e.g ["(N\x00", "(VT\x00", "(SIN\x00", ...]
            timecodes: 1d list of timecodes corresponding to heart rhythm labels, each time code
            denotes starting point of related heart rhythm sequence in frames, e.g [34, 2300, 7500, ...]     
        """
        self.name = name
        self.signal = signal
        self.labels = [l.rstrip("\x00") for l in labels] 
        self.timecodes = timecodes
        self.fs = fs
    
    def get_slices(self, slice_duration, rhythm_filter, rhythm_map=None, reverse=False, sample_rate=1.0):
        """Cuts heart rhythm sequences into a set of fixed-length slices
        Args:
            slice_window: int, slice length in frames
            rhythm_filter: list of heart rhythm types that needs to be included in slices,
            e.g ["(ASYS", "(VT", ...]
            rhythm_map: in case some labels have the same meaning, like "(N" and "(NSR" map them to
            the same label for convinience. Dictionary, e.g:
            {
                "(NSR)": "(N)",
                ...
            }
        Returns:
            list of Slice, each slice is a named tuple ("record", "rhythm", "start", "end", "signal"), e.g:
            [("(N", 32, 1001), ...]
        """

        rhythm_map = rhythm_map or {}
        slices = []
        
        for label, start, end in zip(self.labels, self.timecodes, self.timecodes[1:] + [len(self.signal)]):
            if label in rhythm_map:
                label = rhythm_map[label]
            
            match = self.__match(label, rhythm_filter)
            if match:
                slices.extend(self._cut_slices(slice_duration,
                    match.get("name"),
                    start,
                    end,
                    reverse=reverse,
                    take_last=match.get("take_last"),
                    overlap=match.get("overlap"),
                    sample_rate=sample_rate
                ))
        return slices

    def _cut_slices(self, slice_duration, label, start, end, reverse=False, overlap=0, take_last=False, sample_rate=1.0):
        """ Cust single heart rhythm sequence into fixed-length slices
        Args:
            start: sequence start position, inclusive
            end: sequence end position, exclusive
            reverse: if True, start slicing from the end of a sequence
        """
        slice_window = int(slice_duration * self.fs)
        signal = list(self.signal[start:end])

        if sample_rate != 1.0:
            slice_window = int(slice_duration * self.fs * sample_rate)
            resample_fn = scipy.interpolate.interp1d(np.arange(len(signal)), signal)
            signal_len = int(len(signal) * sample_rate)
            signal = resample_fn(np.linspace(0, len(self.signal) - 1, signal_len))

        length = len(signal)
        slice_num = int(length >= slice_window) + np.maximum((length - slice_window) // (slice_window - int(slice_window * overlap)), 0)
        slices = [None] * slice_num

        for i in range(slice_num):
            if reverse:
                end_pos = end - np.maximum(i * slice_window - i * int(slice_window * overlap), 0) 
                start_pos = end_pos - slice_window
            else:
                start_pos = start + np.maximum(i * slice_window - i * int(slice_window * overlap), 0) 
                end_pos = start_pos + slice_window
            
            slices[i] = Slice(
                record=self.name,
                rhythm=label,
                start=start_pos,
                end=end_pos,
                signal=list(signal[start_pos:end_pos])
            )

        if take_last:
            start_pos = 0 if reverse else len(signal) - slice_window
            end_pos = start_pos + slice_window
            slices.append(Slice(
                record=self.name,
                rhythm=label,
                start=start_pos,
                end=end_pos,
                signal=signal[start_pos:end_pos]
            ))
        
        return slices

    def __match(self, label, rythm_filter):
        f = [f for f in rythm_filter if f.rhythm == label]
        f = f[0] if len(f) > 0 else None
        return f