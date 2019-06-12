import numpy as np
import random
import scipy.interpolate
from datasets.utils.data_structures import Slice

class ECG(object):
    def __init__(self, name, signal, labels, timecodes):
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
    
    def get_slices(self, slice_window, rhythm_filter, rhythm_map, reverse=False, resample=False):
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

        slices = []
        
        for label, start, end in zip(self.labels, self.timecodes, np.append(self.timecodes[1:], len(self.signal))):
            if label in rhythm_map:
                label = rhythm_map[label]
            
            if label in rhythm_filter:
                if not resample:
                    slices.extend(self._cut_slices(slice_window, label, start, end, reverse))
                else:
                    slices.extend(self._cut_resampled_slices(slice_window, label, start, end))

        return slices

    def _cut_slices(self, slice_window, label, start, end, reverse=False, overlap=0):
        """ Cust single heart rhythm sequence into fixed-length slices
        Args:
            start: sequence start position, inclusive
            end: sequence end position, exclusive
            reverse: if True, start slicing from the end of a sequence
        """
        length = end - start
        slice_num = int(length >= slice_window) + np.maximum((length - slice_window) // (slice_window - int(slice_window * overlap)), 0)
        slices = [None] * slice_num

        for i in range(slice_num):
            if reverse:
                end_pos = end - np.maximum(i * slice_window - i * int(slice_window * overlap), 0) 
                start_pos = end_pos - slice_window
            else:
                start_pos = start + np.maximum(i * slice_window - i * int(slice_window * overlap), 0) 
                end_pos = start_pos + slice_window
            
            signal = list(self.signal[start_pos:end_pos])

            slices[i] = Slice(
                record=self.name,
                rhythm=label,
                start=start_pos,
                end=end_pos,
                signal=signal)
        
        return slices

    def _cut_resampled_slices(self, slice_window, label, start, end):
        """Randomly cut signal slices and downsample/interpolate it
        to the fixed slice_window length
        """

        resample_options = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
        slices = []

        start_pos = start
        end_pos = start

        while True:
            signal_len = int(random.choice(resample_options) * slice_window)

            start_pos = end_pos
            end_pos = start_pos + signal_len
            if end_pos > end:
                break
            
            signal = list(self.signal[start_pos:end_pos])
            resample_fn = scipy.interpolate.interp1d(np.arange(signal_len), signal)
            signal = resample_fn(np.linspace(0, signal_len - 1, slice_window))

            slices.append(Slice(
                record=self.name,
                rhythm=label,
                start=start_pos,
                end=end_pos,
                signal=signal
            ))

        return slices