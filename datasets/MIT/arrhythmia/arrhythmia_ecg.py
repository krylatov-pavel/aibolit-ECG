import numpy as np
from datasets.utils.ecg import ECG

class ArrhythmiaECG(ECG):
    def __init__(self, name, signal, labels, beats, timecodes):
        super(ArrhythmiaECG, self).__init__(name, signal, labels, timecodes)
        self.beats = beats

    def get_slices(self, slice_window, rhythm_filter, overlap):
        slices = []

        curr_rhythm = None
        curr_sequence = {
            "name": None,
            "start": None,
            "end": None
        }

        for rhythm, beat, start, prev in zip(self.labels, self.beats, self.timecodes, np.append(0, self.timecodes[:-1])):
            curr_rhythm = rhythm if rhythm else curr_rhythm
            match_name = self._match(rhythm_filter, curr_rhythm, beat)
            
            #end of a previous rhythm sequence
            if curr_sequence["name"] and curr_sequence["name"] != match_name:
                curr_sequence["end"] = prev + ((start - prev) // 2)
                new_slices = self.__cut_sequence(curr_sequence, slice_window, rhythm_filter, overlap)
                slices.extend(new_slices)

                curr_sequence["name"] = None
                curr_sequence["start"] = None

            #start of a new rhythm sequence
            if match_name and match_name != curr_sequence["name"]:
                curr_sequence["name"] = match_name
                curr_sequence["start"] = prev + ((start - prev) // 2)
        
        #if current sequence ends with the ECG
        if curr_sequence["name"]:
            curr_sequence["end"] = len(self.signal)
            new_slices = self.__cut_sequence(curr_sequence, slice_window, rhythm_filter, overlap)
            slices.extend(new_slices)
        
        return slices

    def __cut_sequence(self, sequence, slice_window, rhythm_filter, overlap):
        new_slices = self._cut_slices(slice_window, sequence["name"], sequence["start"], sequence["end"], overlap=overlap)
        
        curr_sequence_filter = [f for f in rhythm_filter if f.name == sequence["name"]][0]
        if curr_sequence_filter.use_augmentation and len(new_slices) > 0:
            new_slices.append(self._cut_slices(slice_window, sequence["name"], sequence["start"], sequence["end"], reverse=True)[0])
        
        return new_slices


    def _match(self, rhythm_filter, rhythm, beat):
        for f in rhythm_filter:
            if f.rhythm:
                if f.rhythm != rhythm:
                    continue
            if hasattr(f, "not_rhythm"):
                if rhythm in f.not_rhythm:
                    continue
            if len(f.beats) > 0:
                if not beat in f.beats:
                    continue
            return f.name
        return None