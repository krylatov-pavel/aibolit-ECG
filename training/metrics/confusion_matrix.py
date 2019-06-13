import numpy as np

class ConfusionMatrix(object):
    def __init__(self, predictions, labels, class_num):
        self._cm = np.zeros((class_num, class_num))
        self.append(predictions, labels)

    def add(self, cm):
        if isinstance(cm, ConfusionMatrix):
            self._cm = self._cm + cm.numpy()
        else:
            raise ValueError("cm must be an instance of ConfusionMatrix class")
            
    def append(self, predictions, labels):
        for predicted, label in zip(predictions, labels):
            self._cm[predicted][label] += 1

    def numpy(self):
        return self._cm

    def accuracy(self):
        tp = self._cm.diagonal().sum()
        total = self._cm.sum()
        return tp / total

    def class_accuracy(self):
        class_total = np.sum(self._cm, axis=1)
        class_tp = self._cm.diagonal()
        return class_tp / class_total

    def plot(self):
        #TODO move plot function from MIT project
        pass