import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

EPSILON = 0.0001

class ConfusionMatrix(object):
    def __init__(self, predictions, ground_truth, class_num):
        self._class_num = class_num
        self._cm = np.zeros((class_num, class_num))
        self.append(predictions, ground_truth)

    def add(self, cm):
        if isinstance(cm, ConfusionMatrix):
            self._cm = self._cm + cm.numpy()
        else:
            raise ValueError("cm must be an instance of ConfusionMatrix class")
            
    def append(self, predictions, ground_truth):
        for predicted, ground_truth in zip(predictions, ground_truth):
            self._cm[predicted][ground_truth] += 1

    def numpy(self):
        return self._cm

    def accuracy(self):
        tp = self._cm.diagonal().sum()
        total = self._cm.sum()
        return tp / (total + EPSILON)

    def class_accuracy(self):
        class_total = np.sum(self._cm, axis=0)
        class_tp = self._cm.diagonal()
        return class_tp / (class_total + EPSILON)

    def plot(self, fpath, class_map=None):
        class_map = class_map or {}
        
        labels = [class_map[i] or i for i in range(self._class_num)]
        df_cm = pd.DataFrame(self._cm, index=labels, columns=labels)

        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')

        plt.ylabel("Predicted")
        plt.xlabel("Actual")

        plt.savefig(fpath)