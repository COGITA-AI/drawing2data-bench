from metrics.base import MetricBase
from scipy.optimize import linear_sum_assignment
import numpy as np
import json

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output, target):
        _target = json.loads(target.ground_truth().model_dump_json())["features"]

        types = ["bores", "chamfers", "dimensions", "gdnts", "radii", "roughnesses", "threads"]

        tps = []
        fps = []
        fns = []
        for t in types:
            inp = output["features"][t]
            tar = _target[t]

            labels_output = []
            for feature in inp:
                labels_output.append(feature["label"])
            
            labels_target = []
            for feature in tar:
                labels_target.append(feature["label"])

            cost = np.zeros((len(labels_output), len(labels_target)))

            for i in range(len(labels_output)):
                for j in range(len(labels_target)):
                    cost[i][j] = 1 if labels_output[i] == labels_target[j] else 0

            row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

            true_positive = (cost[row_ind, col_ind]).sum()
            false_negative = len(labels_target) - true_positive
            false_positive = len(labels_output) - true_positive

            tps.append(true_positive)
            fps.append(false_positive)
            fns.append(false_negative)

        tps = np.array(tps)
        fps = np.array(fps)
        fns = np.array(fns)
        self.tps += tps
        self.fps += fps
        self.fns += fns

    def value(self):
        precision = self.tps / np.maximum(self.tps + self.fps, 1)
        recall = self.tps / np.maximum(self.tps + self.fns, 1)

        metrics = 2 * precision * recall / (precision + recall + self.eps)
        metrics[(self.tps + self.fps + self.fns) == 0] = 1

        return metrics.mean()
    
    def clear(self):
        self.tps = np.zeros((7, ))
        self.fps = np.zeros((7, ))
        self.fns = np.zeros((7, ))