from metrics.base import MetricBase
from scipy.optimize import linear_sum_assignment
import numpy as np
import json
from nltk.metrics.distance import edit_distance

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output, target):
        _target = json.loads(target.ground_truth().model_dump_json())["features"]

        types = ["bores", "chamfers", "dimensions", "gdnts", "radii", "roughnesses", "threads"]

        metrics = []
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
                    cost[i][j] = edit_distance(labels_output[i], labels_target[j]) / max(len(labels_output[i]), len(labels_target[j]), 1)

            row_ind, col_ind = linear_sum_assignment(cost)

            if len(row_ind) != 0:
                metric = (cost[row_ind, col_ind]).mean()
                metrics.append(metric)

        metrics = np.array(metrics)
    
        self.values.append(metrics.mean())
        self.weights.append(len(metrics))

    def value(self):
        return 1-(np.array(self.values) / np.array(self.weights).sum(keepdims=True) * np.array(self.weights)).sum()
    
    def clear(self):
        self.values = []
        self.weights = []