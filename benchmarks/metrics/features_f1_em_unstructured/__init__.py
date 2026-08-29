from metrics.base import MetricBase
from datasets.base import FeatureList, SampleClassBase
from scipy.optimize import linear_sum_assignment
import numpy as np
import json

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output, target : SampleClassBase):
        target = target.ground_truth()
        output = FeatureList.model_validate(output)

        cost = np.zeros((len(output), len(target)))

        for i in range(len(output)):
            for j in range(len(target)):
                cost[i][j] = 1 if output[i].text == target[j].text and output[i].category == target[j].category else 0

        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

        true_positive = (cost[row_ind, col_ind]).sum()
        false_negative = len(target) - true_positive
        false_positive = len(output) - true_positive

        self.tps += true_positive
        self.fps += false_positive
        self.fns += false_negative

    def value(self):
        precision = self.tps / max(self.tps + self.fps, 1)
        recall = self.tps / max(self.tps + self.fns, 1)

        metric = 2 * precision * recall / (precision + recall + self.eps)
        if self.tps + self.fps + self.fns == 0:
            return 1

        return metric
    
    def clear(self):
        self.tps = 0
        self.fps = 0
        self.fns = 0