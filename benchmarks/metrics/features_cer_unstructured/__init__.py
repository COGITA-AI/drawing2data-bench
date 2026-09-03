from metrics.base import MetricBase
from datasets.base import FeatureList, SampleClassBase
from scipy.optimize import linear_sum_assignment
import numpy as np
import json
from nltk.metrics.distance import edit_distance

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output, target : SampleClassBase):
        target = target.ground_truth()
        output = FeatureList.model_validate(output)

        cost = np.zeros((len(output), len(target)))

        for i in range(len(output)):
            for j in range(len(target)):
                output_text = output[i].text
                target_text = target[j].text
                cost[i][j] = edit_distance(output_text, target_text) / max(len(output_text), len(target_text), 1)

        row_ind, col_ind = linear_sum_assignment(cost)

        metrics = []
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