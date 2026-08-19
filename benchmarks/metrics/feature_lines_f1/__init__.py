from metrics.base import MetricBase
from scipy.optimize import linear_sum_assignment
import numpy as np

class Metric(MetricBase):
    def __init__(self, threshold=100):
        super().__init__()
        self.threshold = threshold
    
    def aggregate(self, output, target):
        lines_output = []
        polygons_output = 0
        for reference_position in output["reference_positions"]["reference_positions"]:
            polygon = np.array(reference_position["polygon"]["coordinates"])
            polygons_output += 1
            if polygon.shape[0] == 2:
                lines_output.append(polygon)

        lines_output = np.array(lines_output)

        lines_target = []
        polygons_target = 0
        for reference_position in target.ground_truth().reference_positions.reference_positions:
            polygon = np.array(reference_position.polygon.coordinates)
            polygons_target += 1
            if polygon.shape[0] == 2:
                lines_target.append(polygon)

        lines_target = np.array(lines_target)

        if len(lines_output) and len(lines_target):
            cost = np.linalg.norm(lines_output[:, np.newaxis, :, :] - lines_target[np.newaxis, :, :, :], axis=-1).sum(axis=-1)
        else:
            cost = np.array([[]])
        
        maxi = max(cost.max(), cost.max())
        cost[cost > self.threshold] = maxi * 10

        row_ind, col_ind = linear_sum_assignment(cost)

        true_positive = (cost[row_ind, col_ind] <= self.threshold).sum()
        false_negative = lines_target.shape[0] - true_positive
        false_positive = lines_output.shape[0] - true_positive

        self.tp += true_positive
        self.fp += false_positive
        self.fn += false_negative

    def value(self):
        precision = self.tp / max(self.tp + self.fp, 1)

        recall = self.tp / max(self.tp + self.fn, 1)

        f1 = 2 * precision * recall / (precision + recall + self.eps)
        
        if self.tp == 0 and self.fp == 0 and self.fn == 0:
            f1 = 1

        return f1
    
    def clear(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0