from metrics.base import MetricBase
from scipy.optimize import linear_sum_assignment
import numpy as np

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
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

        row_ind, col_ind = linear_sum_assignment(cost)

        metric = cost[row_ind, col_ind].mean()

        self.values.append(metric)
        self.weights.append(row_ind.shape[0])
        
    def value(self):
        return (np.array(self.values) / np.array(self.weights).sum(keepdims=True) * np.array(self.weights)).sum()
    
    def clear(self):
        self.values = []
        self.weights = []