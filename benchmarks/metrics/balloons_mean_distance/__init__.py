from metrics.base import MetricBase
from scipy.optimize import linear_sum_assignment
import numpy as np

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output, target):
        points_input = []
        for balloon in output["balloons"]["balloons"]:
            points_input.append(balloon['center'])

        points_target = []
        for balloon in target.ground_truth().balloons.balloons:
            points_target.append(balloon.center)

        points_input = np.array(points_input)
        points_target = np.array(points_target)

        if len(points_input) == 0 or len(points_target) == 0:
            return
            
        distances = np.linalg.norm(points_input[:, np.newaxis, :] - points_target[np.newaxis, :, :], axis=-1)

        row_ind, col_ind = linear_sum_assignment(distances)
        metric = distances[row_ind, col_ind].mean()

        self.values.append(metric)
        self.weights.append(row_ind.shape[0])

    def value(self):
        return (np.array(self.values) / np.array(self.weights).sum(keepdims=True) * np.array(self.weights)).sum()
    
    def clear(self):
        self.values = []
        self.weights = []