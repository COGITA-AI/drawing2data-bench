from metrics.base import MetricBase
from scipy.optimize import linear_sum_assignment
import numpy as np

class Metric(MetricBase):
    def __init__(self, threshold=50):
        super().__init__()
        self.threshold = threshold
    
    def aggregate(self, output, target):
        points_input = []
        for balloon in output["balloons"]["balloons"]:
            points_input.append(balloon['center'])

        points_target = []
        for balloon in target.ground_truth().balloons.balloons:
            points_target.append(balloon.center)

        points_input = np.array(points_input)
        points_target = np.array(points_target)

        distances = np.linalg.norm(points_input[:, np.newaxis, :] - points_target[np.newaxis, :, :], axis=-1)
        distances[distances > self.threshold] = distances.max() * 10

        row_ind, col_ind = linear_sum_assignment(distances)
        true_positive = (distances[row_ind, col_ind] <= self.threshold).sum()
        false_negative = points_target.shape[0] - true_positive

        self.tp += true_positive
        self.fn += false_negative

    def value(self):
        recall = self.tp / max(self.tp + self.fn, 1)

        return recall
    
    def clear(self):
        self.tp = 0
        self.fn = 0