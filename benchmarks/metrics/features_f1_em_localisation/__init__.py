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

        labels_output = {}
        labels_target = {}

        for t in types:
            inp = output["features"][t]
            tar = _target[t]

            for feature in inp:
                labels_output[feature["reference_id"]] = feature["label"]
            
            for feature in tar:
                labels_target[feature["reference_id"]] = feature["label"]

        points_input = []
        for balloon in output["balloons"]["balloons"]:
            points_input.append(balloon['center'])

        points_target = []
        for balloon in target.ground_truth().balloons.balloons:
            points_target.append(balloon.center)

        points_input = np.array(points_input)
        points_target = np.array(points_target)

        if len(points_input) and len(points_target):
            distances = np.linalg.norm(points_input[:, np.newaxis, :] - points_target[np.newaxis, :, :], axis=-1)
        else:
            distances = np.array([[]])

        row_ind, col_ind = linear_sum_assignment(distances)

        true_positive = 0
        for i, j in zip(row_ind, col_ind):
            if i not in labels_output:
                continue
            if j not in labels_target:
                continue
            if labels_output[output["balloons"]["balloons"][i]["reference_id"]] == labels_target[target.ground_truth().balloons.balloons[j].reference_id]:
                true_positive += 1

        false_positive = len(labels_output) - true_positive
        false_negative = len(labels_target) - true_positive

        self.tp += true_positive
        self.fp += false_positive
        self.fn += false_negative

    def value(self):
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)

        metric = 2 * precision * recall / (precision + recall + self.eps)

        if self.tp == 0 and self.fp == 0 and self.fn == 0:
            metric = 1

        return metric
    
    def clear(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0