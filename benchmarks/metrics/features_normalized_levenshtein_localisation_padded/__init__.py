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

        labels_output = {}
        labels_target = {}

        for t in types:
            inp = output["features"][t]
            tar = _target[t]

            for feature in inp:
                labels_output[feature["reference_id"]] = feature["label"]
            
            for feature in tar:
                labels_target[feature["reference_id"]] = feature["label"]

        # points_input = [None for _ in range(len(output["balloons"]["balloons"]))]
        points_input = []
        for balloon in output["balloons"]["balloons"]:
            points_input.append(balloon['center'])

        # points_target = [None for _ in range(len(target.ground_truth().balloons.balloons))]
        points_target = []
        for balloon in target.ground_truth().balloons.balloons:
            # points_target[balloon.reference_id] = balloon.center
            points_target.append(balloon.center)

        points_input = np.array(points_input)
        points_target = np.array(points_target)

        if len(points_input) and len(points_target):
            distances = np.linalg.norm(points_input[:, np.newaxis, :] - points_target[np.newaxis, :, :], axis=-1)
        else:
            distances = np.array([[]])

        row_ind, col_ind = linear_sum_assignment(distances)

        metrics = []
        for i in range(len(points_input)):
            if i not in row_ind:
                metrics.append(0)

        for i in range(len(points_target)):
            if i not in col_ind:
                metrics.append(0)
                       

        for i, j in zip(row_ind, col_ind):
            output_idx = output["balloons"]["balloons"][i]["reference_id"]
            target_idx = target.ground_truth().balloons.balloons[j].reference_id

            if output_idx not in labels_output:
                print("Dziwne output!")
                continue
            if target_idx not in labels_target:
                print("Dziwne target!")
                continue

            metrics.append(edit_distance(labels_output[output_idx], labels_target[target_idx]) / max(len(labels_output[output_idx]), len(labels_target[target_idx]), 1))

        if len(metrics) == 0:
            return 0, 1
        
        metrics = np.array(metrics)
    
        self.values.append(metrics.mean())
        self.weights.append(len(metrics))

    def value(self):
        return 1-(np.array(self.values) / np.array(self.weights).sum(keepdims=True) * np.array(self.weights)).sum()
    
    def clear(self):
        self.values = []
        self.weights = []