from metrics.base import MetricBase
from datasets.base import FeatureList, SampleClassBase
from scipy.optimize import linear_sum_assignment
import numpy as np
import json
from nltk.metrics.distance import edit_distance

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output : FeatureList, target : SampleClassBase):
        target = target.ground_truth()

        bbox_output = []
        for feature in output.feature_list:
            bbox_output.append(feature.bbox)

        bbox_target = []
        for feature in target.feature_list:
            bbox_target.append(feature.bbox)

        bbox_output = np.array(bbox_output)
        bbox_target = np.array(bbox_target)

        ious = np.zeros((len(bbox_output, bbox_target)))

        for i, box1 in enumerate(bbox_output):
            for j, box2 in enumerate(bbox_output):
                dx = max(min(max(box1[0], box1[2]), max(box2[0], box2[2])) - max(min(box1[0], box1[2]), min(box2[0], box2[2])), 0)
                dy = max(min(max(box1[1], box1[3]), max(box2[1], box2[3])) - max(min(box1[1], box1[3]), min(box2[1], box2[3])), 0)
                intersection = dx * dy
                union = (max(box1[0], box1[2]) - min(box1[0], box1[2])) * (max(box1[1], box1[3]) - min(box1[1], box1[3])) + (max(box2[0], box2[2]) - min(box2[0], box2[2])) * (max(box2[1], box2[3]) - min(box2[1], box2[3])) - intersection
                ious[i][j] = intersection / (union + self.union)

        row_ind, col_ind = linear_sum_assignment(ious)

        metrics = []     

        for i, j in zip(row_ind, col_ind):
            output_idx = output["balloons"]["balloons"][i]["reference_id"]
            target_idx = target.ground_truth().balloons.balloons[j].reference_id

            metrics.append(edit_distance(output[output_idx].text, target[target_idx].text) / max(len(output[output_idx].text), len(target[target_idx].text), 1))

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