from metrics.base import MetricBase
from datasets.base import FeatureList, SampleClassBase
from scipy.optimize import linear_sum_assignment
import numpy as np

class Metric(MetricBase):
    def __init__(self, threshold=50):
        super().__init__()
        self.threshold = threshold

    def statistics(self, output : FeatureList, target : FeatureList):
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


        same = ious > self.threshold
        true_positive = 0
        if not (len(bbox_output) == 0 or len(bbox_target) == 0):
            row_ind, col_ind = linear_sum_assignment(same)
            true_positive = (same[row_ind, col_ind]).sum()

        false_negative = bbox_target.shape[0] - true_positive
        false_positive = bbox_output.shape[0] - true_positive
        return true_positive, false_positive, false_negative
    
    def aggregate(self, output : FeatureList, target : SampleClassBase):
        target = target.ground_truth()
        
        true_positive, false_positive, false_negative = self.statistics(output, target)

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