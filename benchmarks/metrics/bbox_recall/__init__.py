from metrics.bbox_f1 import Metric as MetricBase
from datasets.base import FeatureList, SampleClassBase

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output : FeatureList, target : SampleClassBase):
        target = target.ground_truth()
        true_positive, false_positive, false_negative = self.statistics(output, target)

        self.tp += true_positive
        self.fn += false_negative

    def value(self):
        recall = self.tp / max(self.tp + self.fn, 1)

        return recall
    
    def clear(self):
        self.tp = 0
        self.fn = 0