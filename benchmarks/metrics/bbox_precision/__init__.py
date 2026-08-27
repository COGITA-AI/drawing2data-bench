from metrics.bbox_f1 import Metric as MetricBase
from datasets.base import FeatureList, SampleClassBase

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output : FeatureList, target : SampleClassBase):
        target = target.ground_truth()
        true_positive, false_positive, false_negative = self.statistics(output, target)

        self.tp += true_positive
        self.fp += false_positive

    def value(self):
        precision = self.tp / max(self.tp + self.fp, 1)

        return precision
    
    def clear(self):
        self.tp = 0
        self.fp = 0