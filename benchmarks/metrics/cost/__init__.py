from metrics.base import MetricBase
from datasets.base import FeatureList, SampleClassBase
from scipy.optimize import linear_sum_assignment
import numpy as np

class Metric(MetricBase):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, output, target : SampleClassBase):
        target = target.ground_truth()
        output = FeatureList.model_validate(output)
        self.values.append(output.cost)

    def value(self):
        if len(self.values) == 0:
            return 0
        return np.array(self.values).mean()
    
    def clear(self):
        self.values = []