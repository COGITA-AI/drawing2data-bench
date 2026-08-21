from metrics.base import MetricBase
import numpy as np
import json
import re

class Metric(MetricBase):
    def __init__(self):
        super().__init__()

    def parse_float_prefix(self, s):
        match = re.match(r'\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?', s)
        return float(match.group()) if match and match.group() else None
    
    def get_volume(self, insights):
        key = "enclosing_cuboid"
        dim_names = ["width", "height", "depth"]
        fact = 1

        if insights["dimensions_after_processing"][key] == None:
            key = "enclosing_cylinder"
            dim_names = ["depth", "diameter", "diameter"]
            fact = np.pi / 4

        dims = []
        for dim_name in dim_names:
            dim_obj = insights["dimensions_after_processing"][key][dim_name]
            unit_fact = 1
            if dim_obj["unit"] == "cm":
                unit_fact = 10
            elif dim_obj["unit"] == "dm":
                unit_fact = 100
            elif dim_obj["unit"] == "m":
                unit_fact = 1
            dims.append(self.parse_float_prefix(["value"]))
        
        volume = np.array(dims).prod() * fact
        return volume

    
    def aggregate(self, output, target):
        output_volume = self.get_volume(output["insights"])
        target_volume = self.get_volume(json.loads(target.ground_truth().model_dump_json())["insights"])

        self.values.append(abs(np.log10(output_volume / (target_volume + self.eps) + self.eps)))

    def value(self):
        if len(self.values) == 0:
            return 1
        return np.array(self.values).mean()

    def clear(self):
        self.values = []