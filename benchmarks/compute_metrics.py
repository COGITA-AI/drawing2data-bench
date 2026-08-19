from pathlib import Path
from tqdm import tqdm
import importlib
import argparse
import yaml
import json
import os
import numpy as np

from datasets.base import DatasetClassBase
from metrics.base import MetricBase

def main(data, dirname, datasets, test):
    metrics : list[MetricBase] = [getattr(importlib.import_module(f"metrics.{metric}"), "Metric")() for metric in tqdm(data["metrics"])]
    os.mkdir(f"results/{dirname}/metrics/")
    for i, dataset in enumerate(datasets):
        print(f"========== { data["datasets"][i] } ==========")
        os.mkdir(f"results/{dirname}/metrics/{data["datasets"][i]}")
        for model in data["models"]:
            print(f"Evaluating {model}...")
            
            metric_dict = {}
            # it would be nice if metrics could be computed in batches
            for j, metric in enumerate(metrics):
                for k, sample in tqdm(enumerate(dataset)):
                    if k == data["samples_cap"]:
                        break
                    if not test:
                        sample_id = str(sample.id).rsplit("/", 1)[-1] # placeholder; it should be resolved later, so you can just use sample.id 
                        output = json.loads(Path(f"results/{dirname}/outputs/{data["datasets"][i]}/{model}/{sample_id}.json").read_text())
                        metric.aggregate(output, sample)

                metric_dict[data["metrics"][j]] = metric.value()
                metric.clear()
            
            Path(f"results/{dirname}/metrics/{data["datasets"][i]}/{model}.json").write_text(json.dumps(metric_dict, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for computing metrics from model outputs")
    parser.add_argument("--name", required=True, help="Name of the benchmark")
    parser.add_argument("--config", default="./default.yml", help="Config file to use")
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = yaml.safe_load(f)

    if "samples_cap" not in data:
        data["samples_cap"] = -1

    datasets : list[DatasetClassBase] = [getattr(importlib.import_module(f"datasets.{dataset}"), "DatasetClass")() for dataset in tqdm(data["datasets"])]

    main(data, args.name, datasets, False)