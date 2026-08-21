from pathlib import Path
from tqdm import tqdm
import importlib
import argparse
import yaml
import json
import os

from datasets.base import DatasetClassBase
from models.base import ModelClassBase

from compute_metrics import main as compute_metrics

if __name__ == "__main__":
    # parsing arguments and yaml file
    parser = argparse.ArgumentParser(description="A simple benchmarking framework")
    parser.add_argument("--config", default="./default.yml", help="Config file to use")
    parser.add_argument("--test", action="store_true", help="Set True if you want to run every loop without inferencing models")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        data = yaml.safe_load(f)

    name = data['name'] if 'name' in data else "benchmark"

    if "samples_cap" not in data:
        data["samples_cap"] = -1

    # printing settings
    print(f"Running benchmark: '{name}'\n")

    print(f"Used datasets: ")
    print("".join([f"- {dataset}\n" for dataset in data["datasets"]]))

    print(f"Used models: ")
    print("".join([f"- {model}\n" for model in data["models"]]))

    print(f"Used metrics: ")
    print("".join([f"- {metric}\n" for metric in data["metrics"]]))

    # searching for available place to save benchmark results without overriding the previous ones
    dirs = os.listdir("results")
    dirname = name
    if name in dirs:
        print(f"The directory results/{name}/ is already taken. Searching for next available option...")
        i : int = 2
        while True:
            if f"{name}_v{i}" not in dirs:
                dirname = f"{name}_v{i}"
                break
            i += 1
    
    print(f"The results will be saved in results/{dirname}/")
    os.mkdir(f"results/{dirname}") # path injection possible
    os.mkdir(f"results/{dirname}/outputs") # path injection possible

    print("Loading datasets...")
    datasets : list[DatasetClassBase] = [getattr(importlib.import_module(f"datasets.{dataset}"), "DatasetClass")() for dataset in tqdm(data["datasets"])] # path injection possible
    models : list[ModelClassBase] = [getattr(importlib.import_module(f"models.{dataset}"), "ModelClass")() for dataset in tqdm(data["models"])] # path injection possible
    print("Inferencing models...")
    for i, dataset in enumerate(datasets):
        print(f"========== { data["datasets"][i] } ==========")
        os.mkdir(f"results/{dirname}/outputs/{data["datasets"][i]}")
        for j, model in enumerate(models):
            print(f"Testing {data["models"][j]}...")
            os.mkdir(f"results/{dirname}/outputs/{data["datasets"][i]}/{data["models"][j]}")
            # it would be nice if requests could be sent in batches
            for k, sample in tqdm(enumerate(dataset)):
                if k == data["samples_cap"]:
                    break
                sample_id = str(sample.id).rsplit("/", 1)[-1] # placeholder; it should be resolved later, so you can just use sample.id 
                if not args.test:
                    try:
                        extracted = model.forward(sample.image())
                        Path(f"results/{dirname}/outputs/{data["datasets"][i]}/{data["models"][j]}/{sample_id}.json").write_text(json.dumps(extracted, indent=2))
                    except Exception as exc:
                        print(f"Unable to process sample {sample_id}: Skipped")

    print("Generating outputs done.")

    print("Computing metrics...")

    compute_metrics(data, dirname, datasets, args.test)

    print("Done")