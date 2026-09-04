# Drawing2Data-Bench

Drawing2Data-Bench is a set of tools for synthetic technical-drawing
generation, annotation-region extraction, and model benchmarking.

The repository is organized as a pipeline:

```text
3D CAD models -> generation -> PNG + COCO dataset -> extraction -> detector
                                                       -> benchmarks -> metrics
```

## Setup

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The repository requirements include RF-DETR training dependencies, OpenRouter
model clients, Pydantic, COCO tools, and SciPy. The tested operating system is
Linux (Ubuntu 26.04 LTS).

## Generation

The [generation README](generation/README.md) documents AutoDraft, which
loads STEP, IGES, and BREP solids, recognizes geometry and PMI, selects views,
renders dimensioned drawings, and writes PNG images with COCO detection
annotations. It also documents house styles, annotation classes, model scale,
sections, detail views, variation, sub-part views, the CLI, dataset checking,
tests, and known limits.

Start a dataset build from the repository root with:

```bash
python -m generation.cli "generation/examples/models/*.step" -o dataset
```

## Extraction

The [extraction README](extraction/README.md) documents the RF-DETR Large
training and inference scripts. Training consumes the generated Roboflow-style
COCO dataset; inference writes an image with detected annotation boxes.
Training artifacts currently available under `output/`, including metrics and
TensorBoard logs, are described there with their recorded results.

```bash
python -m extraction.train
python -m extraction.infer image.png -w output/checkpoint_best_total.pth
```

## Benchmarks

The [benchmarks README](benchmarks/README.md) documents the configurable
benchmark runner, dataset/model interfaces, model outputs, metric outputs,
all registered metric definitions, tests, and the benchmark result layout.
Benchmark scripts are run from `benchmarks/` because they use local imports
and relative `results/` paths.

```bash
cd benchmarks
python benchmark.py --config default.yml
```

The checked-in `benchmark_15` run and its interpretation are documented in
[benchmarks/README.md](benchmarks/README.md). Its localization results show
that the task has strong potential, while its text metrics also make the
current limitation visible: the local detector emits empty text, so its
semantic scores are zero by design.

## Repository data

- `models/` contains repository-level STEP models.
- `generation/examples/models/` contains generation examples, including the
  assembly and AP242 PMI examples described in the generation documentation.
- `dataset/` contains generated train, validation, and test data when present.
- `output/` contains extraction checkpoints, exports, metrics, and logs when
  present.
- `benchmarks/results/` contains benchmark model outputs and metric JSON files
  when benchmark runs have been performed.

Artifacts too large or unsuitable for inclusion in the repository are
available on the shared drive:

<https://drive.google.com/drive/folders/1rHwoiXBxIVY4zxsPVJP3d27EiWnvynUY>

## Known limitations

The detailed limitations are kept with the components that own them. Notable
constraints include synthetic surface-finish values, incomplete support for
some drawing conventions and assembly annotations, weak transfer of the
extractor to real printed drawings, and an extractor that detects annotation
classes but does not read their printed text.
