# Evaluation — benchmarks and metrics

Evaluates models on drawing and table datasets and compares them using a
modular library of metrics.

```
config.yml → datasets + models → benchmark.py → outputs/ → compute_metrics.py → metrics/
```

## Contents

1. Overview
2. Results
3. Key files
4. Datasets
5. Models
6. Metrics
7. Tests
8. Metric details
9. Related subsystems

## 1. Overview

This framework is used to evaluate different models and compare them using
the available metrics. It is designed in a modular way, with an abstract
interface, so that it is easy to extend and flexible to run benchmarks with.

To run a benchmark with the default options (more options are listed in
section 3.1), you need to be in the `evaluation/benchmarks/` folder and run:

```
python benchmark.py
```

## 2. Results

All results are stored in the `results` folder, where the subfolders are the
benchmark names. If a benchmark with the same name is run several times, the
results of later runs are saved under `<original name>_v<run number>`.

Inside a given benchmark's folder there is an `outputs` subfolder (model
outputs) and a `metrics` subfolder (computed metrics). Both of these folders
contain a subfolder per dataset, named after the dataset.

- In `outputs`, inside a given dataset's subfolder, there are further
  subfolders per model (folder name = model name), each containing JSON
  files with the outputs for every sample in the dataset (file name = sample
  id).
- In `metrics`, inside a given dataset's subfolder, there are JSON files with
  the metrics for each model, where the file name is the model name.

Below is an example results structure for a benchmark:

```
benchmark_v3/
├── metrics
│   ├── custom_test
│   │   ├── openai_gpt5_4_mini.json
│   │   └── openai_gpt5_5.json
│   └── generated
│       ├── openai_gpt5_4_mini.json
│       └── openai_gpt5_5.json
└── outputs
    ├── custom_test
    │   ├── openai_gpt5_4_mini
    │   │   ├── img14.json
    │   │   └── img4.json
    │   └── openai_gpt5_5
    │       ├── img14.json
    │       └── img4.json
    └── generated
        ├── openai_gpt5_4_mini
        │   ├── sample1.json
        │   └── sample2.json
        └── openai_gpt5_5
            ├── sample1.json
            └── sample2.json
```

## 3. Key files

### 3.1 benchmark.py

The main script, which can take any `.yml` file as configuration (more on
configuration files below). Its responsibilities are:

- loading datasets,
- loading models,
- running inference of the models on the datasets,
- saving the models' output,
- running the `compute_metrics.py` script (metric computation).

To run this script, you need to be in the same folder as this file and call:

```
python benchmark.py
```

There are also additional arguments available:

| argument | meaning |
|---|---|
| `--test` | Lets you go through the whole process without model inference (all computations that depend on model output are skipped). |
| `--config path/to/config.yml` | Lets you choose a configuration file other than the default `default.yml`. |

### 3.2 compute_metrics.py

A script for computing the requested metrics based on model outputs and
ground truth. It can be run independently of `benchmark.py`, but the model
outputs must already exist. To run this script, you need to be in the same
folder as this file and call:

```
python compute_metrics.py --name benchmark_name
```

Below is an explanation of all the arguments:

| argument | meaning |
|---|---|
| `--name <benchmark_name>` | Required – tells the script which benchmark to compute metrics for. If the benchmark was run multiple times under the same name, you need to provide the automatically generated new name (`<original name>_v<run number>`, as described in section 2), not the original name from the configuration. |
| `--config path/to/config.yml` | Lets you choose a configuration file other than the default `default.yml`. |

### 3.3 default.yml (and other configuration files)

`default.yml` is the default configuration file, chosen when no other file
is provided when running `benchmark.py` or `compute_metrics.py`. Below is the
list of options that can be set in a configuration file (a parameter is
required unless stated otherwise):

| option | meaning |
|---|---|
| `name` | The benchmark name. |
| `datasets` | The list of datasets the models should be benchmarked on. |
| `models` | The list of models to benchmark. |
| `metrics` | The list of metrics to compute. |
| `samples_cap` | *(optional)* A cap on the number of samples the model is tested on (counted separately for each dataset). When this parameter is not set, or is set to a negative number, there is no cap. |

All dataset/model/metric names must match exactly the name of the module
that contains the given element.

Below is an example configuration file structure:

```yaml
name: "benchmark"

samples_cap: 10

datasets:
- generated
- custom_test

models:
- openai_gpt5_4_mini
- openai_gpt5_6_Sol
- gemini_3_1_flash_lite
- gemini_3_5_flash

metrics:
- bbox_f1
- bbox_precision
- bbox_recall
- features_f1_em_unstructured
- features_f1_em_localisation
- features_cer_unstructured
- features_cer_localisation_padded
- features_cer_localisation_unpadded
- custom_DocILE_LIR_f1
- custom_DocILE_LIR_recall
- TEDS_S
- GriTS_Top
```

### 3.4 .env

A file with environment variables, e.g. the OpenRouter API key.

## 4. Datasets

This section describes the `datasets` module. It doesn't implement anything
by itself, but contains all the submodules for each dataset.

### 4.1 Dataset base class

The base of every dataset looks like this:

```python
class DatasetClassBase(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: int) -> SampleClassBase:
        ...
```

Below is an explanation of each function:

- `__len__` – returns the length of the dataset,
- `__getitem__` – returns the i-th element of the dataset.

### 4.2 Sample base class

The base of every sample class looks like this:

```python
class SampleClassBase(ABC):
    @abstractmethod
    def image(self):
        ...

    @abstractmethod
    def ground_truth(self) -> FeatureList:
        ...
```

Below is an explanation of each function:

- `image` – returns the image in base64 format for the given sample,
- `ground_truth` – returns the ground truth: a `FeatureList` for drawings
  (section 4.3) and a `Table` for tables (section 4.4).

### 4.3 Drawing ground truth: Feature, FeatureList and Category

The ground truth for drawings is held in a self-defined format (instead of
the werk24 models used previously). A drawing consists of a list of
features, where each feature is a labelled text region:

```python
class Feature(BaseModel):
    id: int
    bbox: tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    text: str = ""
    category: Category
    confidence: float
```

Below is an explanation of each field:

- `id` – the unique id of the feature,
- `bbox` – the bounding box containing the text (and only text) of the
  feature; the coordinates are non-negative integers (`Coordinate`),
- `text` – the whole text displayed by the feature,
- `category` – the category of the feature, one of the values of the
  `Category` enum below,
- `confidence` – the model's confidence that the feature exists (in [0, 1]).

The `Category` enum marks what a given feature is:

- `gdnt` – feature control frames,
- `roughness` – surface-texture symbols,
- `radius` – fillet / corner radius notes,
- `chamfer` – chamfer and countersink callouts,
- `bore` – hole callouts,
- `thread` – thread specifications,
- `dimension` – linear, aligned, angular, diameter,
- `note` – the paragraph block (e.g. `NOTES:`, `UNLESS OTHERWISE
  SPECIFIED`),
- `datum` – boxed datum letters with their triangle,
- `leader_note` – a short string on a leader (e.g. `THICKNESS 12`, `BODY 2:
  70 X 14`),
- `table` – any ruled block of fields,
- `view_caption` – the caption under a view (e.g. `TOP`, `VIEW A`,
  `ITEM 3`).

A sample's ground truth is a `FeatureList`, which behaves like a list of
features:

```python
class FeatureList(BaseModel):
    feature_list: list[Feature]

    def __getitem__(self, key):
        return self.feature_list[key]

    def __len__(self):
        return len(self.feature_list)
```

### 4.4 Table models

Three pydantic models are defined for tables: `TableCell`, `TableRow`, and
`Table`, arranged in that hierarchy. `Table` and `TableRow` behave basically
like a list, while `TableCell` is more elaborate:

```python
class TableCell(BaseModel):
    score: Optional[float] = None
    text: Optional[str] = None
    fieldtype: Optional[str] = None
    line_item_id: int
    colspan: int = 1
    rowspan: int = 1
```

Cell localisation (bounding box, page) is currently not used.

Note that table support in this framework is only partially implemented: the
table schema (the models below) and the table metrics (section 8.2) exist,
but there is no fully working table dataset yet.

Below is an explanation of each field:

- `score` – the probability, assigned by the model, that the cell actually
  exists,
- `text` – the cell's content,
- `fieldtype` – the cell's type. Currently a string, though the literature
  defines specific cell types, so this should probably become an Enum in the
  future – worth considering,
- `line_item_id` – the id of the row the cell belongs to,
- `colspan` – how many columns the cell spans (cell width),
- `rowspan` – how many rows the cell spans (cell height).

### 4.5 Implementing your own dataset module

To create your own module, the simplest approach is to import
`DatasetClassBase` and `SampleClassBase` from the `datasets.base` module,
then write a `DatasetClass` and a `SampleClass` (the names must match
exactly, otherwise there will be an import problem when running the
benchmark) that inherit from `DatasetClassBase` and `SampleClassBase`
respectively. Don't forget to implement the abstract methods for these
classes – if you do, trying to run a benchmark with that dataset should fail
with an error.

It's worth noting that a `datasets.custom` module is already implemented
(mainly for drawings). It loads its data in the COCO format, so you can
create a folder for your module, place the annotations in
`_annotations.coco.json`, and inherit your own `DatasetClass` and
`SampleClass` from the ones in `datasets.custom`, as in the template below:

```python
class SampleClass(SampleClassBase):
    def __init__(self, id, coco, dirname):
        self.img_info = coco.loadImgs(coco.getImgIds()[int(id)])[0]
        ann_ids = coco.getAnnIds(imgIds=self.img_info['id'])
        self.anns = FeatureList.model_validate({"feature_list": coco.loadAnns(ann_ids)})
        self.path = dirname / self.img_info["file_name"]

    def image(self):
        image = base64.b64encode(self.path.read_bytes()).decode()
        return image

    def ground_truth(self):
        return self.anns

class DatasetClass(DatasetClassBase):
    def __init__(self) -> None:
        super().__init__()
        dirname = Path(f"{os.path.dirname(os.path.abspath(__file__))}")
        coco = COCO(dirname / '_annotations.coco.json')
        img_ids = coco.getImgIds()
        self.samples = [SampleClass(i, coco, dirname) for i in range(len(img_ids))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key : int) -> SampleClassBase:
        return self.samples[key]
```

The data is laid out in the COCO style: a single `_annotations.coco.json`
file with all annotations, and the images it refers to in the same folder,
as shown in the diagram below:

```
data
├── _annotations.coco.json
├── image1.png
├── image2.png
...
```

`DatasetClass` loads the annotations with `pycocotools` and builds one
`SampleClass` per image; each sample's `ground_truth()` is the
`FeatureList` of that image's annotations, and `image()` returns the image
in base64 form.

### 4.6 List of available (fully working) datasets

- `custom_test`
- `generated`
- `drawing_testing_dataset` – a dataset with a single sample, used to test
  the drawing metrics
- `table_recognition_testing_dataset` – a dataset with a single sample, used
  to test the table metrics

## 5. Models

This section describes the `models` module. It doesn't implement anything by
itself, but contains all the submodules for each model.

### 5.1 Model base class

The base of every model looks like this:

```python
class ModelBase(ABC):
    @abstractmethod
    def forward(self, image:str) -> dict[str,tuple[str,str]]:
        ...
```

where `forward` is the given model's inference function, taking an image in
base64 form and returning a JSON object.

### 5.2 OpenRouter models

The `models.base.openrouter` module implements a `ModelClass` that fully
handles inference for OpenRouter models. You just need to inherit from this
class and set `self.name` to the OpenRouter model name in the constructor.
If a given model needs a more advanced way of scaling images, you should
implement a `factor` function, which should return the number that the
coordinates returned by the model are multiplied by, depending on the
image's dimensions (Gemini's scaling approach is implemented by default).
Below is an example implementation of the class:

```python
from models.base.openrouter import ModelClass as ModelClassBase

class ModelClass(ModelClassBase):
    def __init__(self, MAX_TOKENS = 100000):
        super().__init__(MAX_TOKENS)
        self.name:str = "openai/gpt-5.4-mini"

    def factor(self, width: int, height: int):
        ...
```

### 5.3 Implementing your own module

To create your own module with a model, just inherit from `ModelBase` in the
`models.base` module and implement the `forward` function, which takes an
image in base64 form and returns the model's output, ideally as a pydantic
model.

### 5.4 List of available (fully working) models

- `gemini_3_1_flash_lite`
- `gemini_3_5_flash`
- `openai_gpt5_4_mini`
- `openai_gpt5_5`
- `openai_gpt5_6_Sol`

## 6. Metrics

This section describes the `metrics` module. It doesn't implement anything
by itself, but contains all the submodules for each metric.

### 6.1 Metric base class

The base of every metric looks like this:

```python
class MetricBase(ABC):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.threshold = 0.5

        self.clear()

    @abstractmethod
    def aggregate(self, output, target):
        ...

    @abstractmethod
    def value(self):
        ...

    @abstractmethod
    def clear(self):
        ...

    def forward(self, output, target):
        self.aggregate(output, target)
        return self.value()
```

where:

- `aggregate` – lets you fold a sample into the metric's running computation
  (`output` is the model's output, `target` is the sample object),
- `value` – returns the metric's value computed from everything aggregated
  so far,
- `clear` – clears all internal state, "forgetting" previously aggregated
  samples,
- `forward` – takes a model output (`output`) and a sample (`target`), adds
  them to the aggregated values, and returns the metric value computed from
  everything aggregated so far.

The `metrics.base` module also provides a few helper functions used by the
drawing metrics: `norm` (normalises a feature label for comparison), `ned`
(normalised edit distance), `lcs` (longest common subsequence of two
strings), `cost` (a similarity score combining LCS and NED), and
`match_labels` (pairs labels with candidates, allowing exact and partial
matches).

### 6.2 Implementing your own metric

To create your own metric module, just inherit from `MetricBase` in the
`metrics.base` module and implement the `aggregate`, `value`, and `clear`
functions as described in section 6.1.

### 6.3 List of available (fully working) metrics

- `bbox_f1`
- `bbox_precision`
- `bbox_recall`
- `custom_DocILE_LIR_f1`
- `custom_DocILE_LIR_recall`
- `features_cer_localisation_padded`
- `features_cer_localisation_unpadded`
- `features_cer_unstructured`
- `features_f1_em_localisation`
- `features_f1_em_unstructured`
- `GriTS_Top`
- `TEDS_S`

A fairly detailed description of every metric is given in section 8 below.

## 7. Tests

Tests are written using `pytest`. All test-related files live in `tests/`.
Currently the tests only check that each metric is computed correctly.
Ground truth for the tests lives in the `datasets.drawing_testing_dataset`
and `datasets.table_recognition_testing_dataset` modules (for drawings and
tables respectively), and in the `testing_data_drawings.json` and
`testing_data_tables.json` files (also respectively for drawings and
tables).

## 8. Metric details

### 8.1 Drawing metrics

All drawing metrics operate on the drawing ground truth described in section
4.3 (the `Feature`/`FeatureList` models). There are other output formats in
use, though (potentially better ones), for which we don't have metrics yet.

#### 8.1.1 Drawing metric categories

**Localisation metrics**

Localisation metrics assess how well a model can determine **where** a given
annotation is. They are not as important as semantic-localisation or
semantic metrics, but can be useful when location is needed at later stages
of the pipeline.

**Semantic metrics**

Semantic metrics assess how well a model can extract **information** from
the image (the content itself, without looking at location). These are the
most important metrics (or tied with semantic-localisation metrics), since
what matters most is how well the model performs at data extraction.

**Semantic-localisation metrics**

Semantic-localisation metrics combine semantic and localisation evaluation —
content is only checked based on what was paired by location. They let you
verify that the model reads content correctly in the places it correctly
located itself. This is the second most important type of metric (or tied
with semantic metrics), depending on whether location will be useful in
later pipeline stages.

**Hallucination/detection metrics**

Hallucination/detection metrics give a single number for how well the model
finds objects and how many hallucinations it produces in doing so (i.e.
cases where the model claims something exists when it doesn't). They're
useful for a general assessment of the model's detection ability.

#### 8.1.2 bounding boxes (the location where a given annotation is)

The `bbox_*` metrics replaced the old balloon-based location metrics. Instead
of pairing annotations by distance, they work on the annotations' bounding
boxes and pair them by IoU (intersection over union).

**precision**

**Category:** hallucination/detection metric.

**What it measures:** precision for bounding-box detection — the fraction of
the model's detected boxes that match a ground-truth box.

**How it works:** the generated and ground-truth boxes are paired one-to-one
with the Hungarian algorithm, based on the IoU of the boxes. A pair counts as
a true positive if the IoU is above a set threshold. Precision is the number
of true positives divided by the total number of generated boxes
(TP / (TP + FP)), accumulated across all samples.

**Value range:** [0, 1] — higher is better.

**recall**

**Category:** hallucination/detection metric.

**What it measures:** recall for bounding-box detection — the fraction of
ground-truth boxes that the model correctly found.

**How it works:** the pairing is identical to precision (Hungarian algorithm
on box IoU, pairs above a set threshold count as true positives). Recall is
the number of true positives divided by the total number of ground-truth
boxes (TP / (TP + FN)), accumulated across all samples.

**Value range:** [0, 1] — higher is better.

**f1**

**Category:** hallucination/detection metric.

**What it measures:** the f1 score, i.e. a combination of detection
performance and hallucination level in a single number.

**How it works:** the generated and ground-truth boxes are paired one-to-one
so as to maximise the number of matched pairs (Hungarian algorithm), where a
pair matches if the IoU of the two boxes is above a set threshold. Based on
this we count: true positives (number of pairs), false positives (generated,
unpaired boxes) and false negatives (unpaired ground-truth boxes) —
accumulated across all samples — from which f1 is finally computed using the
standard formula.

**Value range:** [0, 1] — higher is better. If the accumulated TP, FP and FN
are all 0, the metric value is 1 (the model had no opportunity to make a
mistake).

#### 8.1.3 features (extracted data/dimensions together with their label and category)

**f1 (unstructured)**

**Category:** hallucination/detection metric and semantic metric.

**What it measures:** the f1 score, i.e. a combination of detection
performance, hallucination level, and correctness of the extracted content
in a single number.

**How it works:** two features are paired one-to-one if and only if their
label (the extracted text) and category are identical (exact match) — the
Hungarian algorithm is used to find the pairing that maximises the number of
matched pairs. We count TP (number of pairs), FP (generated, unpaired
features) and FN (unpaired ground-truth features), accumulated across all
samples, from which f1 is computed using the standard formula.

**Value range:** [0, 1] — higher is better. If the accumulated TP, FP and FN
are all 0, the metric value is 1.

**cer (unstructured)**

**Category:** semantic metric.

**What it measures:** the mean character error rate (CER), i.e. the
normalised edit distance between the labels of paired features — how closely
the extracted text resembles the correct one. The reported value is 1 minus
the mean CER, so identical labels give 1 and completely different labels
give 0.

**How it works:** features are paired one-to-one so as to minimise the sum of
normalised edit distances between labels (Hungarian algorithm). Unpaired
features (when there are too many on one side) are skipped.

**Value range:** [0, 1] — higher is better (1 means identical labels, 0
means completely different).

**f1 (localisation)**

**Category:** hallucination/detection metric and semantic-localisation
metric.

**What it measures:** the f1 score, i.e. a combination of detection
performance and hallucination level in a single number, where the candidate
pair for comparison is chosen based on location.

**How it works:** first, candidate pairs of ground-truth and generated
features are formed based on location — the Hungarian algorithm is applied to
the IoU (intersection over union) of the features' bounding boxes. A pair is
a true positive if the IoU exceeds a set threshold and the features' labels
(the extracted text) and categories are identical. We count TP (number of
pairs), FP (generated, unpaired features) and FN (unpaired ground-truth
features), from which f1 is computed.

**Value range:** [0, 1] — higher is better. Pairing by location here is only
used to decide which features to compare — the metric value itself reflects
detection performance and content correctness, not localisation quality.

**cer (localisation, padded)**

**Category:** semantic-localisation metric and hallucination/detection
metric.

**What it measures:** like the unpadded variant, the mean character error
rate between the labels of features paired by location, with the difference
that unpaired features are also penalised.

**How it works:** the pairing is identical to the unpadded variant, but
features that could not be paired are not skipped — each unpaired feature
contributes the maximum error (1) to the average. The metric therefore also
penalises detection mistakes (missing or hallucinated features).

**Value range:** [0, 1] — higher is better.

### 8.2 Table metrics

#### 8.2.1 Table metric categories

**Structural metrics**

Structural metrics assess how correctly the model reproduces the table's
structure (e.g. the split into rows and cells), regardless of whether the
content read within the cells is correct. They're useful for verifying that
the model handles table layout recognition well.

**Semantic-structural metrics**

Semantic-structural metrics combine an assessment of table structure with an
assessment of content correctness — cells are matched partly based on
structure (e.g. row membership), and then content agreement is checked. They
let you verify that the model reads data correctly in the context of its
correct placement within the table's structure.

#### 8.2.2 TEDS-S

**Category:** structural metric.

**What it measures:** the agreement of table structure (a tree made of rows
and cells) between the generated and ground-truth table.

**How it works:** an implementation of the TEDS-S metric from
[this paper](https://arxiv.org/abs/1911.10683), considering only `tr` and
`td` nodes (rows and cells). Cells are compared structurally — two cells are
the same if and only if their field type, colspan and rowspan are identical
(cell content is not taken into account). No ready-made implementation could
be found for the tree-edit-distance part of the metric, so the current
approach is custom-built — created by Franciszek Zachuta based on dynamic
programming. The resulting tree edit distance is normalised by the maximum
number of nodes in either table, and the metric value is 1 minus that
normalised distance.

**Value range:** [0, 1] — higher is better.

#### 8.2.3 custom DocILE-LIR f1

**Category:** semantic-structural metric.

**What it measures:** the f1 score for table cell matching, combining
structural correctness (row membership) and content correctness.

**How it works:** an implementation of the DocILE LIR metric paradigm from
[this paper](https://arxiv.org/abs/2302.05658), with some modifications.
First, for every pair of rows (ground truth and generated), the number of
matched cells is computed using LCS (Longest Common Subsequence) — two cells
are considered the same if and only if their text (content), colspan and
rowspan are exactly identical (field type is not compared). Then, based on
these counts, the best row assignment is found, maximising the total number
of matched cells. Finally we count TP (number of matched cells), FP
(generated, unmatched cells) and FN (unmatched ground-truth cells), from
which f1 is computed.

**Value range:** [0, 1] — higher is better.

#### 8.2.4 custom DocILE-LIR recall

**Category:** semantic-structural metric.

**What it measures:** recall for table cell matching — what fraction of the
ground-truth cells were correctly found.

**How it works:** analogous to custom DocILE-LIR f1 — for every pair of
rows, the number of matched cells is computed using LCS (cells are the same
if and only if text, colspan and rowspan are identical — field type is not
compared), and then the best row assignment maximising the total number of
matched cells is found. Finally we count TP (number of matched cells) and FN
(unmatched ground-truth cells), from which recall is computed. Unlike f1,
this metric does not account for hallucinations (false positives).

**Value range:** [0, 1] — higher is better.

#### 8.2.5 GriTS-Top

**Category:** structural metric.

**What it measures:** the agreement of table topology (a grid-based
structure) between the generated and ground-truth table, without assessing
cell content.

**How it works:** an implementation of the topological GriTS metric from
[this paper](https://arxiv.org/abs/2203.12555). Rows and columns are matched
between the two grids, and the similarity of each pair of cells is computed
as the IoU (intersection over union) of their bounding boxes; the total
similarity is normalised by the number of cells. If the predicted (or
ground-truth) table cannot be converted into the grid form described in the
paper, the similarity is returned as 0.

**Value range:** [0, 1] — higher is better.

### 8.3 Algorithms used in metric computation

Below is a brief, practical explanation of three algorithms that come up
repeatedly in the metric descriptions above.

**Hungarian algorithm**

**What it's for:** finding the best possible one-to-one pairing between two
sets of elements, so that the sum (or mean) "cost" of all pairs is as small
as possible.

**How it works in practice:** imagine we have 3 generated boxes and 3
ground-truth boxes. For every possible pair (generated, ground truth) we
can compute the distance between them — this is that pair's "cost". The
Hungarian algorithm checks all sensible pairing combinations (without brute
forcing every single possibility, but in a smart, fast way) and picks the
set of pairs where each element is paired at most once and the total cost
of all pairs is the smallest possible.

**Example:** we have boxes A1, A2 (generated) and B1, B2 (ground truth).
Distances: A1-B1 = 10, A1-B2 = 100, A2-B1 = 90, A2-B2 = 20. Pairing them "in
order" (A1-B1, A2-B2) gives a total of 10+20=30. Pairing them crosswise
(A1-B2, A2-B1) would give 100+90=190. The Hungarian algorithm picks the
first pairing, since it has the lower total cost.

**Edit distance / Levenshtein distance**

**What it's for:** measuring how different two character strings (e.g. two
labels/texts) are from each other.

**How it works in practice:** it counts the minimum number of single
operations (inserting a character, deleting a character, substituting one
character for another) needed to turn one text into the other. The fewer
operations needed, the more similar the texts are. "Normalized" Levenshtein
distance means this result is additionally divided by the text length, so
the result falls in the [0, 1] range regardless of whether short or long
texts are being compared.

**Example:** to turn "kot" into "kod", you only need to substitute one
letter (t → d) — the edit distance is 1. To turn "kot" into "psy", you need
to substitute all 3 letters — the edit distance is 3.

**Longest Common Subsequence (LCS)**

**What it's for:** finding the largest set of elements that occur in the
same order in both compared sequences (e.g. in two table rows), even if
other, non-matching elements sit between them.

**How it works in practice:** unlike a plain element-by-element check, LCS
allows skipping elements that don't match, and looks for the longest
possible "path" of common elements that keep their relative order.

**Example:** the ground-truth row has cells [A, B, C, D], and the generated
row has cells [A, X, C, D, Y]. Although they aren't identical, the common
subsequence is [A, C, D] (length 3) — these elements appear in the same
order in both rows, and X and Y are simply skipped as non-matching.

## 9. Related subsystems

The repository also contains two related subsystems, each with its own
README:

- `generation/` – **AutoDraft**, which turns STEP / IGES / BREP models into
  dimensioned 2D drawings rendered as PNG with COCO detection labels. See
  `generation/README.md`.
- `extraction/` – an RF-DETR object detector trained on AutoDraft's output
  layout. See `extraction/README.md`.