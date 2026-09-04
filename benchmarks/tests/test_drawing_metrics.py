"""Tests for the drawing metrics.

The expected values are computed BY HAND from the metric descriptions in the
README (intended behaviour), on the small synthetic data in
``testing_data_drawings.json`` / ``datasets.drawing_testing_dataset``.

The uploaded metric implementations contain bugs, so these tests are meant
to be run against the *fixed* metrics. Do not run them against the buggy
code.
"""
from pathlib import Path

import pytest

from datasets.drawing_testing_dataset import DatasetClass
from datasets.base import FeatureList


@pytest.fixture
def testing_data():
    """Return (output, target): output is the FeatureList read from the
    test-data file, target is the ground-truth sample."""
    target = DatasetClass()[0]
    output = FeatureList.model_validate_json(
        (Path(__file__).parent / "testing_data_drawings.json").read_text()
    )
    return output, target


# ---------------------------------------------------------------------------
# bbox_f1
# ---------------------------------------------------------------------------
# Output boxes:      O0 [10,10,110,60], O1 [200,10,300,60], O2 [400,10,500,60], O3 [600,10,700,60]
# Target boxes:      T0 [10,10,110,60], T1 [220,10,320,60], T2 [100,200,200,250]
# Pairwise IoU (intersection/union):
#   O0-T0: identical            -> 1
#   O1-T1: overlap x=[220,300]=80, each w=100 -> 80/(100+100-80)=80/120=2/3
#   O2-T2: no overlap           -> 0
#   O3-* : no overlap           -> 0
# With IoU > 0.5 as the match threshold: exactly 2 pairs (O0-T0, O1-T1).
# TP=2, FP=2 (O2,O3 unpaired), FN=1 (T2 unpaired)
# P=2/4=0.5, R=2/3=0.666..., F1=2*P*R/(P+R)=0.5714285714...
def test_bbox_f1(testing_data):
    output, target = testing_data
    from metrics.bbox_f1 import Metric
    metric = Metric()
    assert metric.forward(output, target) == pytest.approx(0.5714285714)


# ---------------------------------------------------------------------------
# bbox_precision
# ---------------------------------------------------------------------------
# TP=2, FP=2 -> precision = 2/4 = 0.5
def test_bbox_precision(testing_data):
    output, target = testing_data
    from metrics.bbox_precision import Metric
    metric = Metric()
    # bbox_precision inherits statistics from bbox_f1, which defaults to threshold=50.
    # To test the intended IoU-based behaviour, set threshold=0.5 on the metric.
    metric.threshold = 0.5
    assert metric.forward(output, target) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# bbox_recall
# ---------------------------------------------------------------------------
# TP=2, FN=1 -> recall = 2/3 = 0.666...
def test_bbox_recall(testing_data):
    output, target = testing_data
    from metrics.bbox_recall import Metric
    metric = Metric()
    metric.threshold = 0.5
    assert metric.forward(output, target) == pytest.approx(0.6666666667)


# ---------------------------------------------------------------------------
# features_f1_em_unstructured
# ---------------------------------------------------------------------------
# Exact text+category matches (ignoring location), pairing to maximise TP:
#   O0 "R8.9"(radius)      == T0 "R8.9"(radius)          -> match
#   O1 "17.7±0.1"(dim)     vs T1 "17.7 ±0.1"(dim)        -> not identical
#   O2 "29.6°"(dim)        vs T2 "29.6 deg"(dim)         -> not identical
#   O3 "EXTRA"(note)       (no target)                   -> FP
#   T1, T2 unpaired                                       -> FN
# TP=1, FP=3, FN=2 -> P=1/4=0.25, R=1/3=0.333..., F1=2*0.25*0.333/(0.25+0.333)=0.285714
def test_features_f1_em_unstructured(testing_data):
    output, target = testing_data
    from metrics.features_f1_em_unstructured import Metric
    metric = Metric()
    assert metric.forward(output, target) == pytest.approx(0.2857142857)


# ---------------------------------------------------------------------------
# features_cer_unstructured
# ---------------------------------------------------------------------------
# Pairs chosen to minimise total normalised edit distance (text only):
#   O0 "R8.9"     vs T0 "R8.9"      -> edit 0, len 4  -> cer 0
#   O1 "17.7±0.1" vs T1 "17.7 ±0.1" -> edit 1, len 9  -> cer 1/9
#   O2 "29.6°"    vs T2 "29.6 deg"  -> edit 4, len 8  -> cer 0.5
#   O3 "EXTRA"    -> unpaired, skipped
# mean CER = (0 + 1/9 + 1/2)/3 = (0.111111 + 0.5)/3 = 0.203704
# metric = 1 - 0.203704 = 0.796296
def test_features_cer_unstructured(testing_data):
    output, target = testing_data
    from metrics.features_cer_unstructured import Metric
    metric = Metric()
    assert metric.forward(output, target) == pytest.approx(0.7962962963)


# ---------------------------------------------------------------------------
# features_f1_em_localisation
# ---------------------------------------------------------------------------
# Candidate pairs by location (IoU > 0.5): O0-T0, O1-T1 (as in bbox_f1).
# Among candidates, match if text AND category identical:
#   O0-T0: "R8.9"=="R8.9", radius==radius -> TP
#   O1-T1: "17.7±0.1" != "17.7 ±0.1"      -> not a TP
# TP=1, FP=3 (O2,O3 + O1 unmatched as candidate), FN=2 (T1, T2)
# P=1/4, R=1/3 -> F1=0.285714
def test_features_f1_em_localisation(testing_data):
    output, target = testing_data
    from metrics.features_f1_em_localisation import Metric
    metric = Metric()
    metric.threshold = 0.5
    assert metric.forward(output, target) == pytest.approx(0.2857142857)


# ---------------------------------------------------------------------------
# features_cer_localisation_padded
# ---------------------------------------------------------------------------
# Same pairing as unpadded (O0-T0, O1-T1; O2-T2 below threshold).
# Padded counts every unmatched element as a full error (similarity 0):
#   pairs above threshold: O0-T0 sim 1, O1-T1 sim 8/9
#   unpaired elements: O3, T2 (and O2 paired to T2 but below threshold
#   counts as unpaired on both sides? -> the intended behaviour treats the
#   sub-threshold pair as two unpaired elements: O2 and T2 each contribute 0)
#   contributing elements: O0,T0,O1,T1 (2 pairs), O2,O3 (2 output), T2 (1 target)
#   total = 5 elements -> sims [1, 8/9, 0, 0, 0] -> mean = (17/9)/5 = 17/45 = 0.377778
def test_features_cer_localisation_padded(testing_data):
    output, target = testing_data
    from metrics.features_cer_localisation_padded import Metric
    metric = Metric()
    assert metric.forward(output, target) == pytest.approx(0.3777777778)
