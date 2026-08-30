"""Tests for the table metrics.

The expected values are computed BY HAND from the metric descriptions in the
README (intended behaviour), on the small synthetic data in
``testing_data_tables.json`` / ``datasets.table_recognition_testing_dataset``.

The uploaded metric implementations contain bugs, so these tests are meant
to be run against the *fixed* metrics. Do not run them against the buggy
code.
"""
from pathlib import Path

import pytest

from datasets.table_recognition_testing_dataset import DatasetClass
from datasets.base.tables import Table


@pytest.fixture
def testing_data():
    """Return (output, target): output is the Table read from the test-data
    file, target is the ground-truth table sample."""
    target = DatasetClass()[0]
    output = Table.model_validate_json(
        (Path(__file__).parent / "testing_data_tables.json").read_text()
    )
    return output, target


# ---------------------------------------------------------------------------
# custom DocILE-LIR f1
# ---------------------------------------------------------------------------
# Cell equality (intended): text + colspan + rowspan (fieldtype ignored).
#
# Output (2 rows):
#   R0o: [Item, Qty, Price]              (3 cells)
#   R1o: [Bolt, 4, 2.50(colspan=2)]      (3 cells)
# Target (2 rows):
#   R0t: [Item, Qty, Price]              (3 cells)
#   R1t: [Bolt, 4, 2.50]                 (3 cells, colspan=1)
#
# LCS per row pair (cells equal if text+colspan+rowspan match):
#   lcs(R0o, R0t) = 3   (Item, Qty, Price all equal)
#   lcs(R1o, R1t): "2.50" differs (colspan 2 vs 1); Bolt, 4 match -> 2
#   lcs(R0o, R1t) = 0
#   lcs(R1o, R0t) = 0
# Best row assignment: R0o-R0t (3) + R1o-R1t (2) -> total TP = 5.
# FP = nodes_out - TP = 6-5 = 1, FN = nodes_tgt - TP = 6-5 = 1
# P = 5/6, R = 5/6 -> F1 = 5/6 = 0.833333
def test_custom_DocILE_LIR_f1(testing_data):
    output, target = testing_data
    from metrics.custom_DocILE_LIR_f1 import Metric
    metric = Metric()
    assert metric.forward(output, target) == pytest.approx(0.8333333333)


# ---------------------------------------------------------------------------
# custom DocILE-LIR recall
# ---------------------------------------------------------------------------
# Same as f1 but only TP and FN: TP=5, FN=1 -> recall = 5/6 = 0.833333
def test_custom_DocILE_LIR_recall(testing_data):
    output, target = testing_data
    from metrics.custom_DocILE_LIR_recall import Metric
    metric = Metric()
    assert metric.forward(output, target) == pytest.approx(0.8333333333)


# ---------------------------------------------------------------------------
# TEDS-S
# ---------------------------------------------------------------------------
# Structural cell equality: fieldtype + colspan + rowspan (text ignored).
# Row Levenshtein between output and target rows (structural only):
#   row_lev(R0o, R0t): identical fieldtypes -> 0
#   row_lev(R1o, R1t): "2.50" colspan differs (2 vs 1) -> cost 1
# minimize_path over the 2x2 row-cost matrix [[0,0],[1,1]]:
#   best path: match R0o-R0t (0) and R1o-R1t (1), or R0o-R0t + skip? -> total 1
# max_nodes = max(2 rows + 6 cells, 2 rows + 6 cells) = 8
# metric = 1 - 1/8 = 0.875
def test_TEDS_S(testing_data):
    output, target = testing_data
    from metrics.TEDS_S import Metric
    metric = Metric()
    assert metric.forward(output, target) == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# GriTS-Top
# ---------------------------------------------------------------------------
# The output table's last cell has colspan=2, which makes its grid ragged
# (the 3-cell row overflows the 3-column grid) -> create_grits_grid returns
# None. Per the README: if the table cannot be converted to the grid form,
# the similarity is 0.
def test_GriTS_Top(testing_data):
    output, target = testing_data
    from metrics.GriTS_Top import Metric
    metric = Metric()
    assert metric.forward(output, target) == pytest.approx(0.0)
