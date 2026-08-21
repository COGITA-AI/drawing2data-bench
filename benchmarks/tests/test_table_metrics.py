from pathlib import Path
from datasets.table_recognition_testing_dataset import DatasetClass
from datasets.base.tables import Table
import pytest
import json

@pytest.fixture
def testing_data():
    target = DatasetClass()[0]
    output = Table.model_validate_json((Path(__file__).parent / "testing_data_tables.json").read_text())

    return output, target

def test_custom_DocILE_LIR_f1(testing_data):
    output, target = testing_data
    from metrics.custom_DocILE_LIR_f1 import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.6521739130)

def test_custom_DocILE_LIR_recall(testing_data):
    output, target = testing_data
    from metrics.custom_DocILE_LIR_recall import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.5769230769)

def test_TEDS_S(testing_data):
    output, target = testing_data
    from metrics.TEDS_S import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.6969696969)

def test_GriTS_Top(testing_data):
    output, target = testing_data
    from metrics.GriTS_Top import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.7539682539)