from pathlib import Path
from datasets.drawing_testing_dataset import DatasetClass 
import pytest
import json

@pytest.fixture
def testing_data():
    target = DatasetClass()[0]
    output = json.loads((Path(__file__).parent / "testing_data_drawings.json").read_text())

    return output, target

def test_cuboid_volume_fraction(testing_data):
    output, target = testing_data
    from metrics.cuboid_volume_fraction import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.936894)

def test_balloons_mean_distance(testing_data):
    output, target = testing_data
    from metrics.balloons_mean_distance import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(56.962570986)

def test_balloons_f1(testing_data):
    output, target = testing_data
    from metrics.bbox_f1 import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.5714285714)

def test_feature_lines_f1(testing_data):
    output, target = testing_data
    from metrics.feature_lines_f1 import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.75)

def test_feature_lines_mean_distance(testing_data):
    output, target = testing_data
    from metrics.feature_lines_mean_distance import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(51.964900)

def test_features_f1_em_unstructured(testing_data):
    output, target = testing_data
    from metrics.features_f1_em_unstructured import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.8095238095)

def test_features_normalized_levenshtein_unstructured(testing_data):
    output, target = testing_data
    from metrics.features_cer_unstructured import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.2962962962)

def test_features_f1_em_localisation(testing_data):
    output, target = testing_data
    from metrics.features_f1_em_localisation import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.5714285714)

def test_features_normalized_levenshtein_localisation(testing_data):
    output, target = testing_data
    from metrics.features_cer_localisation_unpadded import Metric
    metric = Metric()
    
    assert metric.forward(output, target) == pytest.approx(0.0370370370)