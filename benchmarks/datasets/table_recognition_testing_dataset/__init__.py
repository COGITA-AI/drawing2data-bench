"""A dataset with a single sample, used to test the table metrics.

The ground truth is a ``Table`` (section 4.4 of the README). Table support
is only partially implemented — this dataset exists solely so the table
metrics have a target to test against.
"""
from ..base.tables import Table
from ..base import SampleClassBase
from ..base import DatasetClassBase


class SampleClass(SampleClassBase):
    def __init__(self):
        self._table = Table.model_validate(
            {
                "table": [
                    {
                        "row": [
                            {"text": "Item", "fieldtype": "Description", "line_item_id": 0},
                            {"text": "Qty", "fieldtype": "Qty", "line_item_id": 0},
                            {"text": "Price", "fieldtype": "Unit Price", "line_item_id": 0},
                        ]
                    },
                    {
                        "row": [
                            {"text": "Bolt", "fieldtype": "Description", "line_item_id": 1},
                            {"text": "4", "fieldtype": "Qty", "line_item_id": 1},
                            {"text": "2.50", "fieldtype": "Unit Price", "line_item_id": 1, "colspan": 1},
                        ]
                    },
                ]
            }
        )

    def image(self):
        raise NotImplementedError("Table samples have no image in this dataset.")

    def ground_truth(self) -> Table:
        return self._table


class DatasetClass(DatasetClassBase):
    def __init__(self) -> None:
        self.samples = [SampleClass()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key: int):
        return self.samples[key]
