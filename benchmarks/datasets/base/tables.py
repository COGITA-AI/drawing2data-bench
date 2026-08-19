from pydantic import BaseModel
from functools import reduce
from typing import Optional, Tuple, List

class TableCell(BaseModel):
    # we don't use localisation
    # bbox: Tuple[float, float, float, float]
    # page: int
    score: Optional[float] = None
    text: Optional[str] = None
    fieldtype: Optional[str] = None
    line_item_id: int
    colspan: int = 1
    rowspan: int = 1

class TableRow(BaseModel):
    row: List[TableCell] = []

    def __getitem__(self, key):
        return self.row[key]
    
    def __len__(self):
        return len(self.row)

    def __iter__(self):
        for item in self.row:
            yield item

class Table(BaseModel):
    table: List[TableRow] = []

    def __getitem__(self, key):
        return self.table[key]
    
    def __len__(self):
        return len(self.table)

    def __iter__(self):
        for item in self.table:
            yield item