from ..base import MetricBase
from datasets.base.tables import Table, TableRow, TableCell
from datasets.base import SampleClassBase
from scipy.optimize import linear_sum_assignment
import numpy as np

class Metric(MetricBase):
    def __init__(self):
        super().__init__()

    def get_nodes(self, input):
        nodes = 0
        for row in input:
            nodes += len(row)

        return nodes
    
    def same(self, output: TableCell, target: TableCell) -> bool:
        # return output.fieldtype == target.fieldtype and output.text == target.text and output.colspan == target.colspan and output.rowspan == target.rowspan
        return output.text == target.text and output.colspan == target.colspan and output.rowspan == target.rowspan
    
    def lcs(self, output: TableRow, target: TableRow) -> int: 
        m = len(output) 
        n = len(target) 
        dp = np.zeros((m + 1, n + 1)) 
        
        for i in range(1, m + 1): 
            for j in range(1, n + 1): 
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                if self.same(output[i - 1], target[j - 1]): 
                    dp[i][j] = dp[i - 1][j - 1] + 1

        return dp[m][n] 

    def aggregate(self, output: Table, target: SampleClassBase):
        target = target.ground_truth()
        
        if not isinstance(output, Table):
            raise ValueError("output should be of Table type.")
        
        if not isinstance(target, Table):
            raise ValueError("target should be of Table type.")

        cost = np.zeros((len(output), len(target)))
        for i in range(len(output)):
            for j in range(len(target)):
                cost[i][j] = self.lcs(output[i], target[j])
        
        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

        true_positives = cost[row_ind, col_ind].sum()
        false_negatives = self.get_nodes(target) - true_positives

        self.tp += true_positives
        self.fn += false_negatives

    def value(self):
        recall = self.tp / max(self.tp + self.fn, 1)

        return recall
    
    def clear(self):
        self.tp = 0
        self.fn = 0
