from ..base import MetricBase
from datasets.base.tables import Table, TableRow, TableCell
from datasets.base import SampleClassBase
import numpy as np

class Metric(MetricBase):
    def __init__(self):
        super().__init__()

    def get_nodes(self, input):
        nodes = len(input)
        for row in input:
            nodes += len(row)

        return nodes
    
    def same(self, output: TableCell, target: TableCell) -> bool:
        return output.fieldtype == target.fieldtype and output.colspan == target.colspan and output.rowspan == target.rowspan

    def row_levenshtein(self, output: TableRow, target: TableRow) -> int:
        m = len(output) 
        n = len(target) 
        dp = np.zeros((m + 1, n + 1)) 

        for i in range(m + 1):
            dp[i][0] = i

        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1): 
            for j in range(1, n + 1): 
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                if self.same(output[i - 1], target[j - 1]): 
                    dp[i][j] = dp[i - 1][j - 1]

        return dp[m][n]
    
    def minimize_path(self, cost, output: TableRow, target: TableRow) -> int:
        m, n = cost.shape
        dp = np.zeros((m + 1, n + 1)) 

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + 1 + len(output[i - 1])
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + 1 + len(target[j - 1])
        
        for i in range(1, m + 1): 
            for j in range(1, n + 1): 
                dp[i][j] = min(dp[i - 1][j] + 1 + len(output[i - 1]), dp[i][j - 1] + 1 + len(target[j - 1]), dp[i - 1][j - 1] + cost[i - 1][j - 1])

        return dp[m][n]

    def aggregate(self, output: Table, target: SampleClassBase):
        target = target.ground_truth()
        
        if not isinstance(output, Table):
            raise ValueError("output should be of Table type.")
        
        if not isinstance(target, Table):
            raise ValueError("target should be of Table type.")

        max_nodes = max(self.get_nodes(output), self.get_nodes(target), 1)

        cost = np.zeros((len(output), len(target)))
        for i in range(len(output)):
            for j in range(len(target)):
                cost[i][j] = self.row_levenshtein(output[i], target[j])
        
        metric = self.minimize_path(cost, output, target)

        metric /= max_nodes
        
        self.values.append(1-metric)

    def value(self):
        return np.array(self.values).mean()

    def clear(self):
        self.values = []
