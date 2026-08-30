from ..base import MetricBase
from datasets.base.tables import Table, TableRow, TableCell
from datasets.base import SampleClassBase
import numpy as np

class Metric(MetricBase):
    def __init__(self):
        super().__init__()

    def create_grits_grid(self, table: Table):
        grid = []

        def ensure_row(r):
            while len(grid) <= r:
                grid.append({})

        for i, row in enumerate(table):
            ensure_row(i)
            col = 0
            for cell in row:
                while col in grid[i]:
                    col += 1

                for y in range(cell.rowspan):
                    ensure_row(i + y)
                    for x in range(cell.colspan):
                        if (col + x) in grid[i + y]:
                            return None
                        grid[i + y][col + x] = [-x, -y, cell.colspan - x, cell.rowspan - y]

                col += cell.colspan

        if len(grid) == 0:
            return None

        max_len = max(len(r) for r in grid)
        if max_len == 0:
            return None

        dense_grid = []
        for r in grid:
            if len(r) != max_len or set(r.keys()) != set(range(max_len)):
                return None
            dense_grid.append([r[c] for c in range(max_len)])

        return np.array(dense_grid)
    
    def similarity(self, output, target) -> float:
        area_output = (output[2] - output[0]) * (output[3] - output[1])
        area_target = (target[2] - target[0]) * (target[3] - target[1])
        intersection = max(min(output[2], target[2]) - max(output[0], target[0]), 0) * max(min(output[3], target[3]) - max(output[1], target[1]), 0)
        union = area_output + area_target - intersection
        return intersection / (union + self.eps)

    def row_grits(self, output, target) -> float:
        m = len(output) 
        n = len(target) 
        dp = np.zeros((m + 1, n + 1)) 
        
        for i in range(1, m + 1): 
            for j in range(1, n + 1): 
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + self.similarity(output[i - 1], target[j - 1]))

        return dp[m][n]
    
    def interrow_grits(self, output, target):
        m = len(output) 
        n = len(target) 
        dp = [[[0, 0] for j in range(n + 1)] for i in range(m + 1)]
        
        for i in range(1, m + 1): 
            for j in range(1, n + 1): 
                dp[i][j][0] = dp[i - 1][j][0]
                if dp[i][j][0] < dp[i][j - 1][0]:
                    dp[i][j] = [dp[i][j - 1][0], 1]
                if dp[i][j][0] < dp[i - 1][j - 1][0] + self.row_grits(output[i - 1], target[j - 1]):
                    dp[i][j] = [dp[i - 1][j - 1][0] + self.row_grits(output[i - 1], target[j - 1]), 2]

        index_output = []
        index_target = []

        i, j = m, n
        while i != 0 and j != 0:
            change = [[-1, 0], [0, -1], [-1, -1]]
            option = dp[i][j][1]
            i += change[option][0]
            j += change[option][1]

            if option == 2:
                index_output.append(i)
                index_target.append(j)

        index_output.reverse()
        index_target.reverse()
        return np.array(index_output), np.array(index_target)
    
    def grid_similarity(self, output, target):
        sim = 0
        for row_output, row_target in zip(output, target):
            for cell_output, cell_target in zip(row_output, row_target):
                sim += self.similarity(cell_output, cell_target)

        return sim


    def aggregate(self, output: Table, target: SampleClassBase):
        target = target.ground_truth()
        
        if not isinstance(output, Table):
            raise ValueError("output should be of Table type.")
        
        if not isinstance(target, Table):
            raise ValueError("target should be of Table type.")

        grid_output = self.create_grits_grid(output)
        grid_target = self.create_grits_grid(target)

        if grid_output is None or grid_target is None:
            self.values.append(0)
            return

        index_output_row, index_target_row = self.interrow_grits(grid_output,                    grid_target)
        index_output_col, index_target_col = self.interrow_grits(grid_output.transpose(1, 0, 2), grid_target.transpose(1, 0, 2))

        grid_output_reduced = grid_output[index_output_row, :][:, index_output_col]
        grid_target_reduced = grid_target[index_target_row, :][:, index_target_col]

        metric = self.grid_similarity(grid_output_reduced, grid_target_reduced)
        metric = 2 * metric / (grid_output.shape[0] * grid_output.shape[1] + grid_target.shape[0] * grid_target.shape[1])
        
        self.values.append(metric)
    
    def value(self):
        return np.array(self.values).mean()

    def clear(self):
        self.values = []