import re
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import linear_sum_assignment

class MetricBase(ABC):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

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
    

def norm(label: str) -> str:
    """Normalise a feature label for comparison."""
    s = str(label).upper().strip()
    s = re.sub(r"\s+", "", s)                # collapse whitespace
    s = s.replace(",", ".")                   # decimal comma → dot
    s = s.replace("\u00b1", "+/-")            # ±
    s = s.replace("\u00d7", "X")              # ×
    s = s.replace("\u2014", "-").replace("\u2013", "-")  # em/en dash
    return s

# function to compute length of the longest common subsequence
def lcs(X:str, Y:str) -> int: 
    m = len(X) 
    n = len(Y) 
    L = [[0]*(n + 1) for _ in range(m + 1)] 
    
    #composing a matrix of values  
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 

    return L[m][n] 

# implementing normalized edit distance:
def ned(s1:str, s2:str) -> float:
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    prev = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1, start=1):
        curr = [i]

        for j, c2 in enumerate(s2, start=1):
            if c1 == c2:
                curr.append(prev[j - 1])
            else:
                curr.append(
                    1 + min(
                        prev[j],      
                        curr[j - 1], 
                        prev[j - 1] 
                    )
                )

        prev = curr
    return 1-(prev[-1]/len(s1))

def cost(X:str, Y:str) -> float:
    '''
    !! placeholder !! 
    comparing longest common subsequence + all matching characters between the strings for a 50/50 score
    '''
    n = len(X)
    m = len(Y)
    score1 = lcs(X,Y)/min(n,m)
    score2 = ned(X,Y)
    print(ned(X,Y))
    score = score1*.5 + score2*.5
    return 0 if score<.5 else score

#    returns dict of {matches:..., unmatched_labels:..., unmatched_candidates:...}
def match_labels(labels:list[dict], candidates:list[dict]) -> dict:
    candidate_map = {}
    unused_candidates = []
    unused_labels = []

    for c in candidates:
        label = c.get("label", "")
        if label:
            candidate_map.setdefault(norm(label), []).append(c)

    matches = []
#assign candidates with an exact normalized label match.
    for l in labels:
        label = l.get("label", "")
        label = norm(label)
        if label and label in candidate_map and candidate_map.get(label):
            c = candidate_map[label].pop(0)
            matches.append((l, c, "exact"))
        else:
            unused_labels.append(l)


    for _, l in candidate_map.items():
        unused_candidates.extend(l)

# if used all candidates orr labels return 
    if not unused_candidates or not unused_labels:
        return {
            "matches": matches,
            "unmatched_labels": unused_labels,
            "unmatched_candidates": unused_candidates
        }

# use hungarian algorithm ( linear sum) to pick best matches between leftover labels and candidates while also allowing unmatched options
# getting a similiarity matrix
    similarity_matrix = []
    for i,c in enumerate(unused_candidates):
        similarity_matrix.append([])
        for l in unused_labels:
            similarity_matrix[i].append(cost(norm(c.get("label")), norm(l.get("label"))))
    similarity_matrix =np.array(similarity_matrix)
    cost_matrix = 1-similarity_matrix

    PAD_VAL = .51
    n, m = cost_matrix.shape
    size = n + m
    padded = np.full((size, size), PAD_VAL)
    padded[:n, :m] = cost_matrix
    rows, cols = linear_sum_assignment(padded)

    matched_rows = set()
    matched_cols = set()

    for r, c in zip(rows, cols):
        if r < n and c < m and cost_matrix[r, c] < PAD_VAL:
            matches.append((unused_labels[c], unused_candidates[r],"partial"))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_candidates = [unused_candidates[r] for r in range(n) if r not in matched_rows]
    unmatched_labels = [unused_labels[c] for c in range(m) if c not in matched_cols]
    
    return {"matches":matches, "unmatched_labels":unmatched_labels, "unmatched_candidates":unmatched_candidates}
