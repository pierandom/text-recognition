from typing import List


class WordAccuracy:
    def __init__(self):
        self.correct = 0
        self.count = 0
    
    def update(self, pred: List[str], target: List[str]):
        for p, t in zip(pred, target):
            self.correct += int(p == t)
            self.count += 1
    
    def compute(self):
        return self.correct / self.count



class Mean:
    def __init__(self):
        self.total = 0
        self.count = 0
    
    def update(self, val):
        self.total += val
        self.count += 1
    
    def compute(self):
        return self.total / self.count