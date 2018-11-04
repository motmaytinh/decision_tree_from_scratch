import numpy as np
import pandas as pd
from decision_tree import DecisionTree

class RandomForest(object):
    def __init__(self):
        self.forest = []

    def fit(self, data, target, num_tree):
        for i in range(num_tree):
            self.forest.append(DecisionTree().fit(data, target))

    def predict(self, data):
        votes = []
        for tree in self.forest:
            votes.append(tree.predict)

