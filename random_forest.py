import numpy as np
import pandas as pd
from decision_tree import DecisionTree

class RandomForest(object):
    def __init__(self):
        self.forest = []

    def fit(self, data, target, num_tree):
        high = data.shape[0]
        for i in range(num_tree):
            idx = np.random.randint(0,high,high//2)
            tree = DecisionTree()
            tree.fit(data.iloc[idx], target.iloc[idx])
            self.forest.append(tree)

    def predict(self, data):
        votes = []
        
        for tree in self.forest:
            votes.append(tree.predict(data))

        return self._most_common(votes)

        
    def _most_common(self, lst):
        return max(set(lst), key=lst.count)

def main():
    clf = RandomForest()
    load = pd.read_csv("mushrooms.csv")
    data = load.iloc[:,1:]
    target = load.iloc[:,0]
    clf.fit(data,target,5)
    for i in range(12):
        print(clf.predict(data.iloc[i,:]))


if __name__ == "__main__":
    main()
