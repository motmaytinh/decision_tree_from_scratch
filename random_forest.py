import numpy as np
import pandas as pd
from decision_tree import DecisionTree

class RandomForest(object):
    def __init__(self, num_tree, random_attribute = False):
        self.forest = []
        self.num_tree = num_tree
        self.random_attribute = random_attribute

    def fit(self, data, target):
        high = data.shape[0]
        n_col = data.count(1).iloc[0]
        for i in range(self.num_tree):
            idx = np.random.choice(high, size=high//2, replace=False, p=None)
            tree = DecisionTree()

            if self.random_attribute:
                col_idx = np.random.choice(n_col, size=2*n_col//3, replace=False, p=None)
                tree.fit(data.iloc[idx,col_idx], target.iloc[idx])
            else:
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
    clf = RandomForest(num_tree = 5, random_attribute = True)
    load = pd.read_csv("mushrooms.csv")
    data = load.iloc[:,1:]
    target = load.iloc[:,0]
    clf.fit(data,target)
    for i in range(12):
        print(clf.predict(data.iloc[i,:]))


if __name__ == "__main__":
    main()
