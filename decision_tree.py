import numpy as np
import pandas as pd
eps = np.finfo(float).eps

class DecisionTree(object):
    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()


    def _entropy(self, values):
        total = np.sum(values)
        pb = [i/total for i in values]
        return np.sum([-p*np.log2(p) for p in pb])


if __name__ == "__main__":
    main()