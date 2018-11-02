import numpy as np
import pandas as pd
# eps = np.finfo(float).eps

class DecisionTree(object):
    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()

    def _get_split_atr(self, data, target):
        remainder = []
        for atr in list(data):
            tmp = pd.concat([data[atr], target], axis=1)
            remainder.append(self._remainder(tmp))
        return np.max(remainder)


    def _remainder(self, attribute):
        atr = list(attribute)[0]
        grouped = attribute.groupby(atr)
        groups_name = list(attribute[atr].unique())
        count = dict(attribute[atr].value_counts())
        total = attribute.count()[0]
        remainder = 0
        for group in groups_name:
            entropy = self._entropy(list(grouped.get_group(group).iloc[:,1].value_counts()))
            remainder += count[group]/total*entropy
        return remainder

    def _entropy(self, values):
        total = np.sum(values)
        pb = [i/total for i in values]
        return np.sum([-p*np.log(p) for p in pb])

def main():
    a = DecisionTree()
    load = pd.read_csv("data.csv")
    data = load.iloc[:,:-1]
    target = load.iloc[:,-1]
    print(a._get_split_atr(data,target))

if __name__ == "__main__":
    main()