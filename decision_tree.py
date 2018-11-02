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
    data = pd.read_csv("data.csv")
    tmp = pd.concat([data['outlook'],data['play']],axis=1)
    print(a._remainder(tmp))

if __name__ == "__main__":
    main()