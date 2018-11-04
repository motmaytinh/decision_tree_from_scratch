import numpy as np
import pandas as pd
from collections import deque
# eps = np.finfo(float).eps

class Node(object):

    def __init__(self, split_attribute = None, label = None):
        # self.entropy = entropy   # entropy, will fill later
        self.split_attribute = split_attribute # which attribute is chosen, if non-leaf
        self.label = label       # label of node if it is a leaf
        # self.condition = condition
        self.children = []
        self.condition = []
    
    def add_child(self, child):
        self.children.append(child)

    def set_condition(self, condition):
        self.condition = condition

class DecisionTree(object):
    def fit(self, data, target):
        # self.Ntrain = data.count()[0]
        self.data = pd.concat([data,target],axis=1)
        # self.attributes = list(data)
        # self.target = target 
        # self.labels = target.unique()
        rem, atr, condition = self._get_split_atr(self.data)

        self.root = Node(split_attribute=atr)
        self.root.set_condition(condition)
        tmp = self._get_sub_table(self.data, atr)

        q = deque()
        q.append((self.root, tmp))

        while q:
            node, subtables = q.popleft()
            for subtable in subtables:
                # print(subtable)
                rem, atr, condition = self._get_split_atr(subtable)
                # print(atr)
                if subtable.shape[1] == 2:
                    # print(subtable)
                    # print(subtable.shape)
                    child = Node(label=self._get_label(subtable))
                    node.children.append(child)
                    continue
                elif rem == 0:
                    child = Node(atr)
                    child.set_condition(condition)
                    node.add_child(child)
                    sub = self._get_sub_table(subtable, atr)
                    for s in sub:
                        tmp = Node(label=self._get_label(s))
                        child.add_child(tmp)
                    continue
                sub = self._get_sub_table(subtable, atr)
                child = Node(atr)
                child.set_condition(condition)
                node.add_child(child)
                q.append((child, sub))
        print("hello")

    def _get_label(self, table):
        return table.iloc[:,-1].value_counts(sort=True).index[0]

    def _get_sub_table(self, data, atr):
        groups = data[atr].unique()
        grouped = data.groupby(atr)
        subtables = []
        for group in groups:
            subtables.append(grouped.get_group(group).drop(atr,1))
        return subtables

    def _get_split_atr(self, input):
        data = input.iloc[:,:-1]
        target = input.iloc[:,-1]
        remainder = 1
        split_atr = ""
        for atr in list(data):
            tmp = pd.concat([data[atr], target], axis=1)
            rm = self._remainder(tmp)
            if rm < remainder:
                remainder = rm
                split_atr = atr
        return remainder, split_atr, list(data[split_atr].unique())


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
        val = np.array(values)
        total = val.sum()
        pb = val/float(total)
        return np.sum(-pb*np.log(pb))

def main():
    a = DecisionTree()
    load = pd.read_csv("data.csv")
    data = load.iloc[:,:-1]
    target = load.iloc[:,-1]
    print(a.fit(data,target))
    # print(a._get_split_atr(data,target))

if __name__ == "__main__":
    main()