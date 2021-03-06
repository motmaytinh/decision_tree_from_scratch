import numpy as np
import pandas as pd
from collections import deque
from graphviz import *

class Node(object):

    def __init__(self, split_attribute = None, label = None):
        self.split_attribute = split_attribute
        self.label = label
        self.children = []
        self.condition = []
    
    def add_child(self, child):
        self.children.append(child)

    def set_condition(self, condition):
        self.condition = condition

class DecisionTree(object):
    def __init__(self):
        self.styles = {
            'label': {'shape': 'circle'},
            'node': {'shape': 'rect'}
        }

    def fit(self, data, target):
        self.data = pd.concat([data,target],axis=1)
        rem, atr, condition = self._get_split_atr(self.data)

        self.root = Node(split_attribute=atr)
        self.root.set_condition(condition)
        tmp = self._get_sub_table(self.data, atr)

        q = deque()
        q.append((self.root, tmp))

        while q:
            node, subtables = q.popleft()
            for subtable in subtables:
                rem, atr, condition = self._get_split_atr(subtable)
                if subtable.shape[1] == 2:
                    # print(subtable.shape)
                    child = Node(label=self._get_label(subtable))
                    node.children.append(child)
                    continue
                elif rem == 0:
                    if len(subtable.iloc[:,-1].unique()) == 1:
                        tmp = Node(label=subtable.iloc[:,-1].unique()[0])
                        node.add_child(tmp)
                        continue
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

    def predict(self, row):
        node = self.root
        while node.label is None:
            index = node.condition.index(row[node.split_attribute])
            node = node.children[index]
        return node.label

    def draw_tree(self):
        g = Digraph('G', filename='decision_tree.gv')

        count = 0

        q = deque()
        q.append((self.root, 0))
        g.node(str(0), self.root.split_attribute, self.styles['node'])

        while q:
            tmp, idx = q.pop()
            condition = tmp.condition
            parent = tmp.split_attribute
            for i, child in enumerate(tmp.children):
                if not child.children:
                    count+=1
                    g.node(str(count), str(child.label), self.styles['label'])
                    g.edge(str(idx) , str(count), str(condition[i]))
                else:
                    count+=1
                    g.node(str(count), str(child.split_attribute), self.styles['node'])
                    g.edge(str(idx) , str(count), str(condition[i]))
                    q.append((child, count))
                
        g.render("decision_tree", format="png")
        #return g

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
    load = pd.read_csv("mushrooms.csv")
    data = load.iloc[:,1:]
    target = load.iloc[:,0]
    a.fit(data,target)
    a.draw_tree()
    for i in range(12):
        print(a.predict(data.iloc[i,:]))

if __name__ == "__main__":
    main()
