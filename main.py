import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2

entropy_node = 0  #Initialize Entropy
values = df.Eat.unique()  #Unique objects - 'Yes', 'No'
for value in values:
    fraction = df.Eat.value_counts()[value]/len(df.Eat)  
    entropy_node += -fraction*np.log2(fraction)