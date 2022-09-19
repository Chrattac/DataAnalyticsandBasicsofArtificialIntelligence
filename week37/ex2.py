import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd

df = pd.read_csv('emp-dep.csv')

mpl.bar(df.age_group.unique(), df.age_group.value_counts(), width=2)
mpl.show()