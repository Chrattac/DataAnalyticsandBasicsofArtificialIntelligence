import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd

df = pd.read_csv('emp-dep.csv')

m_pros = round((df.gender==0).sum() / df.shape[0] * 100,3)
n_pros = round((df.gender==1).sum() / df.shape[0] * 100,3)

gvc = df['gender'].value_counts()
gvc.plot(kind='pie', ylabel='', startangle=270, autopct='%1.3f%%')

mpl.show()
