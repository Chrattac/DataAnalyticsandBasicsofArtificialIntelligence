import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df=pd.DataFrame(columns = ['x','y'])

x = np.array([1,2,3,4,6,7,8])
df.x, df.y = x, 2*x+3

sns.regplot(x=df.x, y=df.y)
plt.show()