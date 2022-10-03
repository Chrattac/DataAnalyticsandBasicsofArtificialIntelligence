import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts

df=pd.DataFrame(columns = ['x','y'])

x = np.array([1,2,3,4,6,7,8])
y = 2*x+3

x_tr, x_te, y_tr, y_te = tts(x.reshape(-1, 1), y, test_size=(float)(3/7))
regr = lr()
regr.fit(x_tr, y_tr)

test_num = np.array(5).reshape(-1, 1)

plt.scatter(x,y, color="red")
plt.scatter(test_num,
            regr.predict(test_num),
            color="blue")

plt.plot(x,y, color="green")
plt.show()

print(f'coefficient: {regr.coef_}')
print(f'intercept: {regr.intercept_}')