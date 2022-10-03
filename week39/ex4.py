import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency as chi 
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv("housing.csv")

plt.scatter(df.median_house_value, df.median_income)
plt.show()

data = pd.DataFrame(columns=["MHV", "MI"])
data.MHV, data.MI = df.median_house_value, df.median_income
sns.heatmap(data.corr(), annot=True, linewidths=0.1)
plt.show()

print(f"Correlation: {data.corr()}")
print(f"P-Value: {chi(data)[1]}\n")

hval_tr, hval_te, inc_tr, inc_te = tts(data.MHV, data.MI, test_size=0.2)
hval_tr = np.array(hval_tr).reshape(-1,1)
inc_tr = np.array(inc_tr).reshape(-1,1)
hval_te = np.array(hval_te).reshape(-1,1)
inc_te = np.array(inc_te).reshape(-1,1)

regr = lr()
regr.fit(hval_tr, inc_tr)

print(f"Equation: {regr.coef_[0][0]}x + {regr.intercept_[0]}\n")

inc_pred = regr.predict(hval_te)

hist_frame = pd.DataFrame(inc_te, columns=['Real value'])
hist_frame2 = pd.DataFrame(inc_pred, columns=['Predict Value'])
hist_frame = hist_frame.join(hist_frame2['Predict Value'])
hist_frame.hist()
plt.show()

print(f"MAE: {mean_absolute_error(inc_te, inc_pred)}\n\
MSE: {mean_squared_error(inc_te, inc_pred)}\n\
RMSE: {mean_squared_error(inc_te, inc_pred, squared=False)}\n\
R2: {r2_score(inc_te, inc_pred)}\n")

sns.regplot(x=hval_te, y=inc_pred)
plt.show()
sns.regplot(x=hval_te, y=inc_te)
plt.show()


print(f"Median income for 30 000 house value: {regr.predict(np.array(30000).reshape(-1,1))}")