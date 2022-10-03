import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency as chi 
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts

df = pd.read_csv("salary.csv")

print(f"correlation: {df.corr()}")
print(f"p-value: {chi(df)[1]}")
plt.scatter(df.YearsExperience, df.Salary, color="red")
plt.show()

sns.heatmap(df.corr(), annot=True, linewidth=0.1)
plt.show()

regr = lr()

exp_tr, exp_te, sal_tr, sal_te = tts(df.YearsExperience, 
                                               df.Salary, test_size = 0.3, random_state=True
exp_tr = np.array(exp_tr).reshape(-1,1)
sal_tr = np.array(sal_tr).reshape(-1,1)
regr.fit(exp_tr, sal_tr)

print(f"Coefficient: {regr.coef_[0][0]}\n\
Intercept: {regr.intercept_[0]}\n\
Equation: {regr.coef_[0][0]}x+{regr.intercept_[0]}")

exp_te = np.array(exp_te).reshape(-1,1)
sal_te = np.array(sal_te).reshape(-1,1)
sal_pred = regr.predict(exp_te)

plt.scatter(exp_te, sal_pred, color="orange")
plt.plot(exp_te, sal_pred, color="red")
plt.scatter(exp_te, sal_test, color="blue")
plt.plot(exp_te, sal_te, color="violet")
plt.show()

print(f"The salary of a new employee with \
7 years of experience would be: \
{regr.predict(np.array(7).reshape(1,-1))[0][0]}")