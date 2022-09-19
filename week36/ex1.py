#ex1
import numpy as np
import pandas as pd
import seaborn

df1 = pd.read_csv("employees.csv")
df2 = pd.read_csv("departments.csv")
df3 = df1.merge(df2, how='inner')
del df3['image']


#ex2
# no. of entries
print(df3.shape[0])

#genders
#men
print(df3['gender'].value_counts()[0])
#women
print(df3['gender'].value_counts()[1])

#percentages
#men
print((df3['gender'].value_counts()[0]/df3.shape[0]).round(4)*100)
#women
print((df3['gender'].value_counts()[1]/df3.shape[0]).round(4)*100)

#salaries
print(f"Max: {df3['salary'].max()}")
print(f"Min: {df3['salary'].min()}")
print(f"Avg: {df3['salary'].mean()}")

#tuotekehitys avg:
print(f"Tuotekehitys avg: {df3.where(df3['dep']==4)['salary'].mean()}")

#no secondary phone:
print(f"No secondary phone: {df3['phone2'].isnull().sum()}")

#add ages:
df3['age'] = pd.to_datetime("today").year-pd.to_datetime(df3['bdate']).dt.year

#add age_group
df3['age_group'] = np.ceil(df3['age']/5)*5

print(df3[['age', 'age_group']])

#choose only salary, age and gender into new dataframe
df4 = df3[['salary','age','gender']]
seaborn.heatmap(df4.corr(),annot=True)
