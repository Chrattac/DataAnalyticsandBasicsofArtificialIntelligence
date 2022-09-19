import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('emp-dep.csv')

plt.scatter(x=df.age, y=df.salary)
plt.title('Employees and salaries')
plt.xlabel(f'Kesipalkka {df.salary.mean()} n:{df.shape[0]}')
plt.show()

dep_counts = df.dname.value_counts() 
#Pistesyntaksi parempi, kuin []
dep_counts.plot(kind='bar')
plt.show()


dep_counts.plot(kind='barh')
plt.show()

sns.barplot(data=dep_counts, x='dname', y='Index')
plt.show()

m_pros = round((df.gender==0).sum() / df.shape[0] * 100,1)
n_pros = round((df.gender==1).sum() / df.shape[0] * 100,1)

gvc = df['gender'].value_counts()
gvc.plot(kind='pie', ylabel='', labels=['men, women'], startangle=270, autopct='%1.1f%%')

plt.show()

