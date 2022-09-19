import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#In the evening
df = pd.read_csv('titanic.csv')

df.Age_Group = np.ceil(df.Age/5)*5
plt.bar(df.Age_Group.unique(), df.Age_Group.value_counts())
plt.show()

passengers = df.id.count()
group= df.groupby(['Gender','Survived']).count()['id']

men = df[df.Gender != 'female']
men = men[men.Survived != 0]
men_surv = men.shape[0]
women = df[df.Gender != 'male']
women = women[women.Survived != 0]
women_surv = women.shape[0]

df_labels = ['Men', 'Women']
df_survived = [men_surv, women_surv]
plt.pie(df_survived, labels=df_labels,autopct='%1.3f%%')
plt.title(f"Passengers: {passengers}\n Survived men: {men_surv}\n Survived women: {women_surv}")
plt.ylabel('Survived')
plt.show()

df2 = df[df.PClass != '*']
df2.Saved = np.where(df2.Survived==1, 'Yes', 'No')
sns.boxplot(x=df2.PClass, y=df.Age, hue=df2.Saved)
plt.show()
sns.swarmplot(x=df2.PClass, y=df2.Age, hue=df2.Saved)