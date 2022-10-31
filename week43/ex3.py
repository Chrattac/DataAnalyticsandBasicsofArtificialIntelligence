import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import StandardScaler as SS
from sklearn.compose import ColumnTransformer as CT
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

df = pd.read_csv('titanic.csv')

x = df.drop(['Survived'], axis=1)
y = df.Survived

ohe = OHE(drop='first')
ct = CT(transformers=[('encoder', ohe, ['Gender', 'PClass'])],remainder='passthrough')

x=ct.fit_transform(x)


x_tr, x_te, y_tr, y_te = TTS(x, y, test_size=0.25, random_state=0)

scaler_x = SS()
x_tr = scaler_x.fit_transform(x_tr)
x_te = scaler_x.transform(x_te)

model = tree.DecisionTreeClassifier()
model.fit(x_tr, y_tr)

plt.figure(figsize=(120,80))
tree.plot_tree(model, rounded=True, class_names=['dies', 'survives'], filled=True)
plt.show()

prediction = model.predict(x_te)
cm = CM(y_te, prediction)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()

print(classification_report(y_te,prediction))

df_new = pd.read_csv('titanic-new.csv')
df_new_pred = ct.transform(df_new)
df_new_pred = scaler_x.transform(df_new_pred)
y_new=model.predict(df_new_pred)

for i in range(len(df_new_pred)):
    print(f'{df_new.iloc[i].values}')
    print(f'Survives: {y_new[i] == 1}\n')