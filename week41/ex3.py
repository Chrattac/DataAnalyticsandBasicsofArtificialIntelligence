import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split as TTS
from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler as SS
import pickle

df = pd.read_csv('titanic-class-age-gender-survived.csv')

df = df.dropna()

y = df.iloc[:, [-1]]
x = df.iloc[:, [0,1,2]]

ohc = OHE(drop='first')
ct = CT(transformers=[('encoder', ohc, ['Gender', 'PClass'])],remainder='passthrough')

x = ct.fit_transform(x)

x_tr, x_te, y_tr, y_te = TTS(x, y, test_size=0.2, random_state=0)

scaler_x = SS()
x_tr = scaler_x.fit_transform(x_tr)
x_te = scaler_x.transform(x_te)

model=LR()
model.fit(x_tr, y_tr.values.ravel())

y_pred=model.predict(x_te)


cm = confusion_matrix(y_te,y_pred)

tn,fp,fn,tp = cm.ravel()

sns.heatmap(cm, annot=True, fmt='g')
plt.show()


acc = accuracy_score(y_te,y_pred)
recal = recall_score(y_te,y_pred)
precision = precision_score(y_te,y_pred)

print(f'Accuracy: {acc}')
print(f'Recall: {recal}')
print(f'Precision: {precision}\n')


df_new = pd.read_csv('titanic-new.csv')
df_pred = ct.transform(df_new)
df_pred = scaler_x.transform(df_pred)
y_new_pred = model.predict(df_pred)
y_new_prob = model.predict_proba(df_pred)


for i in range(len(df_new)):
    print(f'{df_new.iloc[i].values}')
    print(f'Survives: {y_new_pred[i] == 1}\n')

print("Probabilities:")
sns.heatmap(y_new_prob, annot=True, fmt='g')
plt.show()