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
x = df.iloc[:, [1]]

x_tr, x_te, y_tr, y_te = TTS(x, y, test_size=0.2, random_state=0)

scaler_x = SS()
x_tr = scaler_x.fit_transform(x_tr)
x_te = scaler_x.transform(x_te)

model=LR()
model.fit(x_tr, y_tr)

y_pred=model.predict(x_te)


cm = confusion_matrix(y_te,y_pred)
acc = accuracy_score(y_te,y_pred)
recal = recall_score(y_te,y_pred)
precision = precision_score(y_te,y_pred)

print(f'Confusion Matrix: {cm}')
print(f'Accuracy: {acc}')
print(f'Recall: {recal}')
print(f'Precision: {precision}')

# With only age one can't really get precision

tn,fp,fn,tp = cm.ravel()

sns.heatmap(cm, annot=True, fmt='g')

df_pred = pd.read_csv('titanic-new.csv').drop(['Gender','PClass'], axis=1)
df_pred = scaler_x.transform(df_pred)
y_new_pred = model.predict(df_pred)
y_new_prob = model.predict_proba(df_pred)

print(f"Prediction: {y_new_pred}")
print(f"Probability {y_new_prob}")
plt.show()