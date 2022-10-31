import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import classification_report

df = pd.read_csv('iris.csv')

print(df.describe())

x = df.drop(['Species', 'Class'], axis=1)
y = df.iloc[:,-2]


x_tr, x_te, y_tr, y_te = TTS(x, y, test_size=0.25, random_state=1)

rfc = RFC(n_estimators=5, criterion='entropy', random_state=2)
rfc.fit(x_tr, y_tr)


prediction = rfc.predict(x_te)
cm = CM(y_te, prediction)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_te,prediction))

#Would use pickle and other file...
df_new = pd.read_csv('new-iris.csv')

iris_new = rfc.predict(df_new)
for iris in iris_new:
    if iris == 0:
        print('setosa')
    elif iris == 1:
        print('versicolor')
    else:
        print('virginica')