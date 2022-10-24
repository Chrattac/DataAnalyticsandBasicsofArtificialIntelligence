# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('diabetes-new.csv')

with open('diabetes-model.pickle', 'rb') as f: model = pickle.load(f)
with open('diabetes-scaler-x.pickle', 'rb') as f: scaler_x = pickle.load(f)

x = scaler_x.transform(df)
has_diabetes = model.predict(x)
diabetes_prob = model.predict_proba(x)


for i in range(len(x)):
    print(f'{df.iloc[i]}')
    print(f"\n\
Probability:\n\
Doesn't have diabetes:{diabetes_prob[i][0]}\n\
Has diabetes:{diabetes_prob[i][1]}\n")
    print("Conclusion:")
    if (has_diabetes[i] == 1):
        print("Person is diabetic")
    else: print("Person isn't diabetic")
    print('\n')
