import pandas as pd
import pickle  

with open('startup-model.pickle', 'rb') as f: model = pickle.load(f)
with open('startup-ct.pickle', 'rb') as f: ct = pickle.load(f)
with open('startup-scaler-x.pickle', 'rb') as f: scaler_x = pickle.load(f)
with open('startup-scaler-y.pickle', 'rb') as f: scaler_y = pickle.load(f)

Xnew = pd.read_csv('new_company.csv')
Xnew_org = Xnew


Xnew = scaler_x.transform(Xnew)
ynew = scaler_y.inverse_transform(model.predict(Xnew))

print(f'{Xnew_org.iloc[0]}\nVoitto: {ynew[0][0]}\n')


        

df = pd.read_csv('new_company_ct.csv')

new_companies = df
new_companies = ct.transform(new_companies)
new_companies = model.predict(new_companies)

print(f'{df.iloc[0]}\nVoitto: {new_companies[0][0]}\n')