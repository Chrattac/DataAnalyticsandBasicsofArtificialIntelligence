import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split as TTS
from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler as SS
import pickle


df = pd.read_csv("startup.csv")
y = pd.DataFrame(df.Profit)
x = df.drop('Profit', axis=1)

# v = df.iloc[:, :-1]
# w = df.iloc[:, [-1]]

ohc = OHE(drop='first')
ct = CT(transformers=[('encoder', ohc, ['State'])],remainder='passthrough')

dum_1 = pd.get_dummies(df, prefix={"State":"State"}, drop_first=True)
dum_2 = ct.fit_transform(x)


x_tr, x_te, y_tr, y_te = TTS(dum_2, y, test_size=0.3, random_state=0)

scaler_x = SS()
x_tr = scaler_x.fit_transform(x_tr)
x_te = scaler_x.transform(x_te)
scaler_y = SS()
y_tr = scaler_y.fit_transform(y_tr)

model=LR()
model.fit(x_tr, y_tr)

y_pred=scaler_y.inverse_transform(model.predict(x_te))

r2 = r2_score(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
mse = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)

print(f"r2:{r2}\n\
mae: {mae}\n\
rmse: {rmse}")

# # X = pd.get_dummies(X, drop_first=True)
# X_org = X
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])],
#                        remainder='passthrough')
# X = ct.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
#                                                     random_state=0)

# # skaalataan data
# scaler_x = StandardScaler()
# X_train = scaler_x.fit_transform(X_train)
# X_test = scaler_x.transform(X_test)
# scaler_y = StandardScaler()
# y_train  = scaler_y.fit_transform(y_train)

# # mallin opetus
# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = scaler_y.inverse_transform(model.predict(X_test))

# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print (f'r2: {r2}')
# print (f'mae: {mae}')
# print (f'rmse: {rmse}')

# # tallentaan malli levylle
with open('startup-model.pickle', 'wb') as f:
     pickle.dump(model, f)
    
# tallennetaan encoderi
with open('startup-ct.pickle', 'wb') as f:
     pickle.dump(ct, f)
    
# tallennetaan skaaleri x
with open('startup-scaler-x.pickle', 'wb') as f:
        pickle.dump(scaler_x, f)
       
# tallennetaan skaaleri y
with open('startup-scaler-y.pickle', 'wb') as f:
        pickle.dump(scaler_y, f)