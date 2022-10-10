import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('startup.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, [-1]]

# X = pd.get_dummies(X, drop_first=True)
X_org = X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])],
                       remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

# skaalataan data
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
scaler_y = StandardScaler()
y_train  = scaler_y.fit_transform(y_train)

# mallin opetus
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = scaler_y.inverse_transform(model.predict(X_test))

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print (f'r2: {r2}')
print (f'mae: {mae}')
print (f'rmse: {rmse}')

# tallentaan malli levylle
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
      
        
      
        
      
        Xnew = scaler_x.transform(Xnew)
        ynew = scaler_y.inverse_transform(model.predict(Xnew))
        for i in range (len(ynew)):
            print(f'Xnew_org.iloc[i]\nVoitto: {ynew[i][0]\n')