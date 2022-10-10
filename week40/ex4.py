import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split as TTS
from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler as SS
import pickle

df = pd.read_csv('housing.csv')


df = df.drop('population', axis=1)
df = df.drop('households', axis=1)

#drop NaN rows
df = df.dropna()

#ohc = OHE(drop='first')
#ct = CT(transformers=[('encoder', ohc, ['ocean_proximity'])],remainder='passthrough')

y = pd.DataFrame(df.median_house_value)
#x = ct.fit_transform(df.drop('median_house_value', axis=1))
x = pd.get_dummies(df.drop('median_house_value', axis=1), prefix={"ocean_proximity":"ocean_proximity"}, drop_first=True)

x_tr, x_te, y_tr, y_te = TTS(x, y, test_size=0.3, random_state=0)

scaler_x = SS()
x_tr = scaler_x.fit_transform(x_tr)
x_te = scaler_x.transform(x_te)
scaler_y = SS()
y_tr = scaler_y.fit_transform(y_tr)

model = LR()
model.fit(x_tr, y_tr)

y_pred = scaler_y.inverse_transform(model.predict(x_te))

r2 = r2_score(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
mse = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)

# quite big errors due to all >500'000 houses marked as 500'001, data collection failure
print(f"r2:{r2}\n\
mae: {mae}\n\
rmse: {rmse}")


with open('house-model.pickle', 'wb') as f:
         pickle.dump(model, f)
#with open('house-ct.pickle', 'wb') as f:
#         pickle.dump(ct, f)
with open('house-scaler-x.pickle', 'wb') as f:
        pickle.dump(scaler_x, f)
with open('house-scaler-y.pickle', 'wb') as f:
        pickle.dump(scaler_y, f)
