import pandas as pd
import pickle  

with open('house-model.pickle', 'rb') as f: model = pickle.load(f)
#with open('house-ct.pickle', 'rb') as f: ct = pickle.load(f)
with open('house-scaler-x.pickle', 'rb') as f: scaler_x = pickle.load(f)
with open('house-scaler-y.pickle', 'rb') as f: scaler_y = pickle.load(f)

df_n = pd.read_csv('new_house_ct.csv')
new_houses = pd.get_dummies(df_n, prefix={"ocean_proximity":"ocean_proximity"}, drop_first=False)

#Bad approach, using ct gave for some reason mystery errors...
new_houses[['ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR OCEAN']] = [0,0,0]

new_houses = new_houses[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                         'total_bedrooms', 'median_income', 
                         'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
                         'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']]

new_houses = scaler_x.transform(new_houses)
new_price = scaler_y.inverse_transform(model.predict(new_houses))



for i in range (len(new_houses)):
        print(f'{df_n.iloc[i]}\nPrice: {new_price[i][0]}\n')
        
coef = model.coef_
inter = model.intercept_