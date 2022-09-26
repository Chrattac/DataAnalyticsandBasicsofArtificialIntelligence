import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_excel("tt.xlsx", index_col=0, header=0)
df.koulutus = df.koulutus.replace(to_replace=[1, 2, 3, 4], 
                                  value=['Peruskoulu', '2. Aste', 
                                         'Korkeakoulu', 'Ylempi korkeakoulu'])
df.sukup = df.sukup.replace(to_replace=[1, 2], 
                                    value = ['Mies', 'Nainen'])

table = pd.crosstab(df.koulutus, df.sukup)