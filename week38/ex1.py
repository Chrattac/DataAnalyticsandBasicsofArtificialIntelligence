import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_excel("tt.xlsx", index_col=0, header=0)

df.hist()


df.koulutus = df.koulutus.replace(to_replace=[1, 2, 3, 4], 
                                  value=['Peruskoulu', '2. Aste', 
                                         'Korkeakoulu', 'Ylempi korkeakoulu'])

freq = pd.crosstab(df.koulutus, df.koulutus.count())
freq = pd.merge(freq, pd.crosstab(df.koulutus, df.koulutus.count(), normalize=True), how='inner', on='koulutus')
freq = freq.rename(columns={'81_x':'Lukumäärä','81_y':'Prosentti'})
freq.Prosentti = freq.Prosentti.mul(100)

plt.barh(freq.index, freq.Lukumäärä)
plt.show()