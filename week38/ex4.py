import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

df = pd.read_excel("tt.xlsx", index_col=0, header=0)
df_corr = df[['sukup', 'ikä', 'perhe', 'koulutus', 'palkka']]
df_corr.corr()
heatmap = sns.heatmap(df_corr.corr(),annot=True,linewidths=.5)
plt.show()

pearson = stats.pearsonr(df_corr.palkka, df_corr.ikä)
spearman = stats.spearmanr(df_corr.palkka, df_corr.ikä)

regression = sns.regplot(x=df_corr.palkka, y=df_corr.ikä, dropna=True)
plt.show()