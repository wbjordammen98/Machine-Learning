from sklearn.datasets import fetch_california_housing

cali = fetch_california_housing()

#print(cali.DESCR)
#print(cali.data.shape)
print(cali.target_names)
print(cali.feature_names)

import pandas as pd

pd.set_option("precision",4)
pd.set_option("max_columns",9)
pd.set_option("display.width",None)

cali_df = pd.DataFrame(cali.data,columns=cali.feature_names)
cali_df["MedHouseVal"] = pd.Series(cali.target)

print(cali_df.head())

sample_df = cali_df.sample(frac=0.1, random_state=17)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1)
sns.set_style("whitegrid")

grid = sns.pairplot(
        data=cali_df,
        vars=cali_df.columns[0:4]
    )
    
plt.show()