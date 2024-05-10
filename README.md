## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/aae763cd-3117-4c8e-bf2c-c652a5541541)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/b8e416e0-1631-4afc-a303-fba2a3a1febd)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/2cbc1dc1-5234-45a6-a925-840d7e0baee2)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/12ce4365-d23e-4986-941c-b26b61a6bdd7)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/d6b8bdc9-91e6-41d6-b852-e7c343cc096a)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/df24a068-454c-426e-b5c6-77eeb69e719b)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/54fdc26f-c5e7-4026-a48d-707aee14faab)

```
pip install --upgrade category_encoders
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/9deca6ac-1360-4800-93df-d673c5ccd415)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/4c46dc34-429c-489c-bf3a-ce2c62d0fe24)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/7e5f8d02-39f8-429b-a1a6-8223f63cd417)
```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/24b9257d-f1a0-4284-9cfe-bcebe6f13261)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/a7944f5e-dac7-41e6-a061-f1326ce96f1c)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/29e2129f-10e1-448e-9741-1580cbf49b64)

```
df.skew()
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/bdc4afd2-b0f2-4306-b4c2-66b1b369c7f7)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/b78a4f39-f466-46d9-a565-f0e8a8e1a891)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/2b4156b5-dca8-4763-b2d1-384242df4b23)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/f1d49252-4997-4dc3-b905-438a91af37a8)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/01c8afd9-3287-4c58-977f-c43db1aba7ff)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/1f345c86-c0f1-4a7c-99c7-9d7f7ee24ce8)

```
df.skew()
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/a1fcc12d-2f68-4951-b8ba-ce5a13544d23)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/d84c61ec-dd51-45ae-a2d1-0b7272d8e630)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/09a83e2b-950f-463f-9e97-bea2426780be)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/9f8a1915-eeba-4959-a64f-2e469c1a6894)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/e3338bb1-b492-4cd5-b32b-02e2430bc9da)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/b732faef-2b5a-4265-a643-f86e382bbc04)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/svarsha220/EXNO-3-DS/assets/127709117/3f12d222-7685-4aa9-8288-c1717a69c014)


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
      
