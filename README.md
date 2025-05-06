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
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/66aaab1b-9206-4571-b228-cfbf7e526db3)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/dd2e7da7-b08d-48d6-a8ef-ebcb13d5e450)
```
df['bo2']=e1.fit_transform(df[['ord_2']])
df
```
![image](https://github.com/user-attachments/assets/8e9ced52-fdb1-43a1-8eac-acae64ba4f99)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/3154d145-877b-4400-a0ca-d7e5b9dd07c8)
```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/d9f15c7a-fb0d-42e9-a7f6-3e1088b8652e)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/bb5ffae5-3ac5-4af2-b40b-eb73cb5c9224)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/617d3f6a-4e8d-48cb-9c58-6c2c18e59af0)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/014c1f1d-0dd3-4bc4-a918-3ff011c96266)
```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/4ea873b6-ba50-4e1c-9622-1cea6e1a9f60)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/5a406dff-04a9-444f-9d63-395c2f7ad136)
```
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/6557f7e8-e7e4-460f-a44b-827d3698869b)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/c6296433-4182-4ff3-ba30-e4ee74bb4754)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/3bc020f7-5655-439e-9eb3-652ce5f9480a)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ca71e76c-9795-46f8-b117-75d777392295)
```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/ca557775-eb2b-4e99-965c-339e7d0511ac)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/d0697d47-bb9a-4a87-a629-39ac98bcfb21)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/86307104-4091-420e-b3fc-f48783d2b42e)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/ce4c5846-13bd-4837-95d7-365e995e1756)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/95b9a0c0-9756-4238-9725-7fd61aceeb22)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b718a981-94ea-460d-b1d3-a1ffc6e109fe)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/eb1f28b0-b1eb-46a6-bafc-498cd22289e4)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f2344207-1497-48fd-8e3e-fb30af2625d6)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/27591136-d797-40bc-9078-aabd413af4cc)
```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/6f5b9ac7-d56f-46c6-b442-c96f6d9193a2)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/user-attachments/assets/0c9cbb62-bbb3-4d51-85d8-efc9ccee314b)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9ad8670b-4cd8-4cba-adc8-90b1b17bc4e3)



























       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
       
# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       

       
