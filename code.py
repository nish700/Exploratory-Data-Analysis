# --------------
# Code starts here
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

### Data 1
# Load the data
data = pd.read_csv(path)

### Overview of the data
data.info()

data.describe()

### Histogram showing distribution of car prices
# plt.figure()
sns.distplot(data['price'],kde=True,rug=True)

### Countplot of the make column
# data['make'].value_counts()

plt.figure()
sns.countplot(x='make', data=data )
plt.xticks(rotation=90)

### Jointplot showing relationship between 'horsepower' and 'price' of the car
plt.figure()
sns.jointplot(x='horsepower',y='price',data=data, kind='reg')

###Correlation heat map
plt.figure()
sns.heatmap(data.corr(),cmap='YlGnBu')

###boxplot that shows the variability of each 'body-style' with respect to the 'price'
plt.figure()
sns.boxplot(x='body-style',y='price',data=data)

#### Data 2

# Load the data
data_2 = pd.read_csv(path2)

# Impute missing values with mean
# replacing the special characters with null to be identifiable
data_2.replace('?',np.NaN, inplace=True)

#checking the null values
print(data_2.isna().sum())

# using the Imputer function to fit and transform the data with the mean 

imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
imputer = imputer.fit(data_2[['normalized-losses','horsepower']])
data_2[['normalized-losses','horsepower']] = imputer.transform(data_2[['normalized-losses','horsepower']])

# verifying the presence of NaN values
# print(data_2.isna().sum())


# Checking the Skewness of numeric features
numeric_columns = data_2._get_numeric_data().columns

# print(numeric_columns)

# Scaling the numerical features
for i in numeric_columns:
    print(skew(data_2[i]))
    if skew(data_2[i])>1:
        data_2[i] = np.sqrt(data_2[i])


# Identify the categorical features
cat_columns = list(data_2.select_dtypes(include=['object']).columns)

# Label encode the categorical features
le = LabelEncoder()
for col in cat_columns:
    data_2[col]=  le.fit_transform(data_2[col])

## Alternativly fit transform can be carried out as below as well
# d = defaultdict(LabelEncoder)
# data_2[cat_columns] = data_2[cat_columns].apply(lambda x: d[x.name].fit_transform(x))

# feature engineering , combine height and width to make new featur area
data_2['area'] = data_2['height'] * data_2['width']

print(data_2.head())

# Code ends here


