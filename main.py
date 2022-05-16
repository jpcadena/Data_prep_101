"""This is the script for DPhi Data Preparation 101
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

matplotlib.use('Qt5Agg')
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
train = pd.read_csv('titanic_data.csv',
                    dtype={'PassengerId': int, 'Survived': int, 'Pclass': int,
                           'Name': str, 'Sex': str, 'Age': float, 'SibSp': int,
                           'Parch': int, 'Ticket': str, 'Fare': float,
                           'Cabin': str, 'Embarked': str})
# EDA
print(train.shape)
print(train.head())
print(train.dtypes)
print(train.info())
print(train.describe())

# Data treatment
missing_values = (train.isnull().sum())
print(missing_values[missing_values > 0])
print(missing_values[missing_values > 0] / train.shape[0] * 100)

print(train['Sex'].value_counts())
print(train['Sex'].value_counts(normalize=True) * 100)
print(train['Embarked'].value_counts())
print(train['Embarked'].value_counts(normalize=True) * 100)
print(train['Survived'].value_counts())
print(train['Survived'].value_counts(normalize=True) * 100)

# Missing data
del train['Name']
del train['Ticket']
del train['PassengerId']
del train['Cabin']
print(train[train['Fare'] == 0].shape)
train = train[train['Fare'] != 0]
print(train.shape)

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mean_imputer = mean_imputer.fit(X=train['Age'].values.reshape(-1, 1))
train['Age'] = mean_imputer.transform(train[['Age']]).ravel()

mode_imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
# strategy = 'constant', fill_value = 'A'
mode_imputer = mode_imputer.fit(train[['Embarked']])
train['Embarked'] = mode_imputer.transform(train[['Embarked']]).ravel()
train.to_csv(path_or_buf='train_imputed.csv', index=False)

# Outlier by Std Deviation
age_mean = train['Age'].mean()
age_std = train['Age'].std()
lower_limit = age_mean + (-3 * age_std)
higher_limit = age_mean + 3 * age_std

print(age_mean)
print(age_std)
print(lower_limit)
print(higher_limit)
filter_outliers_train = train[(train['Age'] < lower_limit) | (train['Age'] >
                                                              higher_limit)]
print(filter_outliers_train)

fare_mean = train['Fare'].mean()
# calculate the standard deviation
fare_std = train['Fare'].std()
# Lower limit threshold is Mean - 3* SD
ll = fare_mean - (3 * fare_std)
# Higher limit threshold is Mean + 3* SD
hh = fare_mean + (3 * fare_std)

filt_outliers_train = train[(train['Fare'] < ll) | (train['Fare'] > hh)]
print(filt_outliers_train.head())


def out_iqr(data, k=1.5, return_thresholds=False):
    """This method calculates the interquartile range"""
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iq_r = q75 - q25
    cut_off = iq_r * k
    lower, upper = q25 - cut_off, q75 + cut_off
    print(lower, upper)
    if return_thresholds:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in data]


# Outlier by IQR
train['outlier_age'] = out_iqr(train['Age'])
print(train[train['outlier_age']].shape)
sns.boxplot(y='Age', data=train, whis=1.5)
plt.figure()

train['outlier_fare'] = out_iqr(train['Fare'])
print(train[train['outlier_fare']].shape)
sns.boxplot(y='Fare', data=train, whis=1.5)
plt.figure()

# Multivariable Outlier
sns.boxplot(x='Pclass', y='Fare', data=train, whis=1.5)
plt.figure()

# Outlier with Top Coding (Bottom/Zero)
q1 = train['Fare'].quantile(0.25)
q3 = train['Fare'].quantile(0.75)
iqr = q3 - q1
WHISKER_WIDTH = 1.5
upper_whisker = q3 + WHISKER_WIDTH * iqr
train.loc[train.Fare > upper_whisker, 'Fare'] = upper_whisker
train.loc[train.Fare < 0, 'Fare'] = 0

print(train['Fare'].min(), train['Fare'].max())

# Outlier binning
age_range = train.Age.max() - train.Age.min()
min_age = int(np.floor(train.Age.min()))
max_age = int(np.ceil(train.Age.max()))
inter_value = int(np.round(age_range / 10))
print(min_age, max_age, inter_value, age_range)

intervals = [i for i in range(min_age, max_age + inter_value, inter_value)]
labels = ['Bin_' + str(i) for i in range(1, len(intervals))]
print(intervals, labels)
train['age_labels'] = pd.cut(x=train.Age, bins=intervals, labels=labels,
                             include_lowest=True)
train['age_interval'] = pd.cut(x=train.Age, bins=intervals,
                               include_lowest=True)
sns.countplot(train.age_labels)

######################################################################
# Fraud Dataset
fraud_data = pd.read_csv('fraud_data.csv')
print(fraud_data.shape)
print(fraud_data.head())
print(fraud_data.dtypes)
print(fraud_data.info())
print(fraud_data.describe())

print(fraud_data.isFraud.value_counts())
print(fraud_data.isFraud.value_counts(normalize=True) * 100)
sns.countplot(fraud_data.isFraud)

# Missing values
print(fraud_data.isnull().sum() / len(fraud_data) * 100)

num_cols = fraud_data.select_dtypes(include=np.number).columns
print(num_cols)
fraud_data[num_cols] = fraud_data[num_cols].fillna(fraud_data[num_cols].mean())
cat_cols = fraud_data.select_dtypes(include='object').columns
fraud_data[cat_cols] = fraud_data[cat_cols].fillna(
    fraud_data[cat_cols].mode().iloc[0])

print(fraud_data.isnull().sum() / len(fraud_data) * 100)

# One hot encoding
print(fraud_data.shape)
fraud_data = pd.get_dummies(fraud_data, columns=cat_cols)
print(fraud_data.shape)
print(fraud_data.head())

# Feature Transformation
X = fraud_data.drop(columns=['isFraud'])
Y = fraud_data.isFraud
scaled_features = StandardScaler().fit_transform(X)
scaled_features = pd.DataFrame(data=scaled_features)
scaled_features.columns = X.columns
print(scaled_features.head())

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=42)
# Class Imbalance
train_data = pd.concat([X_train, Y_train], axis=1)

# separate minority and majority class
not_fraud = train_data[train_data.isFraud == 0]
fraud = train_data[train_data.isFraud == 1]

# Over sampling minority
fraud_upsampled = resample(fraud,
                           replace=True,  # Sample with replacement
                           n_samples=len(not_fraud),  # Match number
                           random_state=27)
upsampled = pd.concat([not_fraud, fraud_upsampled])
print(upsampled.isFraud.value_counts())

# Under sampling majority
not_fraud_downsampled = resample(not_fraud,
                                 replace=False,  # sample without replacement
                                 n_samples=len(fraud),  # match minority n
                                 random_state=27)
downsampled = pd.concat([not_fraud_downsampled, fraud])  # Concatenation
print(downsampled.isFraud.value_counts())

# SMOTE - Synthetic Minority Oversampling Technique
sm = SMOTE(random_state=25, sampling_strategy=1.0)
X_train, Y_train = sm.fit_resample(X_train, Y_train)
print(X_train.head())
print(Y_train.head())
print(Y_train.value_counts())
