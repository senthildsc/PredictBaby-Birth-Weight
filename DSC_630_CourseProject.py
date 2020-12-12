'''
File              : DSC_630_CourseProject.py
Name              : Senthilraj Srirangan
Date              : 05/20/2020
Assignment Number : 12.2 Course project : Milestone5--Final Project Paper.
Course            : DSC 630 - Predictive Analytics
Exercise Details  :
Course Presentation
Course Final paper.
'''

import calendar
from itertools import groupby

import pandas as pd
import numpy as np
import pydotplus as pydotplus
import seaborn as sns
import string
import re
import matplotlib.pyplot as plt
from collections import Counter
import json
import sys
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model, tree
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn import metrics
from IPython.display import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import os
from google.cloud import bigquery

#keyfile = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
keyfile = 'C:\Python-Projects\senthildsc-1146c60b4ab5.json'
client = bigquery.Client.from_service_account_json(keyfile)

QUERY = (
    """
    SELECT * FROM `bigquery-public-data.samples.natality`   
    WHERE  year > 2000
    AND weight_pounds IS NOT NULL
    AND RAND() < 0.01    
    """
)

pd.options.display.max_columns = None
pd.options.display.max_rows = None

data = (
    client.query(QUERY)
        .result()
        .to_dataframe()
)


# 1. Data Exploration

dataframe = data.corr(min_periods=50)
dataframe.sort_values(by=["weight_pounds"], ascending=False, inplace=True)

d = data.groupby(['is_male']).count().reset_index().groupby('weight_pounds').mean()

gestation_weeks_vs_other = data.groupby(["gestation_weeks"]).mean()[["is_male","plurality"]]
gestation_weeks_vs_other.plot.line()
plt.show()

#Plotting the Least Squares Line

sns.pairplot(data, x_vars=['is_male','gestation_weeks','mother_age','mother_race'], y_vars='weight_pounds', size=7, aspect=0.7, kind='reg')
plt.show()

sns.pairplot(data[["weight_pounds","gestation_weeks","mother_age","weight_pounds"]].dropna())
plt.show()

hm_data = data[["weight_pounds","gestation_weeks","mother_age","mother_race","father_race","father_age","cigarettes_per_day"]].dropna()
sns.heatmap(hm_data.corr(), square=True, cmap='RdYlGn')
plt.show()

def get_distinct_values(column_name):
    sql = """
SELECT
  {0},
  COUNT(1) AS num_babies,
  AVG(weight_pounds) AS avg_wt
FROM
  publicdata.samples.natality
WHERE
  year > 2000
GROUP BY
  {0}
    """.format(column_name)
    return client.query(sql).result().to_dataframe()

# unique values for each of the columns and the count of those values.

# To see Number of babies based on Gender

df = get_distinct_values('is_male')
df.plot(x='is_male', y='num_babies', kind='bar')
df.plot(x='is_male', y='avg_wt', kind='bar')
plt.show()

# To see number of babies based on Mother's Age

df = get_distinct_values('mother_age')
df = df.sort_values('mother_age')
df.plot(x='mother_age', y='num_babies')
df.plot(x='mother_age', y='avg_wt')
plt.show()

# To see number of babies based on Plurality

df = get_distinct_values('plurality')
df = df.sort_values('plurality')
df.plot(x='plurality', y='num_babies', logy=True, kind='bar')
df.plot(x='plurality', y='avg_wt', kind='bar')
plt.show()

# To see number of babies based on Gestation Weeks

df = get_distinct_values('gestation_weeks')
df = df.sort_values('gestation_weeks')
df.plot(x='gestation_weeks', y='num_babies', logy=True, kind='bar', color='royalblue')
df.plot(x='gestation_weeks', y='avg_wt', kind='bar', color='royalblue')
plt.show()


# 2. Data Preparation

data['is_male'] = data[['is_male']].replace({'is_male': {1: 1, 0: 0}})

df = data[['weight_pounds','is_male', 'gestation_weeks', 'mother_age', 'mother_race']].dropna()


# Seperate the independent and target variable on training data
# Feature Selection
train_x = df[['is_male', 'gestation_weeks', 'mother_age', 'mother_race']]
train_y = df['weight_pounds']   # Target Variable


# Split the Data to Train and Test

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.20,random_state=0)

# number of samples in each set
print("No. of samples in training set: ", X_train.shape[0])
print("No. of samples in validation set:", X_test.shape[0])

print('Statistics on Training Set : \n',X_train.describe())
############################# 1. create linear regression object ################################################
# Create a model and fit

regressor = LinearRegression()

# train the model using the training sets
regressor.fit(X_train, y_train)

# Making Predictions
y_pred = regressor.predict(X_test)


# Predict the Model by running with values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head(20),'\n')

# Model Evaluation - Evaluating the Algorithm

print('Regression Model Coefficients: ', regressor.coef_)           # regression coefficients
print('Regression Model Intercept',regressor.intercept_)            # variance score: 1 means perfect prediction
print('Regression Model Score : ',regressor.score(X_test, y_test),'\n')  # Get score

# Finding the values for MAE, MSE and RMSE
print('LINEAR REGRESSION EVALUATION METRICS : \n')
print('Coefficient of determination r2_Score: %.2f' % r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


####################################### 2. Decision Tree Regression ######################################

# Create decision tree classifier object
decision_tree = DecisionTreeRegressor(random_state=0)

# Train model
decision_tree_model = decision_tree.fit(X_train, y_train)
y_predict_dtr = decision_tree_model.predict(X_test)

# Model Evaluation for Decision tree
print('DECISION TREE EVALUATION METRICS : \n')
r_square = metrics.r2_score(y_test,y_predict_dtr)
print('Decision Tree Model R-Square : ',r_square)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict_dtr))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_dtr))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_dtr)))

# Predict the Model by running with values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict_dtr})
print(df.head(20),'\n')

################### 3. Training a Random Forest Regressor #############################################

# Create random forest classifier object
randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)

# Train model
randomforest.fit(X_train, y_train)
y_pred_rf = randomforest.predict(X_test)

print('RANDOM FOREST REGRESSOR EVALUATION METRICS : \n')
r_square = metrics.r2_score(y_test,y_pred_rf)
print('Random Forest Regressor R-Square : ',r_square)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))

# Predict the Model by running with values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
print(df.head(20),'\n')