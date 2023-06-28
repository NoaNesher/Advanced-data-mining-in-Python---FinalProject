# -*- coding: utf-8 -*-
"""model_training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B0ckCDi2M1KUiMIdzEwVOJCy2w-ztGUc
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
from sklearn.compose import make_column_transformer
from madlanOved import prepare_data

data_for_train = 'https://github.com/NoaNesher/Advanced-data-mining-in-Python---FinalProject/raw/main/output_all_students_Train_v10.xlsx'
data = pd.read_excel(data_for_train)
data_for_test = 'https://github.com/NoaNesher/Advanced-data-mining-in-Python---FinalProject/raw/main/Dataset_for_test.xlsx'
test = pd.read_excel(data_for_test)

df = prepare_data(data)
new_data_for_test = prepare_data(test)

y_train = df['price']
X_train = df.drop('price', axis=1)
X_test = new_data_for_test.drop('price', axis=1)
y_test = new_data_for_test['price']
cols_to_drop = ['City', 'total_floors']
X_train.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)

imputer = SimpleImputer(strategy='most_frequent')
X_train[['Area', 'floor']] = imputer.fit_transform(X_train[['Area', 'floor']])
X_test[['Area', 'floor']] = imputer.transform(X_test[['Area', 'floor']])

numerical_features = ['room_number', 'Area', 'hasElevator', 'hasParking', 'hasBars',
                      'hasStorage', 'hasAirCondition', 'hasBalcony',
                      'handicapFriendly', 'floor', 'Index_value']

"""
We performed a VIF test on the numerical columns in order to check which of the features might cause us
to reach multicollinearity and overfit.
The test selects the VIF index of each feature, and if there are features with a VIF index>5
We will download the feature with the maximum VIF, and perform the test again without the column.
until we reach a situation where all the features have a VIF index less than 5

"""

vif_data = pd.DataFrame()
vif_data["Variable"] = numerical_features
vif_data["VIF"] = [variance_inflation_factor(X_train[numerical_features].values.astype(np.float64), i)
                   for i in range(len(numerical_features))]

while vif_data['VIF'].max() > 5:
    high_vif_columns = vif_data['VIF'].idxmax()
    vif_data = vif_data.drop(high_vif_columns, axis=0)

    remaining_columns = vif_data['Variable'].tolist()
    X_train_remaining = X_train[remaining_columns]

    vif_data['VIF'] = [variance_inflation_factor(X_train_remaining.values.astype(np.float64), i)
                       for i in range(len(remaining_columns))]

num_cols = vif_data['Variable'].tolist()
cat_cols = [col for col in X_train.columns if (X_train[col].dtypes == 'O' or (X_train[col].dtypes == 'object'))]
X_train = X_train[num_cols + cat_cols]
X_test = X_test[num_cols + cat_cols]

numerical_pipeline = Pipeline([('scaling', StandardScaler())])

categorical_pipeline = Pipeline([('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'))])

preprocessing_pipeline = make_column_transformer(
    (numerical_pipeline, num_cols),
    (categorical_pipeline, cat_cols),
    remainder='drop')

elastic_net_pipeline = Pipeline(
    [('preprocessing', preprocessing_pipeline), ('elastic_net', ElasticNet(alpha=10, l1_ratio=0.4, random_state=42))])

cross_val = KFold(n_splits=10)

scores = cross_val_score(elastic_net_pipeline, X_train, y_train, cv=cross_val, scoring='neg_mean_squared_error')

print("Cross-validation scores:", scores)
print("Average MSE:", np.mean(-scores))
rmse_scores = np.sqrt(-scores)
print("Average RMSE:", np.mean(rmse_scores))

elastic_net_pipeline.fit(X_train, y_train)

y_pred = elastic_net_pipeline.predict(X_test)

# _______________________________________________________________________
joblib.dump(elastic_net_pipeline, 'trained_model.pkl')
