# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:30:19 2021

@author: sbava
"""



import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.sklearn
    
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


warnings.filterwarnings("ignore")
np.random.seed(40)

   
     # Split the data into training and test sets. (0.75, 0.25) split.
    #train, test = train_test_split(data) 
    #defining the filepath to read tha data
dir_path='C:/Users/sbava/cmapss/'
index_names=['unit_nr','time_cycles']
setting_names=['setting_1','setting_2','setting_3']
sensor_names=['s_{}'.format(i) for i in range(1,22)]
col_names= index_names+setting_names+sensor_names
#define column names for easy indexes
train=pd.read_csv((dir_path+'train_FD001.txt'),sep='\s+',header=None,names=col_names)
test=pd.read_csv((dir_path+'test_FD001.txt'),sep='\s+',header=None,names=col_names)
y_test=pd.read_csv((dir_path+'RUL_FD001.txt'),sep='\s+',header=None,names=['RUL'])
def add_remaining_useful_life(df):
    grouped_by_unit=df.groupby(by="unit_nr")
    max_cycle=grouped_by_unit["time_cycles"].max()
    result_frame=df.merge(max_cycle.to_frame(name='max_cycle'),left_on='unit_nr',right_index=True)
    remaining_useful_life=result_frame["max_cycle"]-result_frame["time_cycles"]
    result_frame["RUL"]=remaining_useful_life
    result_frame=result_frame.drop("max_cycle",axis=1)
    return result_frame
train=add_remaining_useful_life(train)
train[index_names+['RUL']].head()

    # The predicted column is "quality" which is a scalar from [3, 9]
drop_sensors = ['s_1','s_5','s_6','s_10','s_16','s_18','s_19']
drop_labels = index_names+setting_names+drop_sensors

X_train = train.drop(drop_labels, axis=1)
y_train = X_train.pop('RUL')
y_train_float = y_train.astype(float)
X_test = test.groupby('unit_nr').last().reset_index().drop(drop_labels, axis=1)
    

    # Set default values if no alpha is provided
    #if float(in_alpha) is None:
        #alpha = 0.5
   # else:
       # alpha = float(in_alpha)

    # Set default values if no l1_ratio is provided
    #if float(in_l1_ratio) is None:
        #l1_ratio = 0.5
    #else:
      #  l1_ratio = float(in_l1_ratio)

    # Useful for multiple runs (only doing one run in this sample notebook)    
with mlflow.start_run():
        # Execute lr
    lr = LinearRegression()
    lr.fit(X_train,y_train_float)

        # Evaluate Metrics
    predicted_rul = lr.predict(X_test)
    (rmse, mae, r2) = eval_metrics(y_test, predicted_rul)

        # Print out metrics
       # print("linearregression model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
       # mlflow.log_param("alpha", alpha)
        #mlflow.log_param("l1_ratio", l1_ratio)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)

mlflow.sklearn.log_model(lr, "model")