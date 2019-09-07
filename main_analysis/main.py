import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
from math import sqrt
from itertools import combinations
from dm_test import dm_test

# By running functions below, the results of the analysis can be fully reproduced
df_kernel_adl = pd.read_csv('adl_after_kernel_pca.csv', header=0)
df_kernel_adl = df_kernel_adl.dropna()

def get_y_and_date(df_param):
  df_param = df_param.drop([0], axis=0)
  df_param = df_param.dropna()
  return df_param['date'], df_param['consumption']

def get_X_consumption_lag_only(df_param):
  df_param = df_param[['consumption_l1', 'income_l1', 'smp500', 'tbill']]
  df_param = df_param.dropna()
  return df_param

def evaluate_model_performance(true_consumption, predictions):
  from sklearn.metrics import mean_squared_error
  mean_squared_error = mean_squared_error(true_consumption, predictions)
  root_mean_squared_error = sqrt(mean_squared_error)
  print(root_mean_squared_error) 
  return root_mean_squared_error

def generate_lag_consumption_and_composite_X_baseline(df_param):
  print(df_param.head())
  consumption = df_param['consumption']
  lagged_consumption = consumption.shift(1)
  income = df_param['income']
  lagged_income = income.shift(1)
  df_param['consumption_l1'] = lagged_consumption
  df_param['income_l1'] = lagged_income
  df_param = df_param.dropna()
  del df_param['date']
  del df_param['income']
  del df_param['consumption']
  df_param = df_param.dropna()
  return df_param

def nowcaster(X, y, date):
  tscv = TimeSeriesSplit(n_splits=100)
  predictions = []
  true_consumption = []
  dates = []

  for train_index, test_index in tscv.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    
    X_train, X_test = X[:len(train_index) - 1], X[len(train_index) - 1: len(train_index)]
    y_train, y_test = y[:len(train_index) - 1], y[len(train_index) - 1: len(train_index)]
    date_train, date_test = date[:len(train_index)], date[len(train_index) - 1: len(train_index)]

    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train) # training is conducted on the sample before t
    # print(clf.coef_)
    # print(X_test.head(1))
    # print(y_test.head(1))
    prediction = clf.predict(X_test) # nowcast and forecast are distinguished by having different X sets, the caster is the same!!!
    # print(prediction.item(0))
    # print(y_test.values[0])
    predictions.append(prediction.item(0))
    true_consumption.append(y_test.values[0])
    # print("THE TRUE CONS")
    # print(y_test.values[0])
    # print(true_consumption)
    dates.append(date_test.values[0])
    # print(y_test.values[0])
    # print(prediction.item(0))
  return true_consumption, predictions, dates

def plot_results(dates, true_consumption, predictions):
  print("Plotting")
  fig = plt.figure()                                                               
  ax = fig.add_subplot(1,1,1)
  ax.plot(dates, true_consumption, color='red')
  ax.plot(dates, predictions, color='blue')
  tick_spacing = 30
  ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing)) 
  ax.legend(['True consumption yoy growth rate', 'GT indicator nowcast']) 
  plt.title('GT indicator nowcast')
  plt.show()

# Procedure use to select KPCA components that bring value to the nowcasting performance
def recursive_kpca_iterator(X_used, X_unused, y, date, df_param, root_mean_sq_error_param, X_full):
  print("Starting the recursive iterative procedure")
  root_mean_sq_error = root_mean_sq_error_param # as a reference point
  useful_x = ''

  if len(X_unused) == 0:
    print("X_unused is 0")
    print(X_used)
    return X_used

  for x in X_unused:
    combo = X_used
    combo.append(x)
    # print(combo)
    X = df_param[combo]
    X = X.dropna()
    
    true_consumption, predictions, dates = nowcaster(X, y, date)

    if root_mean_sq_error > evaluate_model_performance(true_consumption, predictions):
      root_mean_sq_error = evaluate_model_performance(true_consumption, predictions)
      useful_x = x
    combo.remove(x)

  if root_mean_sq_error < root_mean_sq_error_param:
    print(useful_x)
    X_used_new = X_used.copy()
    X_used_new.append(useful_x)
    X_unused_new = X_unused.copy()
    X_unused_new.remove(useful_x)
    print(X_used_new) 
    print(X_unused_new)
    print(root_mean_sq_error)
    recursive_kpca_iterator(X_used_new, X_unused_new, y, date, df_param, root_mean_sq_error, X_full)
  else:
    print("The error term did not improve!")
    print(X_used)
    print(root_mean_sq_error)
    return X_used

def perform_michigan_nowcast(df_param):
  date, y = get_y_and_date(df_param)
  X_baseline = generate_lag_consumption_and_composite_X_baseline(df_param=df_param)
  X = X_baseline[['smp500', 'tbill', 'consumption_l1', 'income_l1', 'michigan_index']]
  X = X.dropna()
  true_consumption, predictions, dates = nowcaster(X, y, date)
  print(evaluate_model_performance(true_consumption, predictions))
  return true_consumption, predictions

def perform_cci_nowcast(df_param):
  date, y = get_y_and_date(df_param)
  X_baseline = generate_lag_consumption_and_composite_X_baseline(df_param=df_param)
  X = X_baseline[['smp500', 'tbill', 'consumption_l1', 'income_l1', 'cci']]
  X = X.dropna()
  true_consumption, predictions, dates = nowcaster(X, y, date)
  print(evaluate_model_performance(true_consumption, predictions))
  return true_consumption, predictions

def perform_google_nowcast(df_param):
  date, y = get_y_and_date(df_param)
  X_baseline = generate_lag_consumption_and_composite_X_baseline(df_param=df_param)
  X = X_baseline[['smp500', 'tbill', 'consumption_l1', 'income_l1', '131', '90', '104', '165', '109', '8', '68', '5', '23', '85', '34', '149', '28', '126', '44', '39', '49', '14', '80', '59', '139', '151', '18', '43', '7']]
  X = X.dropna()
  true_consumption, predictions, dates = nowcaster(X, y, date)
  print(evaluate_model_performance(true_consumption, predictions))  
  return true_consumption, predictions

# performing D-M test
def perform_dm_test(df_param):
  date, y = get_y_and_date(df_param) 
  true_consumption, google_predictions = perform_google_nowcast(df_param)
  true_consumption, cci_predictions = perform_cci_nowcast(df_param)
  true_consumption, michigan_predictions = perform_michigan_nowcast(df_param)
  dm_test_results = dm_test(true_consumption,google_predictions,michigan_predictions,crit="MSE")
  print(dm_test_results)

# Performing the procedure that results in best KPCA components 25 out of 165

'''
These are the most valuable components
'131', '90', '104', '165', '109', '8', '68', '5', '23', '85', '34', '149', '28', '126', '44', '39', '49', '14', '80', '59', '139', '151', '18', '43', '7' 
'''
def perform_recursive_kpca_iterator(df_param):
  date, y = get_y_and_date(df_param)
  X_baseline = generate_lag_consumption_and_composite_X_baseline(df_param)
  X_baseline = X_baseline.dropna()

  X = X_baseline.copy()
  del X['michigan_index']
  del X['cci']

  X_columns = X.columns
  X_used = X_columns[-4:]
  X_unused = X_columns[:-4]
  print(X_used)
  print(X_unused)

  recursive_kpca_iterator(list(X_used), list(X_unused), y, date, df_param, 10000.0, list(X_columns)) # instead of saving, just print the result

  # At the end of the procedure the best performing KPCA components will be printed with the RMSFE of the best GT-based model


















