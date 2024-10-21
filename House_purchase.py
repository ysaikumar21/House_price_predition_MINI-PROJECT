'''
In this we are going to develop the Multiple Linear Regression using Oops concet for HousePurchasing_data set and predict the prize and calculate accuracy and loss
'''
import numpy as np
import pandas as pd
import sklearn
import sys
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
class HousePurchase:
  def __init__(self,location):
    try:
      self.location=location.drop(['id','date','zipcode','lat','long'],axis=1)#these 5 columns are not more impartence so we drop the columns
      self.X=self.location.iloc[: ,1:]
      self.y=self.location.iloc[: ,0]
      self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
    except FileNotFoundError:
      print("Error: File not found.")
    except Exception as e:
      error_type, error_msg, err_line = sys.exc_info()
      print(f"Error from Line {err_line.tblineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")
  def MultiLinear_Regression(self):
    try:
      self.reg=LinearRegression()
      self.reg.fit(self.X_train,self.y_train)
      self.y_train_pred=self.reg.predict(self.X_train)
      self.y_test_pred=self.reg.predict(self.X_test)
      print(f'Train Accuracy with r2_score fun is :{r2_score(self.y_train,self.y_train_pred)}')
      print(f'Train Loss with MSE fun is : {mean_squared_error(self.y_train,self.y_train_pred)}')
      print(f'Test Accuracy with functions is : {r2_score(self.y_test,self.y_test_pred)}')
      print(f'Test Loss is with functions is : {mean_squared_error(self.y_test,self.y_test_pred)}')
    except Exception as e:
      error_type, error_msg, err_line = sys.exc_info()
      print(f"Error from Line {err_line.tblineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")
  def Ridge_Regression(self,alpha):
    try:
      self.ridge_reg = Ridge(alpha=alpha)  # Create new Ridge object
      self.ridge_reg.fit(self.X_train, self.y_train)
      self.y_train_pred = self.ridge_reg.predict(self.X_train)  # Use ridge_reg
      self.y_test_pred = self.ridge_reg.predict(self.X_test)  # Use ridge_reg
      print(f'Train Accuracy with r2_score fun is :{r2_score(self.y_train, self.y_train_pred)}')
      print(f'Train Loss with MSE fun is : {mean_squared_error(self.y_train, self.y_train_pred)}')
      print(f'Test Accuracy with functions is : {r2_score(self.y_test, self.y_test_pred)}')
      print(f'Test Loss is with functions is : {mean_squared_error(self.y_test, self.y_test_pred)}')
    except Exception as e:
      error_type, error_msg, err_line = sys.exc_info()
      print(f"Error from Line {err_line.tblineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")
  def Lasso_Regression(self,alpha):
    try:
      self.lasso_reg=Lasso(alpha)
      self.lasso_reg.fit(self.X_train,self.y_train)
      self.y_train_pred=self.lasso_reg.predict(self.X_train)
      self.y_test_pred=self.lasso_reg.predict(self.X_test)
      print(f'Train Accuracy with r2_score fun is :{r2_score(self.y_train,self.y_train_pred)}')
      print(f'Train Loss with MSE fun is : {mean_squared_error(self.y_train,self.y_train_pred)}')
      print(f'Test Accuracy with functions is : {r2_score(self.y_test,self.y_test_pred)}')
      print(f'Test Loss is with functions is : {mean_squared_error(self.y_test,self.y_test_pred)}')
    except Exception as e:
      error_type, error_msg, err_line = sys.exc_info()
      print(f"Error from Line {err_line.tblineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")
if __name__=='__main__':
  try:
    obj=HousePurchase(location = pd.read_csv('C:\\Users\\Saiku\\Downloads\\House_Price_prediction\\kc_house_data.csv'))
    print(f'MultiLinear Regression :')
    print('-------------------')
    obj.MultiLinear_Regression()
    print()
    print(f'Ridge Regression :')
    print('-------------------')
    obj.Ridge_Regression(0.5)
    print()
    print(f'Lasso Regression :')
    print('-------------------')
    obj.Lasso_Regression(0.5)
    print()
  except Exception as e:
    error_type, error_msg, err_line = sys.exc_info()
    print(f"Error from Line {err_line.tblineno} -> type {error_type.__name__} -> Error msg -> {error_msg}")