import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score

# loading the preprocessed data
data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', "src", "preprocessing"))
data_file = os.path.join(data_directory, 'Housing_preprocessed.csv')
dataset = pd.read_csv(data_file)

# separating features and target
x=dataset.drop(columns=['price'])
y=dataset['price']

def train_model():
    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Linear Regression
    regressor_lr = LinearRegression()
    regressor_lr.fit(x_train,y_train)

    y_pred_lr = regressor_lr.predict(x_test)

    mse = mean_squared_error(y_test, y_pred_lr) 
    print("LR Mean Squared Error (MSE):", mse)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr)) 
    print("LR Root Mean Squared Error (RMSE):", rmse)
    r2 = r2_score(y_test, y_pred_lr) 
    print("LR R-squared (R²):", r2)

    # Random Forest
    regressor_rf = RandomForestRegressor()
    regressor_rf.fit(x_train,y_train)

    y_pred_rf = regressor_rf.predict(x_test)

    mse = mean_squared_error(y_test, y_pred_rf) 
    print("Random Forest Mean Squared Error (MSE):", mse)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf)) 
    print("Random Forest Root Mean Squared Error (RMSE):", rmse)
    r2 = r2_score(y_test, y_pred_rf) 
    print("Random Forest R-squared (R²):", r2)

    # K-Nearest Neighbors
    regressor_knn = KNeighborsRegressor()
    regressor_knn.fit(x_train, y_train)
    
    y_pred_knn = regressor_knn.predict(x_test)

    mse = mean_squared_error(y_test, y_pred_knn)
    print("KNN Mean Squared Error (MSE):", mse)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    print("KNN Root Mean Squared Error (RMSE):", rmse)
    r2 = r2_score(y_test, y_pred_knn)
    print("KNN R-squared (R²):", r2)

    #we find that Random Forest is the most efficient model to use in this case.

    # saving the model
    model_output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', "src", "models"))
    model_output_file =  os.path.join(model_output_directory, 'model.joblib')
    joblib.dump(regressor_rf, model_output_file)

train_model()