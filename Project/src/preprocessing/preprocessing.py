import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# loading the dataset
data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..',"src","data"))
data_file = os.path.join(data_directory, 'Housing.csv')
df = pd.read_csv(data_file)

# creating the encoder and making a copy of the dataframe
labelEncoder = LabelEncoder()
encoded_df = df

# encoding the dataframe copy
columns_to_encode = ['mainroad', 'guestroom','basement', 'hotwaterheating', 'airconditioning', 'prefarea','furnishingstatus']
for column in columns_to_encode:
    encoded_df[column] = labelEncoder.fit_transform(encoded_df[column])

# standardization to make the data training-ready
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
scaled_df['price'] = df['price']

# creating the correlation matrix to check for useless columns
correlation_matrix = scaled_df.corr()
plt.figure(figsize=(15, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# re-fitting the scaler to the data without the useless columns
scaled_data=scaler.fit_transform(scaled_df.drop(columns=['hotwaterheating','price']))
scaled_df=pd.DataFrame(scaled_data, columns=df.drop(columns=['hotwaterheating','price']).columns)
scaled_df['price']=df['price']
scaled_df = scaled_df[['price']+scaled_df.columns.tolist()[:-1]]
final_df=scaled_df

# saving the encoder and scaler
model_output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', "src", "models"))
joblib.dump(labelEncoder, os.path.join(model_output_directory, 'labelEncoder.joblib')) 
joblib.dump(scaler, os.path.join(model_output_directory, 'scaler.joblib'))

# save the preprocessed data for training
preprocessing_directory = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..',"src","preprocessing"))
final_df.to_csv(os.path.join(preprocessing_directory, 'Housing_preprocessed.csv'), index=False)
