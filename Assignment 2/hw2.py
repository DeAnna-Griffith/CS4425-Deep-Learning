import sklearn
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# fetches the california housing data and stores it in the variable housing
housing = sklearn.datasets.fetch_california_housing()

# Note: these two lines were made with the help of ChatGPT
# converts the dictionary of housing data into a dataframe and labels the columns
dataframe = pd.DataFrame(housing.data, columns=housing.feature_names)
dataframe["target"] = housing.target # adds on a target column in the data frame

scaler = StandardScaler()

# Distinguishes which parameters are a part of input, X, and output/"target", y
X = dataframe[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
y = dataframe['target']

# Splits the dataset into training data (80%) and test data (20%) setting the random_state to 0
# so that the same seed of numbers for train and test data remains the same
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# First fits the mean and standard deviation around those values for X_train and uses that to scale X_train
# Then, since mean and standard deviation is already fitted to X_train, they are used to transform/scale X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(dataframe.head())
print("\n\n")
print(dataframe.describe())
print("\n\n")

print(X_train_scaled)
print(X_test_scaled)
print("\n\n")

reg = LinearRegression().fit(X_train_scaled, y_train)
print(reg.score(X_test_scaled, y_test))

X_train_tensor = torch.from_numpy(X_train_scaled)
X_test_tensor = torch.from_numpy(X_test_scaled)



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        return x
