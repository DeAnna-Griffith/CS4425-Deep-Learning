import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        )
        
    def forward(self, x):
        return self.linear_relu_stack(x)


# fetches the california housing data and stores it in the variable housing
housing = sklearn.datasets.fetch_california_housing()

# Note: these two lines were made with the help of ChatGPT
# converts the dictionary of housing data into a dataframe and labels the columns
dataframe = pd.DataFrame(housing.data, columns=housing.feature_names)
dataframe["target"] = housing.target # adds on the target values to the target column
print(dataframe.head())

# Distinguishes which parameters are a part of input, X, and output/"target", y
X = dataframe[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
y = dataframe['target']

# Splits the dataset into training data (80%) and test data (20%) setting the random_state to 0
# so that the same seed of numbers for train and test data remains the same
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# First fits the mean and standard deviation around those values for X_train and uses that to scale X_train
# Then, since mean and standard deviation is already fitted to X_train, they are used to transform/scale X_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LinearRegression().fit(X_train_scaled, y_train)

X_train_tensor = torch.from_numpy(X_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).float()


model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
i = 140
losses = []

for epoch in range(i):
    
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{i}], Loss: {loss.item():.4f}")


model.eval()

with torch.no_grad():
    MLP_prediction = model(X_test_tensor)

lin_reg_prediction = reg.predict(X_test_scaled)

MLP_MSE = mean_squared_error(y_test, MLP_prediction)
lin_reg_MSE = mean_squared_error(y_test, lin_reg_prediction)

#print("\nMLP model Mean Squared Error: ")
#print(MLP_MSE)
#print("\nLinear Regression model Mean Squared Error: ")
#print(lin_reg_MSE)

MLP_RMSE = root_mean_squared_error(y_test, MLP_prediction)
lin_reg_RMSE = root_mean_squared_error(y_test, lin_reg_prediction)

#print("\nMLP model Root Mean Squared Error: ")
#print(MLP_RMSE)
#print("\nLinear Regression model Root Mean Squared Error: ")
#print(lin_reg_RMSE)

plt.plot(range(1, i + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 2 Training Loss")
plt.grid(True)

plt.savefig("model2_loss.png")
