import hw2
import csv

MLP_MSE = hw2.MLP_MSE
lin_reg_MSE = hw2.lin_reg_MSE
MLP_RMSE = hw2.MLP_RMSE
lin_reg_RMSE = hw2.lin_reg_RMSE

data = [{'MLP MSE': MLP_MSE, 'Linear Regression MSE': lin_reg_MSE, 'RMLP MSE': MLP_RMSE, 'Linear Regression RMSE': lin_reg_RMSE}]

with open('ModelMetrics.csv', 'w', newline='') as csvfile:
    fieldnames = ['MLP MSE', 'Linear Regression MSE', 'RMLP MSE', 'Linear Regression RMSE']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)