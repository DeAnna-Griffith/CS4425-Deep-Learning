## Running the program

First, to ensure that all necessary dependacies are installed, enter the line into the terminal:
    python -m pip install pandas torch matplotlib scikit-learn

Then to run the program enter:
    python hw2.py

## Assignment 2: part 1
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  target
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23   4.526
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22   3.585
2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24   3.521
3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25   3.413
4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25   3.422


The table that was output when printing dataframe.head()


## Model Loss Table

|       MLP MSE       |Linear Regression MSE|      RMLP MSE       |Linear Regression RMSE|
|:-------------------:|:-------------------:|:-------------------:|:--------------------:|
| 0.4110268118111734  |  0.528984167036721  | 0.6411137276733149  |  0.7273129773603115  |


The table shows that of the two models, the MLP model trained with pytorch did marginally better than the sklearn Linear Regression model. 


## Model 2 Loss Graph

![Model 2 Loss](<Assignment 2/model2_loss.png>)

The graph shows a sharp decline in loss from 0 to 20 epochs, followed by a more gradual decline until around 60 epochs. After that point, the loss continues to decrease, but at a much slower rate. The difference in loss between approximately 100 and 140 epochs is relatively small and has little practical impact on the model's performance. Overall, the graph shows that increasing the number of epochs reduces the model's training error, but the improvement becomes increasingly marginal as the number of epochs increases. Based on the graph, a value somewhere in the 100–140 epoch range would reasonably suffice for training this model.


