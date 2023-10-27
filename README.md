# IPL_score_predictor
The project uses IPL data from start of the tournament to year 2022.
It predicts the final score of the match based on overs, batsman1_run, batsman2_run, total_wickets, extra_runs, team1, team2. EDA was performed on the dataset to select these feature variables.
To run the training and prediction pipeline use the main.py file.
It has chose linear regression as best model for current dataset out of 3 models:
1. Linear Regression
2. Random Forest
3. XGB

all three models were chose as they perform well on regression data with less volume.
The R2 score acheived from the model was 0.339 and RMSE was 542