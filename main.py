import pandas as pd
from training_model import trainModel
from modelpredict import prediction
from datetime import datetime
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

matches_data = pd.read_csv('Dataset/IPL_Matches_2008_2022.csv')
match_data = pd.read_csv('Dataset/IPL_Ball_by_Ball_2008_2022.csv')

agg_data = match_data.merge(matches_data,how='inner',on='ID')
agg_data['Date'] = agg_data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

train_data = agg_data[agg_data['Date'].dt.year <= 2021]
#print(agg_data.isna().any())
train_data = train_data.reset_index(drop=True)
model = trainModel()
model.trainingModel(train_data)

test_data = agg_data[agg_data['Date'].dt.year > 2021]
test_data = test_data.reset_index(drop=True)

predictor = prediction()
predictor.predictionFromModel(test_data)
y_est = pd.read_csv("Prediction_Output_File/Predictions.csv")
y_true = pd.read_csv("Prediction_Output_File/actual_y.csv")
print("R2 score: ",r2_score(y_pred=y_est,y_true=y_true))
print("Mean sqaured error: ",mean_squared_error(y_pred=y_est,y_true=y_true))
