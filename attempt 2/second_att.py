import linreg_train
import linreg_test
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Read the data
train = pd.read_csv('train.csv')

#linreg_train
#train = pd.read_csv('train2.csv')
# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['OverallQual','GrLivArea', 'GarageArea', 'GarageCars', 'FullBath']

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# Read the test data
test = pd.read_csv('test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
#linreg_test

#pd.write_csv('test2.csv')
#test = pd.read_csv('test2.csv')
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('to_submit.csv', index=False)