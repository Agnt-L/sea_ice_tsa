import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

import pandas as pd

filename = 'N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx'

# read in the Excel file
xl = pd.ExcelFile(filename)

# get the sheet names from the Excel file
sheet_names = xl.sheet_names

# prompt the user to choose a sheet
for i, name in enumerate(sheet_names):
    print(f"{i+1}. {name}")

choice = int(input("Enter the number of the sheet to read: "))

# read in the chosen sheet as a pandas DataFrame
df = pd.read_excel(filename, index_col=0, header=[0, 1], sheet_name=choice-1)

# print the DataFrame
print(df)

# reformat dataframe
data_initial = df.copy()
data = data_initial.stack(level=0).drop(columns='rank')

# Transform the MultiIndex to a datetime index
data.index = pd.to_datetime(data.index.map(
    lambda x: '-'.join(map(str, x))), format='%Y-%B')

# Sort the DataFrame by date
data = data.sort_index()

# Print the resulting DataFrame with a sorted datetime index
print(data.head())


# get the name of the first column
first_col_name = data.columns
print(first_col_name)

# resample to fill in missing months
data = data.resample('MS').asfreq()

data.index


# Create lag features
for i in range(1, 13):
    data[f't-{i}'] = data[first_col_name].shift(i)
print(data)
data.dropna(inplace=True)
print(data)

# Split the data into training and testing sets
train_size = int(0.8 * len(data))
train = data.iloc[:train_size]
test = data.iloc[train_size:]

# Define X and y variables for training and testing
X_train = train.drop(first_col_name, axis=1)
y_train = train[first_col_name]
X_test = test.drop(first_col_name, axis=1)
y_test = test[first_col_name]

# Train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
                         max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
model.fit(X_train, y_train, eval_metric='rmse', eval_set=[
          (X_train, y_train), (X_test, y_test)], early_stopping_rounds=10, verbose=False)


# Evaluate the model
train_preds = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_mape = mean_absolute_percentage_error(y_train, train_preds) * 100
test_preds = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_mape = mean_absolute_percentage_error(y_test, test_preds) * 100
print('Train RMSE: %.3f' % train_rmse)
print('Train MAPE: %.3f%%' % train_mape)
print('Test RMSE: %.3f' % test_rmse)
print('Test MAPE: %.3f%%' % test_mape)

print(data)


# Make predictions
future = data.iloc[-12:].drop(first_col_name, axis=1)

pickle.dump(future, open('future.pkl', 'wb'))
model = pickle.load(open('future.pkl', 'rb'))

"""
future_preds = model.predict(future)
print('Future Predictions: ', future_preds)

print(y_test)


# Plot the results
plt.plot(test.index.values, test[first_col_name].values, label='Actual')
plt.plot(test.index.values, test_preds, label='Predicted')
plt.plot(future.index.values, future_preds, label='Future Predictions')
plt.xlabel('Date')
plt.ylabel(first_col_name)
plt.legend()
plt.savefig('plot.png')
"""
