import locale

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import datetime
import time
import locale

app = Flask(__name__)

app.debug = True


def tsa(dataframe):
	print(time.strftime("%a, %d %b %Y %H:%M:%S"))
	# rest of the code here...
	# reformat dataframe
	df = dataframe.__deepcopy__()
	df = df.stack(level=0).drop(columns='rank')

	# Transform the MultiIndex to a datetime index
	locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
	print(datetime.datetime.now())
	print(datetime.datetime.strptime("January", "%B").month)
	print(datetime.datetime.strptime("December", "%B").month)

	result = []
	for x in df.index:
		year = x[0]
		print(str(x[1]) + " " + str(type(x[1])))
		month = datetime.datetime.strptime(x[1], "%B").month
		timestamp = pd.Timestamp(year=year, month=month, day=1)
		result.append(timestamp)
	print(result)

	df.index = result

	# Sort the DataFrame by date
	df = df.sort_index()

	# get the name of the first column
	first_col_name = df.columns

	# resample to fill in missing months
	df = df.resample('MS').asfreq()

	df.index

	# Create lag features
	for i in range(1, 13):
		df[f't-{i}'] = df[first_col_name].shift(i)
	df.dropna(inplace=True)

	# Split the df into training and testing sets
	train_size = int(0.8 * len(df))
	train = df.iloc[:train_size]
	test = df.iloc[train_size:]

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

	# Make predictions
	future = df.iloc[-12:].drop(first_col_name, axis=1)
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
	plt.savefig('static/plot.png')
	plt.clf()
	pass


@app.route('/', methods=['GET', 'POST'])
def index():
	filename = 'N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx'
	xl = pd.ExcelFile(filename)
	sheet_names = xl.sheet_names

	print("point1")

	if request.method == 'POST':
		choice = request.form.get('sheet-select', type=int)
		print("post!")
		if choice is None:
			print("point3")
			# User did not select a new sheet
			# Get the current sheet name from the hidden input field
			choice = int(request.form['current-sheet'])

		dataframe = pd.read_excel(filename, index_col=0, header=[0, 1], sheet_name=choice - 1)
		with app.test_request_context():
			print(dataframe)
		tsa(dataframe.__deepcopy__())
	print("point5")
	return render_template('index.html', sheet_names=sheet_names, plot_file='static/plot.png')


