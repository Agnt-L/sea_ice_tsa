import time
import datetime
import locale
import pandas as pd
import numpy as np
from flask import *
from sklearn.metrics import mean_squared_error
import json
import xgboost as xgb
import plotly.graph_objs as go
import plotly

app = Flask(__name__)

app.debug = True

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))



def create_plot(test, model, future_preds):
    # Get the target variable name from the test set
    target_var = test.columns[0]

    # Get the predicted values for the test set
    test_preds = model.predict(test.drop(target_var, axis=1))

    # Create a plot of the actual and predicted values for the test set
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test[target_var], name="Actual"))
    fig.add_trace(go.Scatter(x=test.index, y=test_preds, name="Predicted"))

    # Add a line for the predicted future values
    future_dates = pd.date_range(start=test.index[-1], periods=12, freq='MS')
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Future Predictions"))

    # Set the plot layout
    fig.update_layout(title="Model Predictions", xaxis_title="Date", yaxis_title=target_var)

    # Save the plot as JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Save the plot as HTML
    #plot_html = pyo.plot(fig, output_type="div")

    return graphJSON


def print_time():
    """
    Prints the current time in the format: Day, DD Month YYYY HH:MM:SS
    """
    print(time.strftime("%a, %d %b %Y %H:%M:%S"))

def reformat_dataframe(dataframe):
    """
    Reformat the input dataframe to a desired format
    """
    df = dataframe.__deepcopy__()
    df = df.stack(level=0).drop(columns='rank')
    return df

def transform_index_to_datetime(df):
    """
    Transform the MultiIndex of the input dataframe to a datetime index
    """
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    result = []
    for x in df.index:
        year = x[0]
        month = datetime.datetime.strptime(x[1], "%B").month
        timestamp = pd.Timestamp(year=year, month=month, day=1)
        result.append(timestamp)
    df.index = result
    return df

def resample_dataframe(df):
    """
    Resample the input dataframe to fill in missing months
    """
    df = df.resample('MS').asfreq()
    return df

def create_lag_features(df, first_col_name):
    """
    Create lag features for the input dataframe
    """
    for i in range(1, 13):
        df[f't-{i}'] = df[first_col_name].shift(i)
    df.dropna(inplace=True)
    return df

def split_df_into_training_and_testing_sets(df, first_col_name):
    """
    Split the input dataframe into training and testing sets
    """
    train_size = int(0.8 * len(df))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    X_train = train.drop(first_col_name, axis=1)
    y_train = train[first_col_name]
    X_test = test.drop(first_col_name, axis=1)
    y_test = test[first_col_name]
    return X_train, y_train, X_test, y_test

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model on the input training data and evaluate on the input testing data
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
                             max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train, eval_metric='rmse', eval_set=[
        (X_train, y_train), (X_test, y_test)], early_stopping_rounds=10, verbose=False)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the performance of a trained machine learning model on training and test sets"""
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

    return train_rmse, train_mape, test_rmse, test_mape


def tsa(dataframe):
    print(time.strftime("%a, %d %b %Y %H:%M:%S"))

    # Reformat the dataframe
    df = reformat_dataframe(dataframe)

    # Split the dataframe into training and testing sets
    train, test, X_train, y_train, X_test, y_test, df, target = split_data(df)

    # Train the XGBoost model and make predictions
    model, future_preds = train_and_predict(X_train, y_train, X_test, y_test, df, target)

    # Create and return the plot
    plot_html = create_plot(test, model, future_preds)

    return plot_html

def reformat_dataframe(dataframe):
    """
    Reformat the dataframe by stacking the columns and transforming the MultiIndex to a datetime index
    """
    df = dataframe.__deepcopy__()
    df = df.stack(level=0).drop(columns='rank')

    # Transform the MultiIndex to a datetime index
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    result = []
    for x in df.index:
        year = x[0]
        month = datetime.datetime.strptime(x[1], "%B").month
        timestamp = pd.Timestamp(year=year, month=month, day=1)
        result.append(timestamp)
    df.index = result

    # Sort the DataFrame by date
    df = df.sort_index()

    return df

def split_data(df):
    """
    Split the df into training and testing sets and define X and y variables for training and testing
    """
    # get the name of the first column
    target = df.columns

    # resample to fill in missing months
    df = df.resample('MS').asfreq()

    # Create lag features
    for i in range(1, 13):
        df[f't-{i}'] = df[target].shift(i)
    df.dropna(inplace=True)

    # Split the df into training and testing sets
    train_size = int(0.8 * len(df))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    # Define X and y variables for training and testing
    X_train = train.drop(target, axis=1)
    y_train = train[target]
    X_test = test.drop(target, axis=1)
    y_test = test[target]

    return train, test, X_train, y_train, X_test, y_test, df, target

def train_and_predict(X_train, y_train, X_test, y_test, df, target):
    """
    Train the XGBoost model and make predictions
    """

    # Train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
                             max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train, eval_metric='rmse', eval_set=[(X_train, y_train), (X_test, y_test)],
              early_stopping_rounds=10, verbose=False)

    # Make predictions
    future = df.iloc[-12:].drop(target, axis=1)
    future_preds = model.predict(future)

    return model, future_preds


@app.route('/', methods=['GET', 'POST'])
def index():
	filename = 'N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx'
	xl = pd.ExcelFile(filename)
	sheet_names = xl.sheet_names

	graphJSON = None

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
		graphJSON = tsa(dataframe)
	if graphJSON:
		return render_template('index.html', sheet_names=sheet_names, graphJSON = graphJSON)
	else:
		return render_template('index.html', sheet_names=sheet_names)