import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')


# 1. Get Food & Beverage Services Index
# Read FBSI data
fbsi_df = pd.read_csv(r'C:\Users\Administrator\Desktop\ADMC\Assignment\FoodBeverageServicesIndex2017100AtCurrentPricesMonthly.csv')

# Set DataSeries as an index
fbsi_df.set_index('DataSeries', inplace=True)

# Transpose the data to a column
fbsi_df = fbsi_df.T

# Reset index
fbsi_df.reset_index(inplace=True)

# rename
fbsi_df.rename(columns={'index': 'date'}, inplace=True)

# parse date function
def parse_date(s):
    return datetime.datetime.strptime(s.strip(), '%Y%b')

fbsi_df['date'] = fbsi_df['date'].apply(parse_date)

# Get 'Cafes, Food Courts & Other Eating Places' column
fbsi_df['fbsi'] = pd.to_numeric(fbsi_df['Cafes, Food Courts & Other Eating Places'], errors='coerce')

# delete irrelevant column
fbsi_df = fbsi_df[['date', 'fbsi']]

# sort by date
fbsi_df.sort_values('date', inplace=True)

# Use Prophet to forecast FBSI
from prophet import Prophet

# Prepare Prophet model
fbsi_prophet = fbsi_df.rename(columns={'date': 'ds', 'fbsi': 'y'})

# Train Prophet
m = Prophet()
m.fit(fbsi_prophet)

# Generate future dates to  31-Dec-2024
last_date = fbsi_prophet['ds'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end='2024-12-31', freq='M')
future = pd.DataFrame({'ds': future_dates})

# Forecast FBSI
forecast = m.predict(future)

# Get forecast data
fbsi_future = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'fbsi'})

# Concat past and future data
fbsi_full = pd.concat([fbsi_df, fbsi_future], ignore_index=True)

# 2. Deal with past sales data
# Read data
df_store1 = pd.read_excel(r'C:\Users\Administrator\Desktop\ADMC\Assignment\Merlion Cafe Data AC6103 24T1.xlsx', sheet_name='CBD')
df_store2 = pd.read_excel(r'C:\Users\Administrator\Desktop\ADMC\Assignment\Merlion Cafe Data AC6103 24T1.xlsx', sheet_name='ORD')
df_store3 = pd.read_excel(r'C:\Users\Administrator\Desktop\ADMC\Assignment\Merlion Cafe Data AC6103 24T1.xlsx', sheet_name='TPY')

df_store1['store_number'] = 1
df_store2['store_number'] = 2
df_store3['store_number'] = 3

df = pd.concat([df_store1, df_store2, df_store3], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df.sort_values(['store_number', 'date'], inplace=True)

# Extract hour function
def extract_hour(hour_str):
    return int(hour_str.split(' to ')[0])

df['hour'] = df['hour'].apply(extract_hour)

# Filter business hours data (from 7 to 22)
df = df[(df['hour'] >= 7) & (df['hour'] <= 22)]

# 3. Data preprocessing and feature engineering
# Define public holidays in 2024
# According to 'https://publicholidays.sg/zh/2024-dates/'
holidays_2024 = [
    datetime.date(2024, 1, 1),
    datetime.date(2024, 2, 10),
    datetime.date(2024, 2, 11),
    datetime.date(2024, 2, 12),
    datetime.date(2024, 3, 29),
    datetime.date(2024, 4, 10),
    datetime.date(2024, 5, 1),
    datetime.date(2024, 5, 22),
    datetime.date(2024, 6, 17),
    datetime.date(2024, 8, 9),
    datetime.date(2024, 10, 31),
    datetime.date(2024, 12, 25)
]

# Add the time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Define Meal Period (as a feature)
def get_meal_period(hour):
    if 7 <= hour <= 8:
        return 'Breakfast'
    elif 10 <= hour <= 14:
        return 'Lunch'
    elif 17 <= hour <= 18:
        return 'Dinner'
    else:
        return 'Other'

df['meal_period'] = df['hour'].apply(get_meal_period)

# Add periodic features to df (keep other time features, such as month and day)
df['month_sin'] = df['month'].apply(lambda x: math.sin(2 * math.pi * x / 12))
df['month_cos'] = df['month'].apply(lambda x: math.cos(2 * math.pi * x / 12))

df['weekday_sin'] = df['weekday'].apply(lambda x: math.sin(2 * math.pi * x / 7))
df['weekday_cos'] = df['weekday'].apply(lambda x: math.cos(2 * math.pi * x / 7))

df['hour'] = df['hour'].astype(int)  # to make sure 'hour' is int type

# Add holiday feature
df['date_only'] = df['date'].dt.date
df['is_holiday'] = df['date_only'].isin(holidays_2024).astype(int)
df.drop('date_only', axis=1, inplace=True)

# Set the date of the FBSI to the first day of the month
fbsi_full['date'] = fbsi_full['date'].dt.to_period('M').dt.to_timestamp()

# Set the date of sales to the first day of the month
df['fbsi_month'] = df['date'].dt.to_period('M').dt.to_timestamp()

# Merge FBSI data into sales data
df = df.merge(fbsi_full[['date', 'fbsi']], left_on='fbsi_month', right_on='date', how='left')

# Drop unneeded column
df.drop(['fbsi_month', 'date_y'], axis=1, inplace=True)
df.rename(columns={'date_x': 'date'}, inplace=True)

# Replace the paces in the 'mode' column with underscores and fill in the missing values
df['mode'] = df['mode'].fillna('Unknown').astype(str).str.replace(' ', '_')

# Fill in the missing value of the 'product' column and convert it to a string
df['product'] = df['product'].fillna('Unknown').astype(str)

# Perform one-hot encoding on 'product', 'mode', and 'meal_period'
df = pd.get_dummies(df, columns=['product', 'mode', 'meal_period'], drop_first=False)

# Make sure the column names are of string type
df.columns = df.columns.map(str)

# Create a new target variable: the number of eat-in orders
# check 'mode_Eat_In' if there is null
if 'mode_Eat_In' in df.columns:
    df['is_eat_in'] = df['mode_Eat_In']
else:
    df['is_eat_in'] = 0

df['eat_in_quantity'] = df['quantity'] * df['is_eat_in']

# Calculate the proportion of eat-in orders
df['eat_in_ratio'] = df['eat_in_quantity'] / df['quantity']
df['eat_in_ratio'].fillna(0, inplace=True)  # Deal with the scenario of dividing by zero
df['eat_in_ratio'] = df['eat_in_ratio'].clip(0, 1)

# Calculate average discount rates and marketing spend
store_year_sales = df.groupby(['store_number', 'year'])['quantity'].sum().reset_index(name='total_quantity')

store_discounts = pd.DataFrame({
    'store_number': [1, 1, 2, 2, 3, 3],
    'year': [2022, 2023, 2022, 2023, 2022, 2023],
    'total_discount': [0, 0, 0, 57033.6, 0, 86564.2]
})

store_year_data = pd.merge(store_year_sales, store_discounts, on=['store_number', 'year'], how='left')
store_year_data['avg_discount_per_unit'] = store_year_data['total_discount'] / store_year_data['total_quantity']
df = pd.merge(df, store_year_data[['store_number', 'year', 'avg_discount_per_unit']], on=['store_number', 'year'], how='left')

store_marketing = pd.DataFrame({
    'store_number': [1, 1, 2, 2, 3, 3],
    'year': [2022, 2023, 2022, 2023, 2022, 2023],
    'total_marketing_spend': [30000, 30000, 30000, 12000, 120000, 12000]
})

df = pd.merge(df, store_marketing, on=['store_number', 'year'], how='left')
df = pd.merge(df, store_year_sales, on=['store_number', 'year'], how='left')
df['marketing_spend_per_unit'] = df['total_marketing_spend'] / df['total_quantity']

# Create a DataFrame for the inflation price coefficient and the consumption downgrade coefficient
year_factors = pd.DataFrame({
    'store_number': [1, 1, 2, 2, 3, 3],
    'year': [2022, 2023, 2022, 2023, 2022, 2023],
    'inflation_price_coefficient': [1, 1, 1, 1.05, 1.05, 1.05],
    'consumption_downgrade_coefficient': [1, 1, 1, 1, 0.8425, 1]
})

# Merges year level factors into the master data set
df = pd.merge(df, year_factors, on=['store_number', 'year'], how='left')

# Coefficient of filling missing is 1 (for 2024 and any missing)
df['inflation_price_coefficient'].fillna(1, inplace=True)
df['consumption_downgrade_coefficient'].fillna(1, inplace=True)

# Processing missing value
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Aggregate data (by date, hour, and store)
aggregation_functions = {
    'fbsi': 'first',
    'quantity': 'sum',
    'eat_in_quantity': 'sum',
    'eat_in_ratio': 'mean',
    'avg_discount_per_unit': 'mean',
    'marketing_spend_per_unit': 'mean',
    'inflation_price_coefficient': 'first',
    'consumption_downgrade_coefficient': 'first',
    'is_holiday': 'max',
    'is_weekend': 'max',
    'month': 'first',
    'day': 'first',
    'weekday': 'first',
    'month_sin': 'first',
    'month_cos': 'first',
    'weekday_sin': 'first',
    'weekday_cos': 'first',
    'year': 'first'
}

# Sum the columns that are one-hot
one_hot_cols = [col for col in df.columns if 'product_' in col or 'mode_' in col or 'meal_period_' in col]
for col in one_hot_cols:
    aggregation_functions[col] = 'sum'

df_aggregated = df.groupby(['store_number', 'date', 'hour']).agg(aggregation_functions).reset_index()

# Add hysteresis features and roll statistics features
df_aggregated = df_aggregated.sort_values(['store_number', 'date', 'hour'])

# Lag characteristics on total order quantity
df_aggregated['quantity_lag_1'] = df_aggregated.groupby('store_number')['quantity'].shift(1)
df_aggregated['quantity_lag_7'] = df_aggregated.groupby('store_number')['quantity'].shift(7)
df_aggregated['quantity_roll_mean_7'] = df_aggregated.groupby('store_number')['quantity'].shift(1).rolling(window=7).mean()
df_aggregated['quantity_roll_std_7'] = df_aggregated.groupby('store_number')['quantity'].shift(1).rolling(window=7).std()

# The lag characteristics of the proportion of eat-in orders
df_aggregated['eat_in_ratio_lag_1'] = df_aggregated.groupby('store_number')['eat_in_ratio'].shift(1)
df_aggregated['eat_in_ratio_lag_7'] = df_aggregated.groupby('store_number')['eat_in_ratio'].shift(7)
df_aggregated['eat_in_ratio_roll_mean_7'] = df_aggregated.groupby('store_number')['eat_in_ratio'].shift(1).rolling(window=7).mean()
df_aggregated['eat_in_ratio_roll_std_7'] = df_aggregated.groupby('store_number')['eat_in_ratio'].shift(1).rolling(window=7).std()

# Fill missing value
df_aggregated.fillna(method='bfill', inplace=True)

# Update data set
df = df_aggregated

# 4. Define features and target variables
feature_cols = [
    'fbsi',
    'month', 'day', 'weekday', 'hour', 'is_weekend',
    'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
    'is_holiday',
    'avg_discount_per_unit', 'marketing_spend_per_unit',
    'inflation_price_coefficient', 'consumption_downgrade_coefficient',
    'quantity_lag_1', 'quantity_lag_7', 'quantity_roll_mean_7', 'quantity_roll_std_7',
    'eat_in_ratio_lag_1', 'eat_in_ratio_lag_7', 'eat_in_ratio_roll_mean_7', 'eat_in_ratio_roll_std_7',
    'store_number'
] + one_hot_cols

# Define target variable
target_variables = ['quantity', 'eat_in_ratio']

# 5. Dividing the training set and the test set (with October 15, 2023 as the dividing line)
train_end_date = '2023-10-15'
X_train = df[df['date'] <= train_end_date][feature_cols]
X_test = df[df['date'] > train_end_date][feature_cols]
y_train = df[df['date'] <= train_end_date][target_variables]
y_test = df[df['date'] > train_end_date][target_variables]

# Check the size of the training and test sets
print("Training set：", len(X_train))
print("Test set：", len(X_test))

# 6. Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

# 7. Create sequence data
time_steps = 7

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    y = y.reset_index(drop=True)
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps].values)
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)

# 8. Build and train LSTM multi-output model (predicting total order volume and proportion of food order)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Model

# Define model input
input_layer = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

# Define the shared LSTM layer
shared_lstm = LSTM(units=64, return_sequences=True)(input_layer)
shared_lstm = Dropout(0.3)(shared_lstm)
shared_lstm = LSTM(units=32)(shared_lstm)
shared_lstm = Dropout(0.3)(shared_lstm)

# Defines the output layer for the total order quantity
quantity_output = Dense(1, activation='relu', name='quantity_output')(shared_lstm)

# Define an output layer for the proportion of food orders, using the Sigmoid activation function
eat_in_ratio_output = Dense(1, activation='sigmoid', name='eat_in_ratio_output')(shared_lstm)

# Defining model
model = Model(inputs=input_layer, outputs=[quantity_output, eat_in_ratio_output])

# Compile the model and modify the loss function
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'quantity_output': 'mean_squared_error', 'eat_in_ratio_output': 'mean_squared_error'},
              metrics={'quantity_output': ['mae', 'mape'], 'eat_in_ratio_output': ['mae', 'mape']})

# Define the early stop callback function
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Training model
history = model.fit(X_train_seq,
                    [y_train_seq[:, 0], y_train_seq[:, 1]],
                    epochs=50,
                    batch_size=128,
                    validation_data=(X_test_seq, [y_test_seq[:, 0], y_test_seq[:, 1]]),
                    callbacks=[early_stop])

# 9. Model evaluation and prediction

# Prediction
quantity_pred, eat_in_ratio_pred = model.predict(X_test_seq)

# Limit the percentage of predicted eat-in orders to between 0 and 1
eat_in_ratio_pred = eat_in_ratio_pred.clip(0, 1)

# Calculate the predicted eat-in order quantity
eat_in_pred = quantity_pred * eat_in_ratio_pred

# Converts the predicted and real values to integers
quantity_pred = np.rint(quantity_pred).astype(int)
eat_in_pred = np.rint(eat_in_pred).astype(int)
y_test_quantity = y_test_seq[:, 0].astype(int)
y_test_eat_in_quantity = (y_test_seq[:, 0] * y_test_seq[:, 1]).astype(int)

# Ensure that the order quantity does not exceed the total order quantity
eat_in_pred = np.minimum(eat_in_pred, quantity_pred)

# Calculate the MSE and MAE of the total order quantity
quantity_mse = mean_squared_error(y_test_quantity, quantity_pred)
quantity_mae = mean_absolute_error(y_test_quantity, quantity_pred)
print(f'Total orders - MSE on the test set: {quantity_mse:.2f}')
print(f'Total orders - MAE on the test set: {quantity_mae:.2f}')

# Calculate the MSE and MAE of the eat-in order quantity
eat_in_mse = mean_squared_error(y_test_eat_in_quantity, eat_in_pred)
eat_in_mae = mean_absolute_error(y_test_eat_in_quantity, eat_in_pred)
print(f'Eat-in order volume - MSE on test set: {eat_in_mse:.2f}')
print(f'Eat-in order volume - MAE on test set: {eat_in_mae:.2f}')

# Visualization - order quantity
plt.figure(figsize=(12, 6))
plt.plot(y_test_quantity[:100], label='Actual value')
plt.plot(quantity_pred[:100], label='Predicted value')
plt.legend()
plt.title('Order Quantity - Assessing fit')
plt.xlabel('Sample')
plt.ylabel('Order quantity')
plt.show()

# Visualization - eat-in order quantity
plt.figure(figsize=(12, 6))
plt.plot(y_test_eat_in_quantity[:100], label='Actual value')
plt.plot(eat_in_pred[:100], label='Predicted value')
plt.legend()
plt.title('Eat In Orders - Assessing fit')
plt.xlabel('Sample')
plt.ylabel('Eat in orders')
plt.show()

# 10. Forecast data for 2024
# Generation date range
future_dates = pd.date_range(start='2024-01-01', end='2024-12-31')

# Define a new DataFrame
future_df_list = []

# For each date, add the number of hours of operation
for date in future_dates:
    for hour in range(7, 23):  # from 7 to 22
        future_df_list.append({'date': pd.Timestamp(date.date()) + pd.Timedelta(hours=hour)})

# Convert the list to a DataFrame
future_df = pd.DataFrame(future_df_list)

# Add time features
future_df['year'] = future_df['date'].dt.year
future_df['month'] = future_df['date'].dt.month
future_df['day'] = future_df['date'].dt.day
future_df['weekday'] = future_df['date'].dt.weekday
future_df['hour'] = future_df['date'].dt.hour
future_df['is_weekend'] = future_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Define meal period
future_df['meal_period'] = future_df['hour'].apply(get_meal_period)

# Add periodic features to future_df (keep other time features, such as month and day)
future_df['month_sin'] = future_df['month'].apply(lambda x: math.sin(2 * math.pi * x / 12))
future_df['month_cos'] = future_df['month'].apply(lambda x: math.cos(2 * math.pi * x / 12))

future_df['weekday_sin'] = future_df['weekday'].apply(lambda x: math.sin(2 * math.pi * x / 7))
future_df['weekday_cos'] = future_df['weekday'].apply(lambda x: math.cos(2 * math.pi * x / 7))

# Add holiday features
future_df['date_only'] = future_df['date'].dt.date
future_df['is_holiday'] = future_df['date_only'].isin(holidays_2024).astype(int)
future_df.drop('date_only', axis=1, inplace=True)

# Merge FBSI data
# Set the date of the FBSI to the first day of the month
fbsi_full['date'] = fbsi_full['date'].dt.to_period('M').dt.to_timestamp()

# Set the date of future data to the first day of the month
future_df['fbsi_month'] = future_df['date'].dt.to_period('M').dt.to_timestamp()

# Merge FBSI data to future data
future_df = future_df.merge(fbsi_full[['date', 'fbsi']], left_on='fbsi_month', right_on='date', how='left')

# Delete redundant columns
future_df.drop(['fbsi_month', 'date_y'], axis=1, inplace=True)
future_df.rename(columns={'date_x': 'date'}, inplace=True)

# Add the 'product' and 'mode' columns and fill them with 'Unknown'
future_df['product'] = 'Unknown'
future_df['mode'] = 'Unknown'

# Perform one-hot encoding on 'product', 'mode', and 'meal_period'
future_df = pd.get_dummies(future_df, columns=['product', 'mode', 'meal_period'], drop_first=False)

# Ensure all one-hot coding columns exist in future_df
for col in one_hot_cols:
    if col not in future_df.columns:
        future_df[col] = 0  # 如果不存在，则填充为0

# Make sure the column names are of string type
future_df.columns = future_df.columns.map(str)

# Add 'inflation_price_coefficient' and 'consumption_downgrade_coefficient' with default value 1
future_df['inflation_price_coefficient'] = 1
future_df['consumption_downgrade_coefficient'] = 1

store_numbers = df['store_number'].unique()
future_dfs = []

# Define avg_discount_per_unit and marketing_spend_per_unit for each store
avg_discount_per_unit_values = {
    1: 0.062528648,
    2: 0.071065835,
    3: 0.232483096
}

marketing_spend_per_unit_values = {
    1: 0.100036735,
    2: 0.026274854,
    3: 0.232825931
}

for store_number in store_numbers:
    future_df_store = future_df.copy()
    future_df_store['store_number'] = store_number

    # 设置每个门店的 avg_discount_per_unit 和 marketing_spend_per_unit
    future_df_store['avg_discount_per_unit'] = avg_discount_per_unit_values.get(
        store_number, df['avg_discount_per_unit'].mean())
    future_df_store['marketing_spend_per_unit'] = marketing_spend_per_unit_values.get(
        store_number, df['marketing_spend_per_unit'].mean())

    # Hysteresis features and rolling statistics require special treatment, assuming the last values of the training set are used
    last_quantity = df[df['store_number'] == store_number]['quantity'].iloc[-1]
    last_quantity_roll_mean_7 = df[df['store_number'] == store_number]['quantity_roll_mean_7'].iloc[-1]
    last_quantity_roll_std_7 = df[df['store_number'] == store_number]['quantity_roll_std_7'].iloc[-1]

    last_eat_in_ratio = df[df['store_number'] == store_number]['eat_in_ratio'].iloc[-1]
    last_eat_in_ratio_roll_mean_7 = df[df['store_number'] == store_number]['eat_in_ratio_roll_mean_7'].iloc[-1]
    last_eat_in_ratio_roll_std_7 = df[df['store_number'] == store_number]['eat_in_ratio_roll_std_7'].iloc[-1]

    future_df_store['quantity_lag_1'] = last_quantity
    future_df_store['quantity_lag_7'] = last_quantity  # 简化处理
    future_df_store['quantity_roll_mean_7'] = last_quantity_roll_mean_7
    future_df_store['quantity_roll_std_7'] = last_quantity_roll_std_7

    future_df_store['eat_in_ratio_lag_1'] = last_eat_in_ratio
    future_df_store['eat_in_ratio_lag_7'] = last_eat_in_ratio  # 简化处理
    future_df_store['eat_in_ratio_roll_mean_7'] = last_eat_in_ratio_roll_mean_7
    future_df_store['eat_in_ratio_roll_std_7'] = last_eat_in_ratio_roll_std_7

    future_dfs.append(future_df_store)

# Concat future data for all stores
future_df_full = pd.concat(future_dfs, ignore_index=True)

# Fill in possible missing values
future_df_full.fillna(method='ffill', inplace=True)
future_df_full.fillna(method='bfill', inplace=True)

# Selection feature column
future_X = future_df_full[feature_cols]

# Normalization
future_X_scaled = scaler.transform(future_X)
future_X_scaled = pd.DataFrame(future_X_scaled, columns=feature_cols)

# Create sequence data
X_future_seq = []
for i in range(len(future_X_scaled) - time_steps + 1):
    X_future_seq.append(future_X_scaled.iloc[i:i + time_steps].values)

X_future_seq = np.array(X_future_seq)

# Define the combination of the inflation coefficients and the consumption downgrade coefficients
inflation_coefficients = [1, 1.05]
consumption_downgrade_coefficients = [1, 1]

import itertools
scenarios = list(itertools.product(inflation_coefficients, consumption_downgrade_coefficients))

# Predict and save results
for inflation_coef, consumption_downgrade_coef in scenarios:
    scenario_name = f'inflation_{inflation_coef}_consumption_{consumption_downgrade_coef}'
    print(f'Processing scenario: {scenario_name}')

    # Adjust the year level factor for each store
    future_df_scenario = future_df_full.copy()

    # Set inflation coefficients
    future_df_scenario['inflation_price_coefficient'] = inflation_coef

    # Set consumption_downgrade_coefficients
    # For store 2, use consumption_downgrade_coef and keep 1 for other stores
    future_df_scenario['consumption_downgrade_coefficient'] = 1
    future_df_scenario.loc[future_df_scenario['store_number'] == 2, 'consumption_downgrade_coefficient'] = consumption_downgrade_coef

    # Update feature column
    future_X_scenario = future_df_scenario[feature_cols]

    # Normalization
    future_X_scaled_scenario = scaler.transform(future_X_scenario)
    future_X_scaled_scenario = pd.DataFrame(future_X_scaled_scenario, columns=feature_cols)

    # Create sequence data
    X_future_seq_scenario = []
    for i in range(len(future_X_scaled_scenario) - time_steps + 1):
        X_future_seq_scenario.append(future_X_scaled_scenario.iloc[i:i + time_steps].values)

    X_future_seq_scenario = np.array(X_future_seq_scenario)

    # Forecast
    future_quantity_pred, future_eat_in_ratio_pred = model.predict(X_future_seq_scenario)

    # Limit the percentage of predicted eat-in orders to between 0 and 1
    future_eat_in_ratio_pred = future_eat_in_ratio_pred.clip(0, 1)

    # Calculate the predicted eat-in order volume
    future_eat_in_pred = future_quantity_pred * future_eat_in_ratio_pred

    # Converts the predicted value to an integer
    future_quantity_pred = np.rint(future_quantity_pred).astype(int)
    future_eat_in_pred = np.rint(future_eat_in_pred).astype(int)

    # Ensure that the eat-in order quantity does not exceed the total order quantity
    future_eat_in_pred = np.minimum(future_eat_in_pred, future_quantity_pred)

    # Prepare forecast results
    future_df_results = future_df_scenario[time_steps - 1:].copy()
    future_df_results['predicted_quantity'] = future_quantity_pred
    future_df_results['predicted_eat_in_quantity'] = future_eat_in_pred

    # On public holidays, set the forecast sales to zero
    future_df_results.loc[future_df_results['is_holiday'] == 1, 'predicted_quantity'] = 0
    future_df_results.loc[future_df_results['is_holiday'] == 1, 'predicted_eat_in_quantity'] = 0

    # Output the result to an Excel file
    output_columns = ['date', 'store_number', 'predicted_quantity', 'predicted_eat_in_quantity']
    output_filename = f'2024_quantity_predictions_{scenario_name}.xlsx'
    future_df_results.to_excel(output_filename, columns=output_columns, index=False)

    print(f"Save as {output_filename}")

# 11. Optimization to Maximize Annual Profit Using Simulated Annealing
# Define price base and cost base per store
price_base_per_store = {1: 5.66, 2: 7.33 / 0.8425, 3: 7.07}
cost_base_per_store = {1: 2.55, 2: 3.60, 3: 3.22}

# Define the scenarios
price_scenarios = {'Conservative': 1.0, 'Aggressive': 1.05}
cost_scenarios = {'Unchanged': 1.0, 'Small Increase': 1.05, 'Inflation': 1.10}
consumption_downgrade_options = {
    'Same as 2023': {1: 1, 2: 0.8425, 3: 1},
    'Same as 2022': {1: 1, 2: 1, 3: 1}
}

import itertools

scenario_combinations = list(
    itertools.product(price_scenarios.items(), cost_scenarios.items(), consumption_downgrade_options.items()))

# Store numbers
store_numbers = df['store_number'].unique()

# Initialize results list
results = []

# For each scenario
for (price_name, price_coef), (cost_name, cost_coef), (
consumption_name, consumption_downgrade_dict) in scenario_combinations:
    scenario_name = f"{price_name}_{cost_name}_{consumption_name}"
    print(f"Processing scenario: {scenario_name}")

    # Prepare future_df_scenario similar to before, but set 'inflation_price_coefficient' to price_coef
    # and 'consumption_downgrade_coefficient' per store based on consumption_downgrade_dict

    # Initialize future_df_scenario
    future_df_scenario = future_df_full.copy()

    # Set 'inflation_price_coefficient' to price_coef
    future_df_scenario['inflation_price_coefficient'] = price_coef

    # Set 'consumption_downgrade_coefficient' per store
    for store_num in store_numbers:
        future_df_scenario.loc[future_df_scenario['store_number'] == store_num, 'consumption_downgrade_coefficient'] = \
        consumption_downgrade_dict[store_num]

    # For each store, perform simulated annealing
    for store_num in store_numbers:
        print(f"Optimizing for store {store_num}")
        # Extract data for the store
        future_df_store = future_df_scenario[future_df_scenario['store_number'] == store_num].copy()

        # Initialize 'avg_discount_per_unit' and 'marketing_spend_per_unit' with initial values
        initial_discount = df[df['store_number'] == store_num]['avg_discount_per_unit'].mean()
        initial_marketing_spend = df[df['store_number'] == store_num]['marketing_spend_per_unit'].mean()


        # Define function to compute profit and perform simulated annealing
        def simulated_annealing(future_df_store, initial_discount, initial_marketing_spend, model,
                                price_base_per_store, cost_base_per_store, price_coef, cost_coef,
                                consumption_downgrade_dict, store_num, scaler, feature_cols, time_steps,
                                max_iter=100, T0=1.0, alpha=0.95):
            import random

            # Initialize variables
            current_discount = initial_discount
            current_marketing_spend = initial_marketing_spend
            current_profit = None

            best_discount = current_discount
            best_marketing_spend = current_marketing_spend
            best_profit = -np.inf
            best_total_quantity = 0
            best_price_per_unit = 0

            T = T0
            for i in range(max_iter):
                # Perturb the variables
                new_discount = current_discount + random.uniform(-0.005, 0.005)  # Smaller steps for finer adjustments
                new_marketing_spend = current_marketing_spend + random.uniform(-0.005, 0.005)

                # Clip to valid ranges
                new_discount = max(0, min(new_discount, 0.5))
                new_marketing_spend = max(0, min(new_marketing_spend, 0.5))

                # Update features
                future_df_store['avg_discount_per_unit'] = new_discount
                future_df_store['marketing_spend_per_unit'] = new_marketing_spend

                # Prepare features
                future_X = future_df_store[feature_cols]
                # Scale features
                future_X_scaled = scaler.transform(future_X)
                future_X_scaled = pd.DataFrame(future_X_scaled, columns=feature_cols)

                # Create sequences
                X_future_seq = []
                for j in range(len(future_X_scaled) - time_steps + 1):
                    X_future_seq.append(future_X_scaled.iloc[j:j + time_steps].values)
                X_future_seq = np.array(X_future_seq)

                # Predict quantities
                future_quantity_pred, future_eat_in_ratio_pred = model.predict(X_future_seq, verbose=0)
                future_quantity_pred = future_quantity_pred.flatten()
                future_eat_in_ratio_pred = future_eat_in_ratio_pred.flatten()
                future_quantity_pred = np.rint(future_quantity_pred).astype(int)
                total_quantity = future_quantity_pred.sum()

                # Compute price and cost per unit
                price_base = price_base_per_store[store_num]
                cost_base = cost_base_per_store[store_num]
                consumption_downgrade_coef = consumption_downgrade_dict[store_num]

                price_per_unit = price_base * price_coef * consumption_downgrade_coef
                cost_per_unit = cost_base * cost_coef

                # Compute total revenue and cost
                total_revenue = price_per_unit * total_quantity
                total_discount = new_discount * total_quantity
                total_marketing_spend = new_marketing_spend * total_quantity
                total_cost = (cost_per_unit * total_quantity) + total_discount + total_marketing_spend

                # Compute profit
                new_profit = total_revenue - total_cost

                # Accept or reject
                if current_profit is None or new_profit > current_profit:
                    accept = True
                else:
                    delta = new_profit - current_profit
                    accept_prob = np.exp(delta / T)
                    accept = random.random() < accept_prob

                if accept:
                    current_discount = new_discount
                    current_marketing_spend = new_marketing_spend
                    current_profit = new_profit

                    if current_profit > best_profit:
                        best_discount = current_discount
                        best_marketing_spend = current_marketing_spend
                        best_profit = current_profit
                        best_total_quantity = total_quantity
                        best_price_per_unit = price_per_unit

                # Cool down
                T *= alpha

            return best_discount, best_marketing_spend, best_profit, best_total_quantity, best_price_per_unit


        # Run simulated annealing
        best_discount, best_marketing_spend, best_profit, best_total_quantity, best_price_per_unit = simulated_annealing(
            future_df_store, initial_discount, initial_marketing_spend, model,
            price_base_per_store, cost_base_per_store, price_coef, cost_coef,
            consumption_downgrade_dict, store_num,
            scaler, feature_cols, time_steps, max_iter=100, T0=1.0, alpha=0.95
        )

        # Store results
        results.append({
            'scenario': scenario_name,
            'store_number': store_num,
            'best_discount_per_unit': best_discount,
            'best_marketing_spend_per_unit': best_marketing_spend,
            'best_profit': best_profit,
            'total_quantity': best_total_quantity,
            'price_per_unit': best_price_per_unit
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Output results
print(results_df)
results_df.to_excel('Optimise_Result.xlsx', index=False)