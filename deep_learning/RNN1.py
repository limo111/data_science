import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(layout="wide")

st.sidebar.write("""
# Time Series Forecasting with LSTM

This app uses LSTM to forecast time series data.
""")

data = {
    'DATE': [
        '1992-01-01', '1992-02-01', '1992-03-01', '1992-04-01', '1992-05-01', '1992-06-01', '1992-07-01', '1992-08-01',
        '1992-09-01', '1992-10-01', '1992-11-01', '1992-12-01', '1993-01-01', '1993-02-01', '1993-03-01', '1993-04-01',
        '1993-05-01', '1993-06-01', '1993-07-01', '1993-08-01', '1993-09-01', '1993-10-01', '1993-11-01', '1993-12-01',
        '1994-01-01', '1994-02-01', '1994-03-01', '1994-04-01', '1994-05-01'
    ],
    'RSCCASN': [
        6938, 7524, 8475, 9401, 9558, 9182, 9103, 10513, 9573, 10254, 11187, 18395, 7502, 7524, 8766, 9867, 10063, 9635,
        9794, 10628, 10013, 10346, 11760, 18851, 7280, 7902, 9921, 9869, 10075
    ]
}

df = pd.DataFrame(data)
df['DATE'] = pd.to_datetime(df['DATE'])

tes = len(df) - 4

train = df.iloc[:tes]
test = df.iloc[tes:]

scaler = MinMaxScaler()

scaled_train = scaler.fit_transform(train[['RSCCASN']])
scaled_test = scaler.transform(test[['RSCCASN']])

length = 2
batch_size = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
valid_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=batch_size)

n_features = 1

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=[length, n_features]))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early = EarlyStopping(monitor='val_loss', mode='min', patience=4)

model.fit(generator, validation_data=valid_generator, epochs=20, callbacks=[early])

loss = pd.DataFrame(model.history.history)

st.subheader('Loss Plot')
st.line_chart(loss)

# Get user input for RSCCASN
rsccasn = st.sidebar.slider('RSCCASN', int(df['RSCCASN'].min()), int(df['RSCCASN'].max()), int(df['RSCCASN'].min()))

# Create user input DataFrame
df_user_input = pd.DataFrame({'DATE': [pd.to_datetime('today')], 'RSCCASN': [rsccasn]})

st.subheader('User Input')
st.write(df_user_input)

# Prepare test data with user input
user_input_scaled = scaler.transform(df_user_input[['RSCCASN']])
test_input = np.append(scaled_test[-length+1:], user_input_scaled, axis=0)

test_predictions = []
first = test_input[:length]
current_batch = first.reshape(1, length, n_features)
for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

test_pred = scaler.inverse_transform(test_predictions)

test['test_pred'] = np.round(test_pred, 0).astype('int32')

st.subheader('Test Data Plot')
fig, ax = plt.subplots()
ax.plot(test['DATE'], test['RSCCASN'], label='Actual')
ax.plot(test['DATE'], test['test_pred'], label='Predicted')
ax.legend()
st.pyplot(fig)

scaled_df = scaler.fit_transform(df[['RSCCASN']])

generator = TimeseriesGenerator(scaled_df, scaled_df, length=length, batch_size=batch_size)

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=[length, n_features]))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(generator, epochs=5)

forecast = []
first = scaled_df[-length:]
current_batch = first.reshape(1, length, n_features)
for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

forecast = scaler.inverse_transform(forecast)

forecast_dates = pd.date_range(df['DATE'].iloc[-1], periods=len(test) + 1, freq='MS')[1:]
forecast_df = pd.DataFrame({'DATE': forecast_dates, 'RSCCASN': forecast.flatten()})


st.subheader('Forecast Data Plot')
fig, ax = plt.subplots()
ax.plot(df['DATE'], df['RSCCASN'], label='Historical')
ax.plot(forecast_df['DATE'], forecast_df['RSCCASN'], label='Forecast')
ax.legend()
st.pyplot(fig)
