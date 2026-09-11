import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import load_model

import pandas as pd

# df = pd.read_csv("data/PJME_hourly.csv")

# df['Datetime'] = pd.to_datetime(df['Datetime'])

# df = df.sort_values("Datetime")

# df.rename(columns={'PJME_MW':'demand'}, inplace=True)

# # ADD THESE FEATURES
# df['hour'] = df['Datetime'].dt.hour
# df['dayofweek'] = df['Datetime'].dt.dayofweek
# df['month'] = df['Datetime'].dt.month

st.set_page_config(
    page_title="Electricity Demand Forecast",
    page_icon="⚡",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/PJME_hourly.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values("Datetime")
    df.rename(columns={'PJME_MW':'demand'}, inplace=True)

    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month

    return df

df = load_data()

# ---------------------------
# Load Model and Scaler
# ---------------------------

model = load_model("models/electricity_lstm_model.h5")

scaler = pickle.load(open("models/scaler.pkl","rb"))


# ---------------------------
# Page Title
# ---------------------------

st.title("⚡ Electricity Demand Forecasting Dashboard")

st.write("Predict electricity demand based on time patterns")


# ---------------------------
# Sidebar Inputs
# ---------------------------

st.sidebar.header("Input Parameters")

date = st.sidebar.date_input("Select Date")

hour = st.sidebar.slider("Select Hour",0,23)


# ---------------------------
# Time Feature Extraction
# ---------------------------

dayofweek = date.weekday()
month = date.month


# ---------------------------
# Time of Day Label (UI only)
# ---------------------------

def get_time_period(hour):

    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


time_period = get_time_period(hour)

st.write("### Selected Time Period:", time_period)


# ---------------------------
# Create Input Sequence
# ---------------------------

# demand placeholder = 0

sequence = []

last_24 = df['demand'].values[-24:]

for i in range(24):

    h = (hour + i) % 24

    sequence.append([
        last_24[i],
        h,
        dayofweek,
        month
    ])

# find the latest 24 records
recent_data = df.tail(24).copy()

# update time features based on selected hour
for i in range(24):
    recent_data.iloc[i, recent_data.columns.get_loc("hour")] = (hour - 24 + i) % 24

sequence = recent_data[['demand','hour','dayofweek','month']].values

sequence_scaled = scaler.transform(sequence)

sequence_scaled = sequence_scaled.reshape(1,24,4)

# ---------------------------
# Prediction Button
# ---------------------------

if st.button("Predict Electricity Demand"):

    prediction = model.predict(sequence_scaled)

    dummy = np.zeros((1,4))
    dummy[0,0] = prediction[0][0]

    demand = scaler.inverse_transform(dummy)[0][0]

    st.metric(
    "Predicted Electricity Demand",
    f"{int(demand):,} MW"
    )

    st.write(f"Forecast for **{hour}:00** on **{date}**")

tabs = st.tabs([
    "24 Hour Forecast",
    "Daily Demand Pattern",
    "Weekday vs Weekend",
    "Demand Heatmap"
])

# ---------------------------
# 24 Hour Forecast
# ---------------------------
with tabs[0]:

    st.subheader("Next 24 Hour Electricity Forecast")

    recent_data = df.tail(24)
    history = recent_data[['demand','hour','dayofweek','month']].values.tolist()

    future_hours = []
    future_predictions = []

    for i in range(24):

        h = (hour + i) % 24

        new_row = [history[-1][0], h, dayofweek, month]

        history.append(new_row)

        input_seq = np.array(history[-24:])

        scaled = scaler.transform(input_seq)

        scaled = scaled.reshape(1,24,4)

        pred = model.predict(scaled)[0][0]

        dummy = np.zeros((1,4))
        dummy[0,0] = pred

        demand = scaler.inverse_transform(dummy)[0][0]

        future_predictions.append(demand)
        future_hours.append(h)

    fig, ax = plt.subplots()

    ax.plot(future_hours, future_predictions, marker="o")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Demand (MW)")
    ax.set_title("24 Hour Electricity Demand Forecast")

    ax.grid(True)

    st.pyplot(fig)




# ---------------------------
# Daily Pattern Chart
# ---------------------------
with tabs[1]:

    st.subheader("Typical Daily Demand Pattern")

    hourly_pattern = df.groupby("hour")["demand"].mean()

    fig, ax = plt.subplots()

    ax.plot(hourly_pattern.index, hourly_pattern.values, marker="o")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Demand (MW)")
    ax.set_title("Typical Daily Electricity Demand")

    ax.grid(True)

    st.pyplot(fig)


# ---------------------------
# Weekday vs Weekend Demand
# ---------------------------


with tabs[2]:

    st.subheader("Weekday vs Weekend Electricity Demand")

    df["is_weekend"] = df["dayofweek"] >= 5

    weekday_demand = df[df["is_weekend"] == False]["demand"].mean()
    weekend_demand = df[df["is_weekend"] == True]["demand"].mean()

    labels = ["Weekday", "Weekend"]
    values = [weekday_demand, weekend_demand]

    fig, ax = plt.subplots()

    ax.bar(labels, values, color=["steelblue", "orange"])

    ax.set_ylabel("Average Demand (MW)")
    ax.set_title("Average Electricity Demand")

    ax.grid(axis="y")

    st.pyplot(fig)

# ---------------------------
# Demand Heatmap
# ---------------------------

with tabs[3]:

    st.subheader("Electricity Demand Heatmap (Hour vs Day)")

    heatmap_data = df.pivot_table(
        values="demand",
        index="dayofweek",
        columns="hour",
        aggfunc="mean"
    )

    fig, ax = plt.subplots()

    im = ax.imshow(heatmap_data, aspect="auto")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")

    ax.set_title("Electricity Demand Heatmap")

    plt.colorbar(im)

    st.pyplot(fig)