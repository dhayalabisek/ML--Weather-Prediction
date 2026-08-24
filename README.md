# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
To implement a Random Forest Regression model to predict daily temperature, PM2.5 pollution level, and energy-related solar radiation (tsr) using environmental sensor data.

Dataset

The dataset contains environmental sensor measurements such as:

hum – Humidity
co2 – CO₂ level
illumination – Illumination level
pressure – Atmospheric pressure
pm10 – PM10 pollution level
wind_direction_angle – Wind direction
wind_speed – Wind speed
wind_speed_level – Wind speed category

Target variables:

tem – Temperature
pm2_5 – PM2.5 pollution level
tsr – Total solar radiation (used as the energy-related target)


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the environmental sensor dataset and select sensor readings as input features.

2.Select tem, pm2_5, and tsr as the target variables.

3.Split the data, train a Random Forest Regressor, and evaluate it using MAE and R².

4.Give new sensor values as input and predict temperature, PM2.5, and energy.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: DHAYAL ABISEK R
RegisterNumber:  212225060061 
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("/content/weather-station-eee-block_2024_07_13.csv")

# Input features
features = [
    "hum",
    "co2",
    "illumination",
    "pressure",
    "pm10",
    "wind_direction_angle",
    "wind_speed",
    "wind_speed_level"
]

# Target variables
targets = ["tem", "pm2_5", "tsr"]

# Keep only required columns
df = df[features + targets]

# Remove missing values only from required columns
df = df.dropna()

# Input and output
X = df[features]
Y = df[targets]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Metrics
print("Temperature MAE =", mean_absolute_error(Y_test["tem"], Y_pred[:, 0]))
print("PM2.5 MAE =", mean_absolute_error(Y_test["pm2_5"], Y_pred[:, 1]))
print("Energy MAE =", mean_absolute_error(Y_test["tsr"], Y_pred[:, 2]))

print("Temperature R2 =", r2_score(Y_test["tem"], Y_pred[:, 0]))
print("PM2.5 R2 =", r2_score(Y_test["pm2_5"], Y_pred[:, 1]))
print("Energy R2 =", r2_score(Y_test["tsr"], Y_pred[:, 2]))

# New sensor data
new_data = pd.DataFrame({
    "hum": [90],
    "co2": [500],
    "illumination": [300],
    "pressure": [1000],
    "pm10": [30],
    "wind_direction_angle": [180],
    "wind_speed": [2],
    "wind_speed_level": [1]
})

# Prediction
prediction = model.predict(new_data)

print("\nPredicted Temperature =", prediction[0][0])
print("Predicted PM2.5 =", prediction[0][1])
print("Predicted Energy =", prediction[0][2])
```

## Output:
```
Temperature MAE = 0.9942800000000014
PM2.5 MAE = 3.5460499999999997
Energy MAE = 13.287575000000015
Temperature R2 = 0.8484825860034412
PM2.5 R2 = 0.994891666805256
Energy R2 = 0.9863773698706328

Predicted Temperature = 26.44199999999999
Predicted PM2.5 = 21.85
Predicted Energy = 0.289
```
## Result:
The Random Forest Regression model was successfully implemented to predict temperature, PM2.5 pollution level, and energy-related solar radiation from environmental sensor data. The model performance was evaluated using Mean Absolute Error (MAE) and R² score.
