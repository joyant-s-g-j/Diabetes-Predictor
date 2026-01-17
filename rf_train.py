import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error

city_attr = pd.read_csv("dataset/city_attributes.csv")
humidity = pd.read_csv("dataset/humidity.csv")
pressure = pd.read_csv("dataset/pressure.csv")
temperature = pd.read_csv("dataset/temperature.csv")
weather_desc = pd.read_csv("dataset/weather_description.csv")
wind_dir = pd.read_csv("dataset/wind_direction.csv")
wind_speed = pd.read_csv("dataset/wind_speed.csv")

def melt_df(df, val_name):
  return df.melt(id_vars=['datetime'], var_name='city', value_name=val_name)

temp = melt_df(temperature, "temperature")
humi = melt_df(humidity, "humidity")
pres = melt_df(pressure, "pressure")
wthr_desc = melt_df(weather_desc, "weather_description")
wind_d = melt_df(wind_dir, "wind_direction")
wind_s = melt_df(wind_speed, "wind_speed")

df = temp.merge(humi, on=['datetime', 'city']) \
         .merge(pres, on=['datetime', 'city']) \
         .merge(wthr_desc, on=['datetime', 'city']) \
         .merge(wind_d, on=['datetime', 'city']) \
         .merge(wind_s, on=['datetime', 'city'])

df['temperature'] = df['temperature'] - 273.15
df = df[df['temperature'].between(-50, 60)]

X = df.drop(columns=['temperature','datetime','city'], axis=1)
y = df['temperature']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
)

