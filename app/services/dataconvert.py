import joblib
import datetime
import holidays 
from meteostat import Point
from meteostat import Hourly

import numpy as np
import os
import pandas as pd
from sklearn import cluster

from geopy.geocoders import ArcGIS #Подключаем библиотеку
# для геокодирования используем ArcGIS

from app.services.getroute import get_route
from app.ml.data_preprocessing import get_haversine_distance # Функция определения расстояние Хаверсина



# Функция add_cluster_features() принимает на вход таблицу с данными о поездках 
# и обученный алгоритм кластеризации. Функция возвращает обновленную таблицу 
# с добавленными в нее столбцом geo_cluster - географический кластер, 
# к которому относится поездка.
def add_cluster_features(target: pd.DataFrame, model: cluster.KMeans):
    df = target.copy()
    df['geo_cluster'] = model.predict(df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].to_numpy())
    return df



def dataconvert(data):
    """ Converting data into a dataset for prediction """

    geolocator = ArcGIS()
    # начинаем формировать датафрейм исходных данных
    
    # Получаем координаты (признак)
    departure_address = str(data.get('departure_address'))
    departure_address = departure_address + ", USA"
    destination_address = str(data.get('destination_address'))
    destination_address = destination_address + ", USA"
    departure_location = geolocator.geocode(departure_address)
    destination_location = geolocator.geocode(destination_address)
    passenger_count = int(data.get('passenger_count'))
    date_start = str(data.get('date'))
    store_and_fwd_flag = int(data.get('store_and_fwd_flag'))
    df = pd.DataFrame({'vendor_id': [0,1],
                       'passenger_count': [passenger_count, passenger_count ],
                       'pickup_longitude': [departure_location.longitude, departure_location.longitude],
                       'pickup_latitude': [departure_location.latitude, departure_location.latitude],
                       'dropoff_longitude': [destination_location.longitude, destination_location.longitude],
                       'dropoff_latitude': [destination_location.latitude, destination_location.latitude],
                       'store_and_fwd_flag': [store_and_fwd_flag,store_and_fwd_flag]
                       })
    
    
    # Выделяем время начала поезки
    pickup_time =  datetime.datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')

    # Добавляем в датафрейм час начала поездки (признак)
    pickup_hour = pickup_time.hour
    df["pickup_hour"] = [pickup_hour, pickup_hour]
    
    # Определяем праздничный ли это день
    USA_holidays_list = holidays.US() 
    date1 = pickup_time.date() 
    j = date1 in USA_holidays_list 
    
    # Присваиваем pickup_holiday 1, если день - праздник,  иначе 0 (признак)
    if j : 
        pickup_holiday=1 
    else: 
        pickup_holiday=0
    # Добавляем pickup_holiday в датафрейм 
    df["pickup_holiday"] = [pickup_holiday, pickup_holiday]
    
    # Получаем данные OSRM для маршрута и добавляем в датафрейм
    route, start_point, end_point, total_distance, total_travel_time, number_of_steps=get_route(departure_location.longitude,
                                                                                                departure_location.latitude,
                                                                                                destination_location.longitude,
                                                                                                destination_location.latitude)
    
    df["total_distance"] = [total_distance, total_distance]
    df["total_travel_time"] = [total_travel_time, total_travel_time]
    df["number_of_steps"] = [number_of_steps, number_of_steps]

    # Вычисляем расстояние Хаверсина (признак)
    haversine_distance = get_haversine_distance(departure_location.latitude,
                                                departure_location.longitude,
                                                destination_location.latitude,
                                                destination_location.longitude)
    df["haversine_distance"] = [haversine_distance, haversine_distance]
    # Получаем погоду за день
    start = datetime.datetime(pickup_time.year, pickup_time.month, pickup_time.day, pickup_time.hour)
    end = datetime.datetime(pickup_time.year, pickup_time.month, pickup_time.day, pickup_time.hour)
    place = Point(40.7143, -74.006)
    
    weather_data = Hourly(place, start, end)
    weather_data = weather_data.fetch()
    # получаем температуру (признак)
    temperature =  weather_data['temp'].iloc[0]
    print(temperature)
    # температуру в датафрейм
    df["temperature"] = [temperature, temperature]
    # Определяем номер дня недели и добавляем в датафрэйм
    pickup_day_of_week = date1.weekday()
    df["pickup_day_of_week"] = [pickup_day_of_week, pickup_day_of_week]
    
    # Загружаем предобученную модель kmeans и определяем географический кластер 
    with open(os.path.join(os.path.dirname(__file__), "../ml/model/kmeans.pkl"), 'rb') as file:
        kmeans = joblib.load(file)
        df['geo_cluster'] =  kmeans.predict(df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].to_numpy())
    
    # При обучении модели one_hot кодирования использовался признак 'events' - событие погоды
    # поэтому добавим временный буферный признак
    df['events'] = ['Rain','Rain']
    # Определяем коллонки для one_hot кодирования
    column_to_coding = ['pickup_day_of_week', 'geo_cluster', 'events']
    # Загружаем предобученную модель kmeans и определяем географический кластер 
    with open(os.path.join(os.path.dirname(__file__), "../ml/model/one_hot_encoder.pkl"), 'rb') as file:
        one_hot_encoder = joblib.load(file)
        data_onehot = pd.DataFrame(one_hot_encoder.transform(df[column_to_coding]), columns = one_hot_encoder.get_feature_names_out(column_to_coding))    
    # Убираем лишние колонки
    data_onehot.drop(['geo_cluster_2',  'geo_cluster_3', 'geo_cluster_6', 'geo_cluster_9', 'events_None', 'events_Rain', 'events_Snow'], axis=1, inplace= True)
    # Объединяем таблицы в одну
    df = pd.concat([df.reset_index(drop=True).drop(column_to_coding, axis=1), data_onehot], axis=1)
    # Загрузим подготовленный MinMaxScaler и преобразуем данные
    with open(os.path.join(os.path.dirname(__file__), "../ml/model/min_max_scaler.pkl"), 'rb') as file:
        min_max_scaler = joblib.load(file)
        df = pd.DataFrame(min_max_scaler.transform(df),
                              columns = min_max_scaler.get_feature_names_out())

    # возвращаем подготовленные данные для предсказания    
    return df, route, start_point, end_point, total_distance