import json
import datetime
import requests
import streamlit as st
import pandas as pd
import folium
import streamlit.components.v1 as components

# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon

def serialize_datetime(obj): 
    if isinstance(obj, datetime.datetime): 
        return obj.isoformat() 
    raise TypeError("Type not serializable") 

# Заголок
st.title('Прогноз времени поездки')

# Запрашиваем исходные данные
# Число пассажиров
passenger_count  = st.text_input(
    "Введите число пассажиров 👇",
    value = 1
    )

# Адрес отправления
departure_address = st.text_input(
    "Введите адрес отправления 👇",
    value = "124 E 73rd St, New York, NY 10021"
    )
# Адрес назначения
destination_address = st.text_input(
    "Введите адрес назначения 👇",
    value = "142-06 Foch Blvd, South Ozone Park, NY 11436", 
    )
agree = st.checkbox("Я согласен на запись данных о поездке")
if agree:
     store_and_fwd_flag =  1
else:
     store_and_fwd_flag =  0
# Запрашиваем дату и время поездки
date_trip = st.date_input("Введите день отправления 👇", value="today")
time_tripe = st.time_input("Введите время отправления 👇", value="now")
date = datetime.datetime(date_trip.year,
                         date_trip.month,
                         date_trip.day,
                         time_tripe.hour,
                         time_tripe.minute)
st.write("Дата/время отправления:", date)
date_str = date.strftime("%Y-%m-%d %H:%M:%S")


if (st.button('Рассчитать время в пути')):
    res = {
            'passenger_count': passenger_count,
            'departure_address': departure_address,
            'destination_address': destination_address,
            'store_and_fwd_flag': store_and_fwd_flag,
            'date': date_str,
            }
    # отправляем данные, получаем предсказания
    requestpost = requests.post(url="http://127.0.0.1:8000/predict", data=json.dumps(res))
    response_data = requestpost.json()
    #st.write("Время поездки на такси компании Джафар", response)
    predict_result = response_data['result']['trip_duration']
    route = response_data['route']
    start_point = response_data['start_point']
    end_point = response_data['end_point']
    total_distance = d = round(response_data['total_distance']/1000, 3)
    st.write("Расстояние:", total_distance, ' км')
    st.write("Время поездки на такси компании Алладин", str(datetime.timedelta(seconds=predict_result['0'])))
    st.write("Время поездки на такси компании Джафар", str(datetime.timedelta(seconds=predict_result['1'])))
    
    # Наносим на карту
    
    #st.write("Время поездки на такси компании Джафар", start_point)


    # Создаём карту
    New_York_map = folium.Map(
        location = [40.70134, -73.88988],    # широта и Нью Йорка
        zoom_start = 10
        )                    

    # Создаём группы
    departure_destination_point = folium.FeatureGroup(
                                        name='Departure and destination point',
                                        show=True
                                        )
    # Сохдаём маркеры точек
    departure_point = folium.Marker(
                        [start_point[0], start_point[1]],
                        popup = ("Departure location: "+ departure_address),
                        icon=folium.Icon(color='green',icon='ok-sign')
                        ).add_to(New_York_map)

    destination_point = folium.Marker(
                        [end_point[0], end_point[1]],
                        popup = ("Destination location: "+ destination_address),
                        icon=folium.Icon(color='red',icon='ok-sign')
                        ).add_to(New_York_map)
    
    folium.PolyLine(route,weight=8,color='blue',opacity=0.6).add_to(New_York_map)
    
    components.html(folium.Figure().add_child(New_York_map).render(), height=500) 
