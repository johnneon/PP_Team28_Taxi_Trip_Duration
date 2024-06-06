import joblib
import numpy as np
import pandas as pd
import os

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import cluster
from sklearn import feature_selection

# Функция определения расстояние Хаверсина между точкой, 
# в которой был включен счетчик, и точкой, в которой счетчик был выключен;
def get_haversine_distance(lat1, lng1, lat2, lng2):
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # радиус земли в километрах
    EARTH_RADIUS = 6371
    # считаем кратчайшее расстояние h по формуле Хаверсина
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

# Функция определения направление движения из точки, в которой был включен счетчик, 
# в точку, в которой счетчик был выключен.
def get_angle_direction(lat1, lng1, lat2, lng2):
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # считаем угол направления движения alpha по формуле угла пеленга
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha


# Функцию add_geographical_features() принимает на вход таблицу с данными
# о поездках и возвращает обновленную таблицу с добавленными в нее 2 столбцами:
# haversine_distance - расстояние Хаверсина между точкой,
# в которой был включен счетчик, и точкой, в которой счетчик был выключен;
# direction - направление движения из точки, в которой был включен счетчик,
# в точку, в которой счетчик был выключен.
def add_geographical_features(target: pd.DataFrame):
    df = target.copy()
    df['haversine_distance'] = get_haversine_distance(target['pickup_latitude'], target['pickup_longitude'], target['dropoff_latitude'], target['dropoff_longitude'])
    df['direction'] = get_angle_direction(target['pickup_latitude'], target['pickup_longitude'], target['dropoff_latitude'], target['dropoff_longitude'])
    return df


# Функция add_cluster_features() принимает на вход таблицу с данными о поездках 
# и обученный алгоритм кластеризации. Функция возвращает обновленную таблицу 
# с добавленными в нее столбцом geo_cluster - географический кластер, 
# к которому относится поездка.
def add_cluster_features(target: pd.DataFrame, model: cluster.KMeans):
    df = target.copy()
    df['geo_cluster'] = model.predict(df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].to_numpy())
    return df


# Функция для слияния таблиц погоды и данных о поездках
def add_weather_features(target: pd.DataFrame, weather: pd.DataFrame):
    df = target.copy()
    df = df.merge(weather, how='left', left_on=['pickup_date',	'pickup_hour'], right_on=['date', 'hour'])
    df.drop(columns=['date', 'hour'], inplace=True)
    return df

# Функция fill_null_weather_data(),принимает на вход таблицу с данными о поездках. 
# Функция заполняет пропущенные значения в столбцах.
# Пропуски в столбцах с погодными условиями - temperature, visibility, wind speed, precip
# заполняются медианным значением температуры, влажности, скорости ветра 
# и видимости в зависимости от даты начала поездки.
def fill_null_weather_data(target: pd.DataFrame):
  df = target.copy()
  weather_cols = ['temperature', 'visibility', 'wind speed', 'precip', 'total_distance', 'total_travel_time', 'number_of_steps']

  for col in weather_cols:
    df[col] = df[col].fillna(df.groupby('pickup_date')[col].transform('median'))


  df['events'] = df['events'].fillna('None')

  return df

def preprocessing_data(
        taxi_data: pd.DataFrame, 
        holiday_data: pd.DataFrame, 
        osrm_data: pd.DataFrame,
        weather_data: pd.DataFrame,
        ) -> pd.DataFrame:
        
    # Переведите признак pickup_datetime в тип данных datetime с форматом 
    # год-месяц-день час:минута:секунда (в функции pd.to_datetime() 
    # параметр format='%Y-%m-%d %H:%M:%S').

    taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    
    taxi_data['pickup_date'] = taxi_data['pickup_datetime'].dt.date
    taxi_data['pickup_hour'] = taxi_data['pickup_datetime'].dt.hour
    taxi_data['pickup_day_of_week'] = taxi_data['pickup_datetime'].dt.weekday

    # Преобразуем столбец 'pickup_date' таблицы taxi_data в формат datetime
    taxi_data['pickup_date'] = pd.to_datetime(taxi_data['pickup_date'], format='%Y-%m-%d')

    # Преобразуем столбец 'date' таблицы holiday_data в формат datetime
    holiday_data['date'] = pd.to_datetime(holiday_data['date'], format='%Y-%m-%d')

    # Смержим в новый датафрейм таблицы taxi_data и holiday_data на основании даты
    taxi_data = taxi_data.merge(holiday_data['date'], how='left', left_on='pickup_date', right_on='date')

    # Добавим новый столбец 'pickup_holiday', в котором будет 1, если в столбце 'date' есть дата, иначе 0.
    taxi_data['pickup_holiday'] = taxi_data['date'].apply(lambda x: 0 if pd.isnull(x) else 1)

    # Удалим теперь ненужный нам столбец 'date'
    taxi_data.drop('date', axis= 1 , inplace= True)

    # Смержим в новый датафрейм таблицы taxi_data и osrm_data
    taxi_data = taxi_data.merge(
        osrm_data[['id', 'total_distance', 'total_travel_time', 'number_of_steps']],
        how='left', left_on='id', right_on='id'
        )

    # Добавляем два столбца: 
    # haversine_distance - расстояние Хаверсина между точкой, 
    # в которой был включен счетчик, и точкой, в которой счетчик был выключен;
    # direction - направление движения из точки, в которой был включен счетчик,
    # в точку, в которой счетчик был выключен.
    taxi_data = add_geographical_features(taxi_data)
    
    # создаем обучающую выборку из географических координат всех точек
    coords = np.hstack((taxi_data[['pickup_latitude', 'pickup_longitude']],
                        taxi_data[['dropoff_latitude', 'dropoff_longitude']]))
    # обучаем алгоритм кластеризации
    kmeans = cluster.KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(coords)
    
    taxi_data = add_cluster_features(taxi_data, kmeans)
    
    joblib.dump(kmeans, os.path.join(os.path.dirname(__file__), "model/kmeans.pkl"))
    
    # Выделим интересующие нас столбцы
    needle_weather_data = weather_data[['temperature', 'visibility', 'wind speed', 'precip', 'events', 'date', 'hour']].copy()
    # Приведем поля date и hour к типас даты/инта для последующего слияния таблиц
    needle_weather_data['date'] = pd.to_datetime(needle_weather_data['date'], format='%Y-%m-%d')
    needle_weather_data['hour'].astype(np.int32)
    
    # Выполним слияние таблиц:
    taxi_data = add_weather_features(taxi_data, needle_weather_data)

    # Заполним пропуски
    taxi_data = fill_null_weather_data(taxi_data)

    # Проверим выброс по длительности, преобразовав столбец с секундами к часам:
    trip_duration_in_hours = taxi_data['trip_duration'] / 60 / 60

    # Вычмслим выбросы по скорости:
    avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6
    
    # Зачищаем выбросы из данных
    taxi_data.drop(taxi_data[trip_duration_in_hours > 24].index, inplace=True)
    taxi_data.drop(taxi_data[avg_speed > 300].index, inplace=True)
    
    # Логорифмируем признак длительности поездки
    taxi_data['trip_duration_log'] = np.log(taxi_data['trip_duration']+1)

    train_data = taxi_data.copy()
    
    # Сразу позаботимся об очевидных неинформативных и избыточных признаках.
    drop_columns = ['id', 'dropoff_datetime']
    train_data = train_data.drop(drop_columns, axis=1)

    # Ранее мы извлекли всю необходимую для нас информацию из даты начала поездки, 
    # теперь мы можем избавиться от этих признаков, 
    # так как они нам больше не понадобятся:
    drop_columns = ['pickup_datetime', 'pickup_date']
    train_data = train_data.drop(drop_columns, axis=1)


    # Закодируем признак vendor_id в таблице train_data таким образом, 
    # чтобы он был равен 0, если идентификатор таксопарка равен 1, 
    # и 1 — в противном случае.
    # Закодируем признак store_and_fwd_flag в таблице train_data таким образом, 
    # чтобы он был равен 0, если флаг выставлен в значение 'N', 
    # и 1 — в противном случае.
    train_data['vendor_id'] = train_data['vendor_id'].map(lambda x: 0 if x == 1 else 1)
    train_data['store_and_fwd_flag'] = train_data['store_and_fwd_flag'].map(
                                                lambda x: 0 if x == 'N' else 1
                                                )

    # Выполним one_hot кодирование

    # определяем колонки, которые будем преобразовывать
    column_to_coding = ['pickup_day_of_week', 'geo_cluster', 'events']

    # определяем энкодер
    one_hot_encoder = preprocessing.OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    # обучаем
    one_hot_encoder.fit(train_data[column_to_coding])
    # сохраняем
    

    # преобразуе колонки в новый датафрэйм
    data_onehot = pd.DataFrame(one_hot_encoder.transform(train_data[column_to_coding]), 
                               columns = one_hot_encoder.get_feature_names_out(column_to_coding)
                               )
    
    # Добавим полученную таблицу с закодированными признаками:
    train_data = pd.concat(
        [train_data.reset_index(drop=True).drop(column_to_coding, axis=1), data_onehot],
        axis=1
        )
    
    joblib.dump(one_hot_encoder, os.path.join(os.path.dirname(__file__), "model/one_hot_encoder.pkl"))

    #  Сформируем матрицу наблюдений X, вектор целевой переменной y и его логарифм y_log.
    X = train_data.drop(['trip_duration', 'trip_duration_log'], axis=1)
    y = train_data['trip_duration']
    y_log = train_data['trip_duration_log']

   
    
    # Выбранный тип валидации - hold-out. Разобьем выборку на обучающую и валидационную в соотношении 67/33:
    X_train, X_valid, y_train_log, y_valid_log = model_selection.train_test_split(
        X, y_log,
        test_size=0.33,
        random_state=42
        )

    # С помощью SelectKBest отберём 25 признаков, наилучшим образом подходящих для предсказания 
    # целевой переменной в логарифмическом масштабе.
    # определяем объект реализующий KBest
    skb = feature_selection.SelectKBest(feature_selection.f_regression, k=25)
    # обучаем функцию на тренировочных данных и сразу преобразуем их
    X_train = pd.DataFrame(skb.fit_transform(X_train,y_train_log),columns = skb.get_feature_names_out())
    # X_train.info()
    # преобразуем проверочные данные
    X_valid = pd.DataFrame(skb.transform(X_valid),columns = skb.get_feature_names_out())

    # Нормализуем предикторы в обучающей и валидационной выборках с помощью MinMaxScaler из библиотеки sklearn. 
    # определяем объект реализующий mim-max преобразование
    min_max_scaler = preprocessing.MinMaxScaler()
    # обучаем функцию на тренировочных данных и сразу преобразуем их
    X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train),
                       columns = min_max_scaler.get_feature_names_out())
    # преобразуем проверочные данные
    X_valid = pd.DataFrame(min_max_scaler.transform(X_valid),
                              columns = min_max_scaler.get_feature_names_out())
    joblib.dump(min_max_scaler, os.path.join(os.path.dirname(__file__), "model/min_max_scaler.pkl"))
    
    

    return X_train, X_valid, y_train_log, y_valid_log



def to_teach_and_separate_data():
    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "train")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "train"))
    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "test")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "test"))
    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "model")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "model"))
    
    taxi_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset/train.csv')) 
    holiday_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset/holiday_data.csv'), sep = ';') 
    osrm_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset/osrm_data_train.csv'))
    weather_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset/weather_data.csv'))
    
    X_train, X_val, y_train, y_val = preprocessing_data(
                                                taxi_data,
                                                holiday_data,
                                                osrm_data,
                                                weather_data
                                                )

  

    X_train.to_csv(os.path.join(os.path.dirname(__file__), 'train/X_train.csv'), index=False)
    y_train.to_csv(os.path.join(os.path.dirname(__file__), 'train/y_train.csv'), index=False)

    X_val.to_csv(os.path.join(os.path.dirname(__file__), 'test/X_val.csv'), index=False)
    y_val.to_csv(os.path.join(os.path.dirname(__file__), 'test/y_val.csv'), index=False)


if __name__ == '__main__':
    to_teach_and_separate_data()