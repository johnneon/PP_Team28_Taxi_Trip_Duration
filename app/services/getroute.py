import requests
import json
import polyline
import folium

def get_route(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    loc = "{},{};{},{}".format(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    url = "http://router.project-osrm.org/route/v1/driving/"
    options = "?steps=true"
    r = requests.get(url + loc + options)
    if r.status_code!= 200:
        return {}
    res = r.json()
    routes = polyline.decode(res['routes'][0]['geometry'])
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']
    time = res['routes'][0]['duration']
    number_of_steps = len(res['routes'][0]['legs'][0]['steps'])

    return routes, start_point, end_point, distance, time, number_of_steps