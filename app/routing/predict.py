from fastapi import APIRouter
from app.schemas.TripData import TripData
from app.services.predict import predict as predict_fn
from app.services.dataconvert import dataconvert as dataconvert_fn


router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("")
async def predict(request: TripData):
    """Method for predicting house price"""

    def prepare(data):
        res = {
            'passenger_count': data.passenger_count,
            'departure_address': data.departure_address,
            'destination_address': data.destination_address,
            'store_and_fwd_flag': data.store_and_fwd_flag,
            'date': data.date
            }
        
        return res
    
    trip_data_convert, route, start_point, end_point, total_distance = dataconvert_fn(prepare(request))
    result = predict_fn(trip_data_convert)
    response_predict = {'result': result,
                        'route': route,
                        'start_point': start_point,
                        'end_point':  end_point,
                        'total_distance': total_distance
                        }
    return response_predict
