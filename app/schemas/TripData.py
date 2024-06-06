from pydantic import BaseModel


class TripData(BaseModel):
    passenger_count: int
    departure_address: str
    destination_address: str
    store_and_fwd_flag: int
    date: str
