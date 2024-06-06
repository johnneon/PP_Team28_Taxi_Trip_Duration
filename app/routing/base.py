from fastapi import APIRouter

router = APIRouter(prefix="")


@router.get("/")
async def root():
    """Empty method, just for test is app running"""

    return {"message": "Hi! For test, you should send POST request "
                       "to the `/predict` endpoint,with following parameters:",
            "body": {
                "departure_address": "124 E 73rd St, New York, NY 10021",
                "latitude": "124 E 73rd St, New York, NY 10021",
                "store_and_fwd_flag": 0,
                "date": "2016-03-14 17:24:55"
            }}
