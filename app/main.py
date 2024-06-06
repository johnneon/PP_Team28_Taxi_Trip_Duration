from fastapi import FastAPI
from app.routing.base import router as base_router
from app.routing.predict import router as predict_router


def start_application() -> FastAPI:
    application = FastAPI()

    application.include_router(base_router)
    application.include_router(predict_router)

    return application


app = start_application()