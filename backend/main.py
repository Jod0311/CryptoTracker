"""FastAPI backend for the Crypto ML application."""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    """Root endpoint returning a welcome message."""
    return {"message": "Welcome to the Crypto ML Backend!"}
