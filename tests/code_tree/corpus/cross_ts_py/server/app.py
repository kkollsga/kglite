from fastapi import FastAPI

app = FastAPI()


@app.post("/api/session")
def create_session():
    return {}


@app.get("/api/unused")
def unused():
    return {}
