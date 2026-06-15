from fastapi import FastAPI

app = FastAPI(title="GST Copilot API")

@app.get("/health")
def health_check():
    return {"status": "ok"}