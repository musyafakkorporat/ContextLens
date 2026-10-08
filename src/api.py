from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contextlens_pipeline import analyze_text


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="ContextLens API",
    description="API untuk menganalisis konteks dan kualitas informasi.",
    version="1.0.0"
)


# =========================
# REQUEST MODEL
# =========================

class AnalyzeRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Teks yang ingin dianalisis."
    )


# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def root():

    return {
        "message": "ContextLens API berjalan."
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================
# ANALYZE ENDPOINT
# =========================

@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    try:

        result = analyze_text(
            request.text
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
