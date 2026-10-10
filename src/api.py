import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from contextlens_pipeline import analyze_text


# =========================
# LOGGING
# =========================

logger = logging.getLogger(__name__)


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="ContextLens API",
    description=(
        "API untuk menganalisis konteks "
        "dan kualitas informasi."
    ),
    version="1.1.0",
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# =========================
# REQUEST VALIDATION
# =========================

class AnalyzeRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description=(
            "Teks yang ingin dianalisis. "
            "Panjang maksimum 5000 karakter."
        ),
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Teks tidak boleh kosong."
            )

        return value


# =========================
# ROOT ENDPOINT
# =========================

@app.get("/")
def root():

    return {
        "message": "ContextLens API berjalan."
    }


# =========================
# HEALTH CHECK
# =========================

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
        return analyze_text(request.text)

    except Exception:
        logger.exception(
            "Terjadi kesalahan saat menganalisis teks."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Analisis gagal diproses. "
                "Periksa log backend."
            ),
        ) from None
