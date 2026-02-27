import io
import os
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.logger import get_logger

load_dotenv()
logger = get_logger("api")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="BLIP Caption Generator API",
    description="Upload an image and get an AI-generated caption",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    status_code=429, content={"detail": "Rate limit exceeded. Max 5 requests/minute."}
))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid or missing API Key.")

# Model state (loaded once at startup, reused for all requests)
processor = None
model = None
device = None


@app.on_event("startup")
def load_model():
    global processor, model, device

    MODEL_DIR = "saved_models/fine_tuned_blip"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {MODEL_DIR} on {device}...")

    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    logger.info("Model loaded and ready.")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/caption", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def generate_caption(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG or BMP."
        )

    # Read and decode image
    image_bytes = await file.read()

    # File size check — reject files larger than 5MB
    MAX_FILE_SIZE = 5 * 1024 * 1024
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max size is 5MB.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # Run inference
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    logger.info(f"Caption generated: {caption}")
    return JSONResponse(content={"caption": caption})
