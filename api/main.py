import base64
import mimetypes
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.pipeline import LuxuryPipeline, bytes_to_data_url  # adjust import

app = FastAPI()
pipe = LuxuryPipeline()  # reads OPENAI_API_KEY from env

# CORS: allow your dev + prod frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:5173",
        "http://127.0.0.1:5173",

        # Production frontend
        "https://appraiseai.co",
        "https://ai-appraisal-suite.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    image_bytes = await image.read()
    mime = image.content_type or "image/png"
    data_url = bytes_to_data_url(image_bytes, mime=mime)

    out = pipe.run([data_url], max_results=10)

    ident = out["identification"]
    listings = out.get("listings", {}).get("results", [])

    # Shape the response for your frontend demo
    return {
        "predicted_price": float(ident["estimated_market_value_usd"]),
        "currency": "USD",
        "confidence": float(ident["confidence"]),
        "brand": ident["brand"],
        "model": ident["model"],
        "category": ident["category"],
        "similar_listings": [
            {
                "id": str(i),
                "title": r.get("title", ""),
                "price_text": r.get("price_text", ""),
                "url": r.get("url", ""),
                "source": r.get("source", ""),
            }
            for i, r in enumerate(listings)
        ],
    }
