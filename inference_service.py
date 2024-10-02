from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional, Union
import logging
import numpy as np
from tensorflow import keras
import redis
import json
from datetime import datetime
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ad Recommendation API with A/B Testing")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis setup
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
    logger.info("Connected to Redis successfully")
except redis.ConnectionError:
    logger.warning("Failed to connect to Redis. Proceeding without caching.")
    redis_client = None

# Load models
try:
    user_model_a = keras.models.load_model('user_model_a.keras')
    ad_model_a = keras.models.load_model('ad_model_a.keras')
    user_model_b = keras.models.load_model('user_model_b.keras')
    ad_model_b = keras.models.load_model('ad_model_b.keras')
    logger.info("Models loaded successfully")
    print("User Model A Summary:")
    user_model_a.summary()
    print("\nAd Model A Summary:")
    ad_model_a.summary()
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

class UserFeatures(BaseModel):
    age: int
    gender: str
    country: str
    account_age: int
    follower_count: int
    avg_daily_usage: float
    interests: Union[List[str], None] = []

class AdFeatures(BaseModel):
    category: str
    duration: int
    is_skippable: bool
    ad_quality_score: float
    advertiser_rating: float
    target_age_min: int
    target_age_max: int
    target_interests: Union[List[str], None] = []

class RecommendationRequest(BaseModel):
    user: UserFeatures
    ads: List[AdFeatures]

class RecommendationResponse(BaseModel):
    ad_scores: List[float]
    explanation: List[str]
    variant: str


def preprocess_user_features(user: UserFeatures):
    user_vector = [
        user.age / 100,  # Normalize age
        1 if user.gender == 'M' else 0,
        hash(user.country) % 10 / 10,  # Normalize country hash
        user.account_age / 3650,  # Normalize account age (assume max 10 years)
        np.log1p(user.follower_count) / 10,  # Normalize follower count with log
        user.avg_daily_usage / 1440,  # Normalize daily usage (minutes in a day)
    ] + [1 if interest in user.interests else 0 for interest in ['tech', 'fashion', 'sports', 'food', 'travel']]
    
    # Add two more features to make it 13
    user_vector.extend([0, 0])  # You can replace these with meaningful features if available
    
    return np.array(user_vector).reshape(1, -1)  # Reshape to (1, 13)

def preprocess_ad_features(ad: AdFeatures):
    ad_vector = [
        hash(ad.category) % 10 / 10,  # Normalize category hash
        float(ad.duration) / 120,  # Normalize duration (assume max 2 minutes)
        1 if ad.is_skippable else 0,
        float(ad.ad_quality_score) / 10,  # Normalize ad quality score
        float(ad.advertiser_rating) / 5,  # Normalize advertiser rating
        float(ad.target_age_min) / 100,  # Normalize target age min
        float(ad.target_age_max) / 100,  # Normalize target age max
    ] + [1 if interest in ad.target_interests else 0 for interest in ['tech', 'fashion', 'sports', 'food', 'travel']]
    
    # Add one more feature to make it 13
    ad_vector.append(0)  # You can replace this with a meaningful feature if available
    
    return np.array(ad_vector)
def get_variant(user_id: str):
    if redis_client is None:
        return random.choice(["A", "B"])
    
    variant = redis_client.get(f"user_variant:{user_id}")
    if variant is None:
        variant = random.choice(["A", "B"])
        redis_client.set(f"user_variant:{user_id}", variant)
    return variant.decode() if isinstance(variant, bytes) else variant

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_ads(request: RecommendationRequest):
    try:
        logger.info(f"Received recommendation request: {request}")

        user_id = hash(json.dumps(request.user.dict()))
        variant = get_variant(user_id)
        
        user_model = user_model_a if variant == "A" else user_model_b
        ad_model = ad_model_a if variant == "A" else ad_model_b
        
        user_vector = preprocess_user_features(request.user)
        ad_vectors = np.array([preprocess_ad_features(ad) for ad in request.ads])
        
        logger.info(f"User vector shape: {user_vector.shape}")
        logger.info(f"Ad vectors shape: {ad_vectors.shape}")
        
        user_embedding = user_model.predict(user_vector)
        ad_embeddings = ad_model.predict(ad_vectors)
        
        logger.info(f"User embedding shape: {user_embedding.shape}")
        logger.info(f"Ad embeddings shape: {ad_embeddings.shape}")
        
        # Calculate similarity scores
        scores = np.sum(user_embedding * ad_embeddings, axis=1)
        
        explanations = generate_explanations(request.user, request.ads, scores)
        
        log_recommendation(user_id, request.ads, scores, variant)
        
        logger.info(f"Generated recommendations: {scores.tolist()}")
        return RecommendationResponse(ad_scores=scores.tolist(), explanation=explanations, variant=variant)
    except Exception as e:
        logger.error(f"Error in recommend_ads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
def generate_explanations(user: UserFeatures, ads: List[AdFeatures], scores: List[float]):
    explanations = []
    for i, (score, ad) in enumerate(zip(scores, ads)):
        if score > 0.7:
            explanations.append(f"Ad {i+1} strongly matches your interests in {', '.join(set(user.interests) & set(ad.target_interests))}")
        elif score > 0.3:
            explanations.append(f"Ad {i+1} partially matches your profile")
        else:
            explanations.append(f"Ad {i+1} may not be relevant to your interests")
    return explanations

def log_recommendation(user_id: int, ads: List[AdFeatures], scores: List[float], variant: str):
    if redis_client is None:
        return
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "variant": variant,
        "recommendations": [{"ad_id": hash(json.dumps(ad.dict())), "score": float(score)} for ad, score in zip(ads, scores)]
    }
    redis_client.lpush("recommendation_logs", json.dumps(log_entry))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": "user_model_a" in globals()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)