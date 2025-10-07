from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
from datetime import datetime
import uuid

app = FastAPI(title="Wash This API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models
class LaundryRequest(BaseModel):
    images: List[str]
    analysis_type: str
    user_notes: Optional[str] = None

class WashingRecommendation(BaseModel):
    can_wash_together: bool
    temperature: str
    cycle: str
    detergent_type: str
    special_instructions: List[str]
    reasoning: str

class LaundryResponse(BaseModel):
    recommendation: WashingRecommendation
    items_analyzed: List[str]

def analyze_with_ai(images: List[str], analysis_type: str) -> dict:
    """Analyze with OpenAI or provide fallback"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return get_fallback_response(analysis_type)
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Use first image only
        image_content = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{images[0]}"
            }
        }]
        
        if analysis_type == "wash_tag":
            prompt = "Analyze this care label and decode each symbol. Provide specific washing instructions for each symbol you can see (temperature, drying, ironing, etc.)."
        else:
            prompt = "Analyze these clothing items and determine if they can be washed together. Provide specific washing recommendations."
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + image_content
            }],
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_text = result["choices"][0]["message"]["content"]
            return parse_response(ai_text, analysis_type)
        else:
            return get_fallback_response(analysis_type)
            
    except Exception as e:
        print(f"AI error: {e}")
        return get_fallback_response(analysis_type)

def parse_response(text: str, analysis_type: str) -> dict:
    """Parse AI response"""
    
    text_lower = text.lower()
    
    # Determine compatibility
    can_wash_together = True
    if analysis_type == "clothing" and any(word in text_lower for word in ["cannot", "separate", "different"]):
        can_wash_together = False
    
    # Extract temperature
    temperature = "cold"
    if "30°" in text or "30 degrees" in text_lower:
        temperature = "30°C maximum"
    elif "40°" in text or "40 degrees" in text_lower:
        temperature = "40°C maximum"
    elif "warm" in text_lower:
        temperature = "warm"
    elif "hand wash" in text_lower:
        temperature = "hand wash only"
    
    # Extract cycle
    cycle = "normal"
    if "gentle" in text_lower or "delicate" in text_lower:
        cycle = "gentle"
    elif "hand" in text_lower:
        cycle = "hand wash"
    
    # Extract detergent
    detergent = "regular detergent"
    if "no bleach" in text_lower or "color-safe" in text_lower:
        detergent = "color-safe detergent"
    elif "mild" in text_lower or "gentle" in text_lower:
        detergent = "mild detergent"
    
    # Extract instructions
    instructions = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and any(marker in line for marker in ['•', '-', '1.', '2.']):
            clean_line = line.replace('•', '').replace('-', '').strip()
            if len(clean_line) > 10:
                instructions.append(clean_line)
    
    if not instructions:
        if analysis_type == "wash_tag":
            instructions = [
                "Follow temperature symbols on label",
                "Check drying instructions",
                "Look for ironing guidelines",
                "Observe bleaching restrictions"
            ]
        else:
            instructions = [
                "Separate by color",
                "Check care labels",
                "Pre-treat stains",
                "Use appropriate temperature"
            ]
    
    return {
        "can_wash_together": can_wash_together,
        "temperature": temperature,
        "cycle": cycle,
        "detergent_type": detergent,
        "special_instructions": instructions[:5],
        "reasoning": text,
        "items_analyzed": ["Care label" if analysis_type == "wash_tag" else "Clothing items"]
    }

def get_fallback_response(analysis_type: str) -> dict:
    """Fallback when AI unavailable"""
    if analysis_type == "wash_tag":
        return {
            "can_wash_together": True,
            "temperature": "follow label symbols",
            "cycle": "as indicated",
            "detergent_type": "mild detergent",
            "special_instructions": [
                "Look for temperature number (30°, 40°, 60°)",
                "Check for hand wash symbol",
                "Find drying method symbols",
                "Look for iron temperature dots",
                "Check bleach restrictions"
            ],
            "reasoning": "Care label symbols provide specific instructions. Look for washing temperature, drying method, ironing heat, and bleach guidelines.",
            "items_analyzed": ["Care label"]
        }
    else:
        return {
            "can_wash_together": True,
            "temperature": "cold water",
            "cycle": "normal",
            "detergent_type": "color-safe detergent", 
            "special_instructions": [
                "Separate dark and light colors",
                "Check individual care labels",
                "Pre-treat stains",
                "Use cold water to prevent bleeding"
            ],
            "reasoning": "Sort by color and fabric type. Use cold water and color-safe detergent for best results.",
            "items_analyzed": ["Clothing items"]
        }

@app.get("/")
def root():
    return {
        "message": "Wash This API is running",
        "version": "1.0.0",
        "ai_enabled": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.get("/api/")
def api_root():
    return {
        "message": "Wash This API is running",
        "version": "1.0.0", 
        "ai_enabled": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.post("/api/analyze-laundry")
def analyze_laundry(request: LaundryRequest):
    try:
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        result = analyze_with_ai(request.images, request.analysis_type)
        
        recommendation = WashingRecommendation(
            can_wash_together=result["can_wash_together"],
            temperature=result["temperature"],
            cycle=result["cycle"],
            detergent_type=result["detergent_type"],
            special_instructions=result["special_instructions"],
            reasoning=result["reasoning"]
        )
        
        return LaundryResponse(
            recommendation=recommendation,
            items_analyzed=result["items_analyzed"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analysis-history")
def get_history():
    return {
        "analyses": [],
        "message": "History available with database"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
