from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid
import base64

# Load environment variables
load_dotenv()

app = FastAPI(title="Wash This?? API", version="1.0.0", description="AI-Powered Laundry Analysis")

# Enhanced CORS middleware for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Enhanced Pydantic Models
class LaundryAnalysisRequest(BaseModel):
    images: List[str] = Field(..., min_length=1, max_length=5, description="Base64 encoded images")
    analysis_type: str = Field(..., description="Type of analysis: clothing or wash_tag")
    user_notes: Optional[str] = Field(None, max_length=500, description="Optional user notes")

    @field_validator('images')
    @classmethod
    def validate_images(cls, v):
        """Validate base64 image format and size"""
        for i, img in enumerate(v):
            try:
                # Check if it's valid base64
                decoded = base64.b64decode(img, validate=True)
                # Check minimum size (1KB)
                if len(decoded) < 1000:
                    raise ValueError(f"Image {i+1} is too small (minimum 1KB)")
                # Check maximum size (15MB)
                if len(decoded) > 15 * 1024 * 1024:
                    raise ValueError(f"Image {i+1} is too large (maximum 15MB)")
            except Exception as e:
                raise ValueError(f"Invalid image format at index {i+1}: {str(e)}")
        return v

    @field_validator('analysis_type')
    @classmethod
    def validate_analysis_type(cls, v):
        if v not in ["clothing", "wash_tag"]:
            raise ValueError("analysis_type must be 'clothing' or 'wash_tag'")
        return v

class WashingRecommendation(BaseModel):
    can_wash_together: bool = Field(..., description="Whether items can be washed together")
    temperature: str = Field(..., description="Recommended water temperature")
    cycle: str = Field(..., description="Recommended wash cycle")
    detergent_type: str = Field(..., description="Recommended detergent type")
    special_instructions: List[str] = Field(..., description="Special care instructions")
    separate_loads: Optional[List[Dict[str, Any]]] = Field(None, description="Separate load recommendations")
    reasoning: str = Field(..., description="AI reasoning for recommendations")

class LaundryAnalysisResponse(BaseModel):
    recommendation: WashingRecommendation
    items_analyzed: List[str]
    analysis_id: str
    timestamp: datetime
    confidence_score: float = Field(default=0.85, description="Analysis confidence level")

# Enhanced AI Analysis Functions using Google Gemini
def get_enhanced_care_label_prompt():
    """Advanced care label analysis prompt for Gemini"""
    return """You are a professional laundry care expert. Analyze this care label image and decode EACH SPECIFIC SYMBOL you can see clearly.

CRITICAL: Provide SPECIFIC symbol meanings, not generic advice.

SYMBOL DECODING:

WASHING (basin symbols):
- Number in basin (30, 40, 60) = "Machine wash at [X]°C maximum"
- Hand in basin = "Hand wash only"
- Basin with X = "Do not wash"
- Lines under basin = gentleness level

BLEACHING (triangles):
- Empty triangle = "Bleach allowed"
- Triangle with X = "Do not bleach"
- Triangle with Cl crossed = "Non-chlorine bleach only"

DRYING (squares):
- Square with circle = "Tumble dry" (dots = heat level)
- Square with line = "Air dry method"
- Square with X = "Do not tumble dry"

IRONING (iron symbols):
- Iron + dots = temperature (1=low, 2=medium, 3=high)
- Iron with X = "Do not iron"

DRY CLEANING (circles):
- Circle with letter = professional cleaning type
- Circle with X = "Do not dry clean"

Analyze the image and decode each visible symbol with its specific meaning."""

def get_enhanced_clothing_prompt():
    """Advanced clothing compatibility analysis prompt for Gemini"""
    return """You are a professional laundry expert. Analyze these clothing items for washing compatibility.

ANALYSIS CRITERIA:

FABRIC TYPES:
- Cotton: durable, handles higher temps
- Polyester: lower temps, may pill
- Wool: gentle care, may shrink
- Silk: delicate, often hand wash
- Denim: heavy, may bleed
- Delicates: gentle treatment needed

COLOR ANALYSIS:
- New dark items: high bleeding risk
- Bright colors: medium bleeding risk
- Light colors: victim of bleeding
- White: shows stains, can bleach

COMPATIBILITY CHECK:
1. Will colors bleed onto each other?
2. Do fabrics need similar temperatures?
3. Do items need same cycle intensity?
4. Will textures damage each other?

Provide specific recommendations:
- Exact temperature and cycle
- Detergent type needed
- Items that must be separated and why
- Special precautions needed

Be specific about what you observe in the images."""

def analyze_with_gemini(images: List[str], analysis_type: str, user_notes: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced analysis using Google Gemini Vision"""
    
    # Try Emergent LLM Key first (supports Gemini)
    EMERGENT_LLM_KEY = os.getenv("EMERGENT_LLM_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    api_key = EMERGENT_LLM_KEY or GOOGLE_API_KEY
    
    if not api_key:
        print("WARNING: No Gemini/Google API key found, using enhanced fallback response")
        return get_enhanced_fallback_response(analysis_type)
    
    try:
        # Use Emergent LLM endpoint if available, otherwise Google directly
        if EMERGENT_LLM_KEY:
            return analyze_with_emergent_gemini(images, analysis_type, user_notes, EMERGENT_LLM_KEY)
        else:
            return analyze_with_google_gemini(images, analysis_type, user_notes, GOOGLE_API_KEY)
            
    except Exception as e:
        print(f"Gemini analysis error: {str(e)}")
        return get_enhanced_fallback_response(analysis_type)

def analyze_with_emergent_gemini(images: List[str], analysis_type: str, user_notes: Optional[str], api_key: str) -> Dict[str, Any]:
    """Analysis using Emergent LLM key with Gemini"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Select prompt
        if analysis_type == "wash_tag":
            prompt = get_enhanced_care_label_prompt()
        else:
            prompt = get_enhanced_clothing_prompt()
        
        # Add user notes if provided
        if user_notes and user_notes.strip():
            prompt += f"\n\nUser notes: {user_notes.strip()}"
        
        # Prepare request for Emergent endpoint
        payload = {
            "model": "gemini-1.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{images[0]}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        # Use Emergent LLM endpoint
        response = requests.post(
            "https://api.emergentmethods.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print(f"Emergent Gemini API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            print("Successfully received Emergent Gemini analysis")
            return parse_enhanced_response(ai_response, analysis_type, confidence=0.90)
        else:
            print(f"Emergent API error: {response.status_code} - {response.text}")
            return get_enhanced_fallback_response(analysis_type)
            
    except Exception as e:
        print(f"Emergent Gemini error: {str(e)}")
        return get_enhanced_fallback_response(analysis_type)

def analyze_with_google_gemini(images: List[str], analysis_type: str, user_notes: Optional[str], api_key: str) -> Dict[str, Any]:
    """Analysis using Google Gemini API directly"""
    try:
        # Select prompt
        if analysis_type == "wash_tag":
            prompt = get_enhanced_care_label_prompt()
        else:
            prompt = get_enhanced_clothing_prompt()
        
        # Add user notes if provided
        if user_notes and user_notes.strip():
            prompt += f"\n\nUser notes: {user_notes.strip()}"
        
        # Prepare request for Google Gemini
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": images[0]
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 1000,
                "temperature": 0.1
            }
        }
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        print(f"Google Gemini API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["candidates"][0]["content"]["parts"][0]["text"]
            print("Successfully received Google Gemini analysis")
            return parse_enhanced_response(ai_response, analysis_type, confidence=0.88)
        else:
            print(f"Google Gemini API error: {response.status_code} - {response.text}")
            return get_enhanced_fallback_response(analysis_type)
            
    except Exception as e:
        print(f"Google Gemini error: {str(e)}")
        return get_enhanced_fallback_response(analysis_type)

def parse_enhanced_response(ai_response: str, analysis_type: str, confidence: float = 0.85) -> Dict[str, Any]:
    """Enhanced parsing with advanced pattern recognition"""
    
    ai_lower = ai_response.lower()
    
    # Enhanced compatibility detection for clothing
    can_wash_together = True
    if analysis_type == "clothing":
        negative_indicators = [
            "cannot", "can't", "should not", "do not wash together", 
            "separate", "different loads", "incompatible", "not safe",
            "will bleed", "may damage", "wash separately"
        ]
        if any(indicator in ai_lower for indicator in negative_indicators):
            can_wash_together = False
    
    # Advanced temperature extraction
    temperature = "cold water"
    temp_patterns = [
        (["30°c", "30 degrees", "30°", " 30 "], "30°C maximum"),
        (["40°c", "40 degrees", "40°", " 40 "], "40°C maximum"),
        (["60°c", "60 degrees", "60°", " 60 "], "60°C maximum"),
        (["90°c", "90 degrees", "90°", " 90 "], "90°C maximum"),
        (["hand wash only", "hand washing"], "hand wash only"),
        (["cold water", "cold wash"], "cold water (30°C or below)"),
        (["warm water", "warm wash"], "warm water (40°C)"),
        (["hot water", "hot wash"], "hot water (60°C)")
    ]
    
    for patterns, temp_value in temp_patterns:
        if any(pattern in ai_lower for pattern in patterns):
            temperature = temp_value
            break
    
    # Enhanced cycle detection
    cycle = "normal"
    cycle_patterns = [
        (["gentle cycle", "delicate cycle", "gentle wash", "delicate wash"], "gentle/delicate"),
        (["hand wash", "hand washing"], "hand wash"),
        (["permanent press", "perm press"], "permanent press"),
        (["heavy duty", "heavy cycle"], "heavy duty"),
        (["quick wash", "speed wash"], "quick wash")
    ]
    
    for patterns, cycle_value in cycle_patterns:
        if any(pattern in ai_lower for pattern in patterns):
            cycle = cycle_value
            break
    
    # Enhanced detergent detection
    detergent_type = "regular detergent"
    detergent_patterns = [
        (["no bleach", "color-safe", "non-chlorine", "bleach-free"], "color-safe detergent (no bleach)"),
        (["gentle detergent", "mild detergent", "sensitive skin"], "gentle/mild detergent"),
        (["enzyme detergent", "enzyme-based"], "enzyme-based detergent"),
        (["wool detergent", "delicate detergent"], "specialized delicate detergent"),
        (["heavy duty detergent"], "heavy-duty detergent")
    ]
    
    for patterns, detergent_value in detergent_patterns:
        if any(pattern in ai_lower for pattern in patterns):
            detergent_type = detergent_value
            break
    
    # Advanced instruction extraction
    special_instructions = []
    lines = ai_response.split('\n')
    
    # Look for structured instructions
    for line in lines:
        line = line.strip()
        
        if not line or len(line) < 15:
            continue
            
        # Look for instruction markers
        instruction_markers = ['•', '-', '1.', '2.', '3.', '4.', '5.', '*', '→', '▪']
        has_marker = any(marker in line for marker in instruction_markers)
        
        # Look for instruction keywords
        instruction_keywords = [
            'wash at', 'do not', 'iron at', 'dry clean', 'air dry', 'tumble dry',
            'hand wash', 'separate', 'pre-treat', 'inside out', 'maximum', 'only'
        ]
        has_keyword = any(keyword in line.lower() for keyword in instruction_keywords)
        
        if has_marker or has_keyword:
            # Clean up the instruction
            instruction = line
            for marker in instruction_markers:
                instruction = instruction.replace(marker, '').strip()
            
            # Remove numbering
            import re
            instruction = re.sub(r'^\d+\.\s*', '', instruction)
            
            if len(instruction) > 15 and instruction not in special_instructions:
                special_instructions.append(instruction)
    
    # Add default instructions if needed
    if len(special_instructions) < 3:
        default_instructions = get_default_instructions(analysis_type, temperature, cycle)
        for instruction in default_instructions:
            if instruction not in special_instructions:
                special_instructions.append(instruction)
    
    # Limit to most important instructions
    special_instructions = special_instructions[:6]
    
    return {
        "can_wash_together": can_wash_together,
        "temperature": temperature,
        "cycle": cycle,
        "detergent_type": detergent_type,
        "special_instructions": special_instructions,
        "reasoning": ai_response,
        "items_analyzed": get_items_analyzed(analysis_type),
        "confidence_score": confidence
    }

def get_default_instructions(analysis_type: str, temperature: str, cycle: str) -> List[str]:
    """Get contextual default instructions"""
    if analysis_type == "wash_tag":
        return [
            f"Follow the temperature guideline: {temperature}",
            f"Use {cycle} cycle as indicated by symbols",
            "Check for drying method symbols (squares)",
            "Look for ironing temperature indicators",
            "Observe bleaching restrictions (triangles)",
            "Follow dry cleaning symbols if present"
        ]
    else:
        return [
            "Sort clothing by color intensity",
            "Check individual care labels",
            "Pre-treat visible stains",
            "Turn dark items inside out",
            "Avoid overloading the machine"
        ]

def get_items_analyzed(analysis_type: str) -> List[str]:
    """Get appropriate items analyzed description"""
    if analysis_type == "wash_tag":
        return ["Care label symbols and instructions"]
    else:
        return ["Clothing items and fabric compatibility"]

def get_enhanced_fallback_response(analysis_type: str) -> Dict[str, Any]:
    """Enhanced fallback responses with detailed guidance"""
    if analysis_type == "wash_tag":
        return {
            "can_wash_together": True,
            "temperature": "follow care label symbols",
            "cycle": "as indicated by symbols",
            "detergent_type": "mild detergent",
            "special_instructions": [
                "🌡️ Look for temperature numbers (30°, 40°, 60°) in washing symbol",
                "✋ Hand symbol in basin means hand wash only",
                "🔄 Square with circle = tumble dry, lines = air dry",
                "🔥 Iron dots show heat: 1=low, 2=medium, 3=high",
                "🚫 Triangle crossed out = no bleach allowed",
                "💼 Circle symbols indicate dry cleaning requirements"
            ],
            "reasoning": "Care label symbol guide: Numbers show max temperature, hand symbols require hand washing, square symbols show drying method, iron dots indicate heat level, triangles show bleach rules, circles indicate dry cleaning needs.",
            "items_analyzed": ["Care label symbols"],
            "confidence_score": 0.75
        }
    else:
        return {
            "can_wash_together": True,
            "temperature": "cold water (30°C or below)",
            "cycle": "normal",
            "detergent_type": "color-safe detergent",
            "special_instructions": [
                "🔄 Separate dark, medium, and light colors",
                "🏷️ Check individual garment care labels",
                "🧽 Pre-treat stains before washing",
                "❄️ Use cold water to prevent color bleeding",
                "↩️ Turn dark items inside out",
                "⚖️ Don't overload washing machine"
            ],
            "reasoning": "General laundry best practices: Sort by color to prevent bleeding, use cold water to protect fabrics, check care labels for special requirements, pre-treat stains for best results.",
            "items_analyzed": ["Clothing items"],
            "confidence_score": 0.75
        }

# API Routes
@app.get("/")
async def root():
    """API health check"""
    emergent_key = bool(os.getenv("EMERGENT_LLM_KEY"))
    google_key = bool(os.getenv("GOOGLE_API_KEY"))
    
    return {
        "message": "Wash This?? API is running",
        "version": "1.0.0",
        "ai_enabled": emergent_key or google_key,
        "ai_provider": "Gemini (Google)" if emergent_key or google_key else "None",
        "status": "healthy"
    }

@app.get("/api/")
async def api_root():
    """API root endpoint"""
    emergent_key = bool(os.getenv("EMERGENT_LLM_KEY"))
    google_key = bool(os.getenv("GOOGLE_API_KEY"))
    
    return {
        "message": "Wash This?? API is running",
        "version": "1.0.0",
        "ai_enabled": emergent_key or google_key,
        "ai_provider": "Gemini via Emergent" if emergent_key else ("Gemini Direct" if google_key else "Fallback"),
        "endpoints": [
            "POST /api/analyze-laundry",
            "GET /api/analysis-history",
            "GET /health"
        ]
    }

@app.post("/api/analyze-laundry", response_model=LaundryAnalysisResponse)
async def analyze_laundry(request: LaundryAnalysisRequest):
    """Enhanced analysis endpoint with Gemini AI"""
    try:
        print(f"Received analysis request: {request.analysis_type}, {len(request.images)} images")
        
        # Analyze with Gemini
        analysis_result = analyze_with_gemini(
            request.images,
            request.analysis_type,
            request.user_notes
        )
        
        # Create response
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Handle separate loads for clothing
        separate_loads = None
        if request.analysis_type == "clothing" and not analysis_result["can_wash_together"]:
            separate_loads = [
                {
                    "load_number": 1,
                    "items": ["Dark colored items"],
                    "settings": f"{analysis_result['temperature']}, {analysis_result['cycle']} cycle"
                },
                {
                    "load_number": 2,
                    "items": ["Light colored items"], 
                    "settings": "Cold water, normal cycle"
                }
            ]
        
        recommendation = WashingRecommendation(
            can_wash_together=analysis_result["can_wash_together"],
            temperature=analysis_result["temperature"],
            cycle=analysis_result["cycle"],
            detergent_type=analysis_result["detergent_type"],
            special_instructions=analysis_result["special_instructions"],
            separate_loads=separate_loads,
            reasoning=analysis_result["reasoning"]
        )
        
        response = LaundryAnalysisResponse(
            recommendation=recommendation,
            items_analyzed=analysis_result["items_analyzed"],
            analysis_id=analysis_id,
            timestamp=timestamp,
            confidence_score=analysis_result.get("confidence_score", 0.85)
        )
        
        print(f"Analysis completed with confidence: {response.confidence_score}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analysis-history")
async def get_analysis_history(limit: int = 10):
    """Get analysis history"""
    return {
        "analyses": [],
        "message": "History feature ready for database integration",
        "limit": limit
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    emergent_key = bool(os.getenv("EMERGENT_LLM_KEY"))
    google_key = bool(os.getenv("GOOGLE_API_KEY"))
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_provider": "Gemini via Emergent" if emergent_key else ("Gemini Direct" if google_key else "Fallback Only"),
        "version": "1.0.0",
        "features": {
            "ai_analysis": emergent_key or google_key,
            "care_label_decoding": True,
            "clothing_analysis": True,
            "enhanced_parsing": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
