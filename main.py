from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid

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

# Pydantic v2 Models with Enhanced Validation
class LaundryAnalysisRequest(BaseModel):
    images: List[str] = Field(..., min_items=1, max_items=5, description="Base64 encoded images")
    analysis_type: str = Field(..., pattern="^(clothing|wash_tag)$", description="Type of analysis")
    user_notes: Optional[str] = Field(None, max_length=500, description="Optional user notes")

    @validator('images')
    def validate_images(cls, v):
        """Validate base64 image format"""
        import base64
        for i, img in enumerate(v):
            try:
                # Check if it's valid base64
                decoded = base64.b64decode(img, validate=True)
                # Check minimum size
                if len(decoded) < 100:
                    raise ValueError(f"Image {i+1} is too small")
                # Check maximum size (10MB)
                if len(decoded) > 10 * 1024 * 1024:
                    raise ValueError(f"Image {i+1} is too large (max 10MB)")
            except Exception as e:
                raise ValueError(f"Invalid image format at index {i+1}: {str(e)}")
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

# Enhanced AI Analysis Functions
def get_enhanced_care_label_prompt():
    """Enhanced prompt specifically for decoding care labels"""
    return """You are an expert at reading clothing care labels. Analyze this care label image and decode EACH SPECIFIC SYMBOL you can see.

CRITICAL: Do NOT give generic advice. Decode the actual symbols visible in the image.

For each symbol you can identify, provide the exact meaning:

WASHING SYMBOLS (tub/basin shapes):
- Numbers (30, 40, 60) = Maximum temperature in Celsius
- Hand symbol = Hand wash only
- Crossed out = Do not wash

BLEACHING SYMBOLS (triangles):
- Empty triangle = Bleach allowed
- Crossed out triangle = Do not bleach
- Triangle with "Cl" crossed out = Non-chlorine bleach only

DRYING SYMBOLS (squares):
- Square with circle = Tumble dry (dots indicate heat: 1 dot=low, 2 dots=medium, 3 dots=high)
- Square with lines = Air dry (horizontal line=lay flat, vertical lines=hang dry)
- Crossed out = Do not tumble dry

IRONING SYMBOLS (iron shape):
- Dots indicate temperature (1 dot=low/110°C, 2 dots=medium/150°C, 3 dots=high/200°C)
- Crossed out = Do not iron

DRY CLEANING SYMBOLS (circles):
- Letters (P, F, W) = Professional cleaning type
- Crossed out = Do not dry clean

Look at each symbol in the image and tell me exactly what it means. If you can't see a symbol clearly, say "Symbol not visible" for that category."""

def get_enhanced_clothing_prompt():
    """Enhanced prompt for clothing compatibility analysis"""
    return """You are a professional laundry expert. Analyze these clothing items to determine washing compatibility.

Examine each item for:
1. FABRIC TYPE: Cotton, polyester, wool, silk, denim, delicates, synthetic blends
2. COLOR INTENSITY: Dark colors, bright colors, light colors, white, new items (may bleed)
3. CARE REQUIREMENTS: Temperature needs, gentle vs normal cycle, special treatments
4. FABRIC WEIGHT: Heavy (denim, towels) vs light (delicates, silk)

Provide specific recommendations:
- Can these specific items be washed together safely?
- What exact temperature should be used?
- What cycle is appropriate for these specific fabrics?
- What type of detergent works best?
- Any special precautions needed?

Be specific about why items can or cannot be washed together based on what you observe."""

def analyze_with_openai_enhanced(images: List[str], analysis_type: str, user_notes: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced OpenAI analysis with better error handling and prompts"""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("No OpenAI API key found, using fallback response")
        return get_enhanced_fallback_response(analysis_type)
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Prepare image content with high detail
        image_content = []
        for img_base64 in images[:2]:  # Limit to 2 images for cost efficiency
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                    "detail": "high"
                }
            })
        
        # Select appropriate prompt
        if analysis_type == "wash_tag":
            system_prompt = get_enhanced_care_label_prompt()
        else:
            system_prompt = get_enhanced_clothing_prompt()
        
        # Prepare content with user notes if provided
        content = [{"type": "text", "text": system_prompt}] + image_content
        
        if user_notes:
            content.append({
                "type": "text", 
                "text": f"Additional user notes to consider: {user_notes}"
            })
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2,  # Lower temperature for more consistent responses
            "top_p": 0.9
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            return parse_enhanced_response(ai_response, analysis_type)
        else:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            return get_enhanced_fallback_response(analysis_type)
            
    except Exception as e:
        print(f"AI analysis error: {str(e)}")
        return get_enhanced_fallback_response(analysis_type)

def parse_enhanced_response(ai_response: str, analysis_type: str) -> Dict[str, Any]:
    """Enhanced parsing with better extraction logic"""
    
    ai_lower = ai_response.lower()
    
    # Determine compatibility for clothing
    can_wash_together = True
    if analysis_type == "clothing":
        negative_indicators = ["cannot", "can't", "should not", "separate", "different loads", "not together", "incompatible"]
        if any(indicator in ai_lower for indicator in negative_indicators):
            can_wash_together = False
    
    # Enhanced temperature extraction
    temperature = "cold water"
    temp_patterns = {
        "30°c": "30°C maximum",
        "30 degrees": "30°C maximum", 
        "40°c": "40°C maximum",
        "40 degrees": "40°C maximum",
        "60°c": "60°C maximum", 
        "60 degrees": "60°C maximum",
        "hand wash": "hand wash only",
        "cold water": "cold water",
        "warm water": "warm water",
        "hot water": "hot water"
    }
    
    for pattern, temp_value in temp_patterns.items():
        if pattern in ai_lower:
            temperature = temp_value
            break
    
    # Enhanced cycle extraction
    cycle = "normal"
    if any(word in ai_lower for word in ["gentle", "delicate", "sensitive", "mild"]):
        cycle = "gentle/delicate"
    elif "hand wash" in ai_lower:
        cycle = "hand wash"
    elif "permanent press" in ai_lower:
        cycle = "permanent press"
    
    # Enhanced detergent extraction
    detergent_type = "regular detergent"
    if any(phrase in ai_lower for phrase in ["no bleach", "color-safe", "non-chlorine"]):
        detergent_type = "color-safe detergent (no bleach)"
    elif any(word in ai_lower for word in ["gentle", "mild", "sensitive", "delicate"]):
        detergent_type = "gentle/mild detergent"
    elif "enzyme" in ai_lower:
        detergent_type = "enzyme-based detergent"
    
    # Enhanced instruction extraction
    special_instructions = []
    lines = ai_response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Look for structured instructions
        if any(marker in line for marker in ['•', '-', '1.', '2.', '3.', '4.', '5.', '*']):
            instruction = line
            # Clean markers
            for marker in ['•', '-', '1.', '2.', '3.', '4.', '5.', '*']:
                instruction = instruction.replace(marker, '').strip()
            
            if len(instruction) > 10 and instruction not in special_instructions:
                special_instructions.append(instruction)
        
        # Look for care instructions
        elif any(keyword in line.lower() for keyword in [
            'do not', 'iron at', 'dry clean', 'air dry', 'tumble dry', 
            'wash separately', 'pre-treat', 'inside out'
        ]):
            if len(line) > 10 and line.strip() not in special_instructions:
                special_instructions.append(line.strip())
    
    # Add default instructions if none found
    if not special_instructions:
        if analysis_type == "wash_tag":
            special_instructions = [
                "Follow temperature symbols exactly as shown on label",
                "Check drying method symbols (square shapes) for proper drying",
                "Look for ironing temperature dots if iron symbol present",
                "Observe bleaching restrictions (triangle symbols)"
            ]
        else:
            special_instructions = [
                "Sort by color intensity to prevent bleeding",
                "Check individual care labels for special requirements",
                "Pre-treat stains before washing",
                "Consider fabric weight when loading machine"
            ]
    
    return {
        "can_wash_together": can_wash_together,
        "temperature": temperature,
        "cycle": cycle,
        "detergent_type": detergent_type,
        "special_instructions": special_instructions[:6],  # Limit to 6 instructions
        "reasoning": ai_response,
        "items_analyzed": ["Care label symbols" if analysis_type == "wash_tag" else "Clothing items"]
    }

def get_enhanced_fallback_response(analysis_type: str) -> Dict[str, Any]:
    """Enhanced fallback responses when AI is unavailable"""
    if analysis_type == "wash_tag":
        return {
            "can_wash_together": True,
            "temperature": "follow label temperature",
            "cycle": "as indicated on label", 
            "detergent_type": "mild detergent",
            "special_instructions": [
                "Look for washing temperature number (30°, 40°, 60°) in basin symbol",
                "Check hand symbol in basin - means hand wash only",
                "Find square with circle for tumble dry (dots = heat level)",
                "Look for iron symbol with dots (1 dot=low, 2=medium, 3=high heat)",
                "Triangle symbol shows bleach rules (crossed out = no bleach)",
                "Circle symbol indicates dry cleaning requirements"
            ],
            "reasoning": "Care label analysis: Each symbol on the label provides specific care instructions. Look for washing temperature numbers, drying method symbols (squares), ironing heat levels (iron with dots), and bleaching restrictions (triangles).",
            "items_analyzed": ["Care label symbols"]
        }
    else:
        return {
            "can_wash_together": True,
            "temperature": "cold water",
            "cycle": "normal",
            "detergent_type": "color-safe detergent",
            "special_instructions": [
                "Separate dark colors from light colors to prevent bleeding",
                "Check all garment care labels for special requirements", 
                "Pre-treat any visible stains before washing",
                "Use cold water to preserve colors and prevent shrinking",
                "Consider fabric weight - wash heavy and light items separately",
                "Turn printed or dark items inside out to protect colors"
            ],
            "reasoning": "General laundry guidance: For best results, sort clothing by color and fabric type. Use cold water and color-safe detergent to prevent color bleeding and fabric damage. Always check individual care labels for specific requirements.",
            "items_analyzed": ["Clothing items"]
        }

# API Routes
@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Wash This?? API is running",
        "version": "1.0.0",
        "ai_enabled": bool(os.getenv("OPENAI_API_KEY")),
        "model": "gpt-4o-mini",
        "status": "healthy"
    }

@app.get("/api/")
async def api_root():
    """API root endpoint"""
    return {
        "message": "Wash This?? API is running", 
        "version": "1.0.0",
        "ai_enabled": bool(os.getenv("OPENAI_API_KEY")),
        "model": "gpt-4o-mini",
        "endpoints": [
            "/api/analyze-laundry (POST)",
            "/api/analysis-history (GET)"
        ]
    }

@app.post("/api/analyze-laundry", response_model=LaundryAnalysisResponse)
async def analyze_laundry(request: LaundryAnalysisRequest):
    """Main analysis endpoint with enhanced AI processing"""
    try:
        # Analyze with enhanced AI
        analysis_result = analyze_with_openai_enhanced(
            request.images,
            request.analysis_type, 
            request.user_notes
        )
        
        # Create response with all enhanced features
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        recommendation = WashingRecommendation(
            can_wash_together=analysis_result["can_wash_together"],
            temperature=analysis_result["temperature"],
            cycle=analysis_result["cycle"], 
            detergent_type=analysis_result["detergent_type"],
            special_instructions=analysis_result["special_instructions"],
            reasoning=analysis_result["reasoning"]
        )
        
        response = LaundryAnalysisResponse(
            recommendation=recommendation,
            items_analyzed=analysis_result["items_analyzed"],
            analysis_id=analysis_id,
            timestamp=timestamp,
            confidence_score=0.88 if os.getenv("OPENAI_API_KEY") else 0.75
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analysis-history")
async def get_analysis_history(limit: int = 10):
    """Get analysis history (simplified version without database)"""
    return {
        "analyses": [],
        "message": "History feature ready for database integration",
        "limit": limit
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
