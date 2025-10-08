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
    images: List[str] = Field(..., min_items=1, max_items=5, description="Base64 encoded images")
    analysis_type: str = Field(..., description="Type of analysis: clothing or wash_tag")
    user_notes: Optional[str] = Field(None, max_length=500, description="Optional user notes")

    @validator('images')
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

    @validator('analysis_type')
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

# Enhanced AI Analysis Functions
def get_enhanced_care_label_prompt():
    """Advanced care label analysis prompt"""
    return """You are a professional laundry care expert analyzing clothing care labels. Your task is to decode EACH SPECIFIC SYMBOL visible in the care label image.

CRITICAL INSTRUCTIONS:
- Do NOT provide generic advice like "follow the symbols" or "read the label"
- DECODE each symbol you can actually see in the image
- Be specific about what each symbol means

SYMBOL DECODING GUIDE:

🔹 WASHING SYMBOLS (basin/tub shapes):
- Number in basin (30, 40, 60) = "Machine wash at [X]°C maximum temperature"
- Hand in basin = "Hand wash only - do not machine wash"
- Basin with X = "Do not wash with water"
- One line under basin = "Gentle/delicate cycle required"
- Two lines under basin = "Very gentle cycle required"

🔹 BLEACHING SYMBOLS (triangle shapes):
- Empty triangle = "Bleach allowed when needed"
- Triangle with "Cl" crossed out = "Non-chlorine bleach only"
- Triangle with X = "Do not bleach - no bleach products"

🔹 DRYING SYMBOLS (square shapes):
- Square with circle = "Tumble dry allowed"
- Square with circle + 1 dot = "Tumble dry low heat (60°C)"
- Square with circle + 2 dots = "Tumble dry medium heat (80°C)"
- Square with circle + 3 dots = "Tumble dry high heat (80°C)"
- Square with vertical line = "Line dry - hang vertically"
- Square with horizontal line = "Flat dry - lay flat to dry"
- Square with curved line = "Drip dry while wet"
- Square with X = "Do not tumble dry"

🔹 IRONING SYMBOLS (iron shapes):
- Iron + 1 dot = "Iron at low temperature (110°C maximum)"
- Iron + 2 dots = "Iron at medium temperature (150°C maximum)"  
- Iron + 3 dots = "Iron at high temperature (200°C maximum)"
- Iron with X = "Do not iron"
- Iron with steam crossed out = "Do not steam iron"

🔹 DRY CLEANING SYMBOLS (circle shapes):
- Circle with "P" = "Professional dry clean with perchloroethylene"
- Circle with "F" = "Professional dry clean with petroleum solvents" 
- Circle with "W" = "Professional wet clean"
- Circle with X = "Do not dry clean"

ANALYSIS TASK:
Look carefully at the care label image and identify each symbol you can see clearly. For each symbol, provide the exact meaning using the guide above. If a symbol is unclear or partially hidden, mention that specific symbol is "not clearly visible."

Provide a comprehensive analysis of all visible symbols with their specific meanings."""

def get_enhanced_clothing_prompt():
    """Advanced clothing compatibility analysis prompt"""
    return """You are a professional laundry expert with 20+ years of experience. Analyze the clothing items in these images for washing compatibility.

ANALYSIS FRAMEWORK:

🔹 FABRIC IDENTIFICATION:
- Cotton: Durable, can handle higher temperatures
- Polyester/Synthetic: Lower temperatures, may pill with rough fabrics
- Wool: Requires gentle care, may shrink or felt
- Silk: Delicate, often requires hand washing
- Denim: Heavy, may bleed dye, needs sturdy cycle
- Delicates: Lace, thin materials need gentle treatment
- Mixed blends: Follow most restrictive care requirement

🔹 COLOR ANALYSIS:
- New dark items (especially red, black, navy): High bleeding risk
- Bright colors: Medium bleeding risk, separate from lights
- Light colors/pastels: Victim of color bleeding
- White items: Show stains easily, can be bleached if 100% cotton
- Faded items: Lower bleeding risk

🔹 FABRIC WEIGHT & TEXTURE:
- Heavy items (jeans, towels, sweatshirts): Need sturdy cycle
- Light items (t-shirts, undergarments): Can be damaged by heavy items
- Delicate textures: Can snag on zippers, buttons, rough fabrics

🔹 COMPATIBILITY ASSESSMENT:
Determine if items can wash together by evaluating:
1. Will colors bleed onto each other?
2. Do fabrics need similar water temperatures?
3. Do items need same cycle intensity?
4. Will textures damage each other?

PROVIDE SPECIFIC RECOMMENDATIONS:
- Exact temperature (cold/30°C/40°C/60°C)
- Specific cycle (gentle/normal/heavy duty)
- Detergent type needed
- Any items that must be separated and why
- Special precautions for fabric protection

Be detailed and specific based on what you observe in the images."""

def analyze_with_openai_enhanced(images: List[str], analysis_type: str, user_notes: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced OpenAI analysis with advanced prompts and error handling"""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("WARNING: No OpenAI API key found, using enhanced fallback response")
        return get_enhanced_fallback_response(analysis_type)
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Prepare image content (up to 3 images for better analysis)
        image_content = []
        for i, img_base64 in enumerate(images[:3]):
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                    "detail": "high"  # High detail for better symbol recognition
                }
            })
        
        # Select enhanced prompt based on analysis type
        if analysis_type == "wash_tag":
            system_prompt = get_enhanced_care_label_prompt()
        else:
            system_prompt = get_enhanced_clothing_prompt()
        
        # Prepare message content
        content = [{"type": "text", "text": system_prompt}] + image_content
        
        # Add user notes if provided
        if user_notes and user_notes.strip():
            content.append({
                "type": "text", 
                "text": f"\nADDITIONAL USER CONTEXT: {user_notes.strip()}\nPlease consider this information in your analysis."
            })
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 1200,  # Increased for detailed responses
            "temperature": 0.1,   # Very low for consistent, factual responses
            "top_p": 0.9
        }
        
        print(f"Making OpenAI API request for {analysis_type} analysis...")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60  # Increased timeout for complex analysis
        )
        
        print(f"OpenAI API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            print("Successfully received AI analysis")
            return parse_enhanced_response(ai_response, analysis_type, confidence=0.92)
        elif response.status_code == 401:
            print("OpenAI API authentication failed")
            return get_enhanced_fallback_response(analysis_type)
        elif response.status_code == 429:
            print("OpenAI API rate limit exceeded")
            return get_enhanced_fallback_response(analysis_type)
        else:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            return get_enhanced_fallback_response(analysis_type)
            
    except requests.exceptions.Timeout:
        print("OpenAI API timeout")
        return get_enhanced_fallback_response(analysis_type)
    except requests.exceptions.RequestException as e:
        print(f"OpenAI API request error: {str(e)}")
        return get_enhanced_fallback_response(analysis_type)
    except Exception as e:
        print(f"Unexpected error in AI analysis: {str(e)}")
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
    
    # Advanced temperature extraction with specific patterns
    temperature = "cold water"
    temp_patterns = [
        (["30°c", "30 degrees celsius", "30°", " 30 "], "30°C maximum"),
        (["40°c", "40 degrees celsius", "40°", " 40 "], "40°C maximum"),
        (["60°c", "60 degrees celsius", "60°", " 60 "], "60°C maximum"),
        (["90°c", "90 degrees celsius", "90°", " 90 "], "90°C maximum"),
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
        
        # Skip empty lines and headers
        if not line or len(line) < 15:
            continue
            
        # Look for instruction markers
        instruction_markers = ['•', '-', '1.', '2.', '3.', '4.', '5.', '6.', '*', '→', '▪']
        has_marker = any(marker in line for marker in instruction_markers)
        
        # Look for instruction keywords
        instruction_keywords = [
            'wash at', 'do not', 'iron at', 'dry clean', 'air dry', 'tumble dry',
            'hand wash', 'separate', 'pre-treat', 'inside out', 'zip up', 'button up',
            'remove', 'check', 'avoid', 'use only', 'maximum', 'minimum'
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
            
            # Add if it's substantial and unique
            if len(instruction) > 15 and instruction not in special_instructions:
                special_instructions.append(instruction)
    
    # Add default instructions if none found or too few
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
            "Check for specific drying method symbols (squares with lines or circles)",
            "Look for ironing temperature dots (1=low, 2=medium, 3=high heat)",
            "Observe any bleaching restrictions shown by triangle symbols",
            "Follow dry cleaning symbols if present (circles with letters)"
        ]
    else:
        return [
            "Sort clothing by color intensity before washing",
            "Check all individual care labels for special requirements",
            "Pre-treat any visible stains with appropriate stain remover", 
            "Turn dark or printed items inside out to protect colors",
            "Zip up zippers and fasten buttons to maintain shape",
            "Avoid overloading the machine for proper cleaning and care"
        ]

def get_items_analyzed(analysis_type: str) -> List[str]:
    """Get appropriate items analyzed description"""
    if analysis_type == "wash_tag":
        return ["Care label symbols and washing instructions"]
    else:
        return ["Clothing items and fabric compatibility"]

def get_enhanced_fallback_response(analysis_type: str) -> Dict[str, Any]:
    """Enhanced fallback responses with detailed guidance"""
    if analysis_type == "wash_tag":
        return {
            "can_wash_together": True,
            "temperature": "follow care label temperature symbols",
            "cycle": "as indicated by care symbols",
            "detergent_type": "mild detergent recommended",
            "special_instructions": [
                "🌡️ Temperature: Look for numbers in basin symbol (30°, 40°, 60°C)",
                "✋ Hand wash: Hand symbol in basin means hand wash only",
                "🔄 Drying: Square with circle = tumble dry, square with lines = air dry",
                "🔥 Iron heat: Dots on iron symbol (1 dot=low, 2=medium, 3=high)",
                "🚫 Bleach: Triangle crossed out = no bleach allowed",
                "💼 Dry clean: Circle symbol shows professional cleaning needs"
            ],
            "reasoning": "Care label symbol guide: Each symbol provides specific care instructions. Numbers in washing symbols indicate maximum temperature, dots on iron symbols show heat levels, and triangle symbols indicate bleaching rules. Always follow the most restrictive instruction when multiple symbols are present.",
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
                "🔄 Sort by color: Separate dark, medium, and light colored items",
                "🏷️ Check labels: Read individual care labels for special requirements",
                "🧽 Pre-treat stains: Address stains before washing for best results", 
                "❄️ Use cold water: Prevents color bleeding and fabric shrinkage",
                "↩️ Turn inside out: Protect prints and dark colors from fading",
                "⚖️ Don't overload: Allow items to move freely for proper cleaning"
            ],
            "reasoning": "General laundry best practices: Sort clothing by color intensity and fabric type to prevent damage. Cold water protects colors and prevents shrinking. Always check individual care labels as they provide specific requirements for each garment. Pre-treating stains improves cleaning results.",
            "items_analyzed": ["Clothing items"],
            "confidence_score": 0.75
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
        "status": "healthy",
        "features": ["enhanced_care_labels", "clothing_analysis", "advanced_parsing"]
    }

@app.get("/api/")
async def api_root():
    """API root endpoint with feature information"""
    return {
        "message": "Wash This?? API is running",
        "version": "1.0.0",
        "ai_enabled": bool(os.getenv("OPENAI_API_KEY")),
        "model": "gpt-4o-mini",
        "endpoints": {
            "analyze": "POST /api/analyze-laundry",
            "history": "GET /api/analysis-history",
            "health": "GET /health"
        },
        "features": {
            "care_label_decoding": "Advanced symbol recognition and interpretation",
            "clothing_analysis": "Fabric compatibility and washing recommendations", 
            "ai_powered": "GPT-4 Vision for accurate image analysis",
            "fallback_system": "Comprehensive guidance when AI unavailable"
        }
    }

@app.post("/api/analyze-laundry", response_model=LaundryAnalysisResponse)
async def analyze_laundry(request: LaundryAnalysisRequest):
    """Enhanced analysis endpoint with comprehensive AI processing"""
    try:
        print(f"Received analysis request: {request.analysis_type}, {len(request.images)} images")
        
        # Enhanced AI analysis
        analysis_result = analyze_with_openai_enhanced(
            request.images,
            request.analysis_type,
            request.user_notes
        )
        
        # Create comprehensive response
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Handle separate loads for clothing analysis
        separate_loads = None
        if request.analysis_type == "clothing" and not analysis_result["can_wash_together"]:
            # Create separate load suggestions based on reasoning
            separate_loads = [
                {
                    "load_number": 1,
                    "items": ["Dark colored items"],
                    "settings": f"Cold water, {analysis_result['cycle']} cycle, color-safe detergent"
                },
                {
                    "load_number": 2, 
                    "items": ["Light colored items"],
                    "settings": "Cold water, normal cycle, regular detergent"
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
        
        print(f"Analysis completed successfully with confidence: {response.confidence_score}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analysis-history")
async def get_analysis_history(limit: int = 10):
    """Get analysis history (ready for database integration)"""
    return {
        "analyses": [],
        "message": "History feature ready for database integration",
        "limit": limit,
        "total_count": 0
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for monitoring"""
    api_key_status = "configured" if os.getenv("OPENAI_API_KEY") else "missing"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_key_status": api_key_status,
        "version": "1.0.0",
        "features": {
            "ai_analysis": bool(os.getenv("OPENAI_API_KEY")),
            "care_label_decoding": True,
            "clothing_analysis": True,
            "enhanced_parsing": True
        }
    }

# Error handlers
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return {
        "detail": "Validation error - please check your request format",
        "errors": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
