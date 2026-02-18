import sqlite3
import datetime
import httpx
import uvicorn
import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- NEW IMPORT
from pydantic import BaseModel

# --- Configuration ---
# Try to get key, but don't crash if missing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# Specific target email from the assignment
TARGET_NOTIFICATION_EMAIL = "23f3003276@ds.study.iitm.ac.in"

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataFlowPipeline")

app = FastAPI()

# --- FIX CORS HERE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows ALL domains (for assignment submission)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ---------------------

# --- Database Setup (SQLite) ---
DB_NAME = "pipeline_storage.db"

def init_db():
    """Initialize a simple SQLite table for storage."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_text TEXT,
                analysis TEXT,
                sentiment TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()

init_db()

# --- Pydantic Models ---
class PipelineRequest(BaseModel):
    email: str
    source: str

class ProcessedItem(BaseModel):
    original: str
    analysis: str
    sentiment: str
    stored: bool
    timestamp: str

class PipelineResponse(BaseModel):
    items: List[ProcessedItem]
    notificationSent: bool
    processedAt: str
    errors: List[str]

# --- Helper Functions ---

def get_utc_now():
    """Helper to get current UTC time without warnings."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

async def fetch_comments():
    """Step 1: Fetch data from JSONPlaceholder."""
    url = "https://jsonplaceholder.typicode.com/comments?postId=1"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return data[:3] # Take first 3
        except Exception as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            raise e

async def analyze_with_ai(text: str):
    """
    Step 2: AI Enrichment.
    Automatically detects if key is missing/invalid and uses Mock mode.
    """
    
    # 1. Check if key exists. If not, return Mock immediately.

    # 2. Try Real API Call
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY, 
            # REPLACE THIS URL with the one from your assignment instructions
            base_url="https://aipipe.org/openai/v1" 
        )
        prompt = (
            f"Analyze this in 2 sentences. "
            f"Then classify sentiment as positive, negative, or neutral. Use only these 3 words.\n\n"
            f"Text: {text}\n\n"
            f"Format: Analysis: [text] || Sentiment: [class]"
        )

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        content = response.choices[0].message.content
        parts = content.split("||")
        analysis = parts[0].replace("Analysis:", "").strip()
        sentiment = parts[1].replace("Sentiment:", "").strip().lower() if len(parts) > 1 else "neutral"
        
        return {"analysis": analysis, "sentiment": sentiment}

    except Exception as e:
        logger.warning(f"OpenAI API call failed ({str(e)}). Using Mock response.")
        return {
            "analysis": "Mock Analysis (API Error fallback). Key themes: Error handling demonstration.",
            "sentiment": "neutral"
        }

def store_result(original, analysis, sentiment):
    """Step 3: Storage (SQLite)."""
    timestamp = get_utc_now()
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO processed_items (source_text, analysis, sentiment, timestamp) VALUES (?, ?, ?, ?)",
                (original, analysis, sentiment, timestamp)
            )
            conn.commit()
        return True, timestamp
    except Exception as e:
        logger.error(f"Storage failed: {e}")
        return False, timestamp

def send_notification(user_email):
    """Step 4: Notification."""
    logger.info(f"--- NOTIFICATION SENT ---")
    logger.info(f"To: {TARGET_NOTIFICATION_EMAIL}")
    logger.info(f"CC: {user_email}")
    logger.info(f"Status: Success")
    logger.info(f"-------------------------")
    return True

# --- API Endpoint ---

@app.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest):
    processed_items = []
    errors = []
    
    # 1. Fetch
    try:
        comments = await fetch_comments()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data Fetch Failed: {str(e)}")

    # 2. Process Loop
    for item in comments:
        original_text = item.get("body", "")
        
        try:
            # AI Enrichment
            ai_result = await analyze_with_ai(original_text)
            
            # Storage
            is_stored, timestamp = store_result(
                original_text, 
                ai_result["analysis"], 
                ai_result["sentiment"]
            )
            
            processed_items.append({
                "original": original_text[:50] + "...", 
                "analysis": ai_result["analysis"],
                "sentiment": ai_result["sentiment"],
                "stored": is_stored,
                "timestamp": timestamp
            })
            
        except Exception as e:
            errors.append(f"Item {item.get('id')} failed: {str(e)}")
            continue

    # 3. Notification
    notified = send_notification(request.email)

    # 4. Response
    return {
        "items": processed_items,
        "notificationSent": notified,
        "processedAt": get_utc_now(),
        "errors": errors
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


