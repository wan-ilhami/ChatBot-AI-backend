from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import sqlite3
import re

# ============================================================================
# SETUP
# ============================================================================

app = FastAPI(
    title="Backend - Mindhive AI",
    version="2.0.0",
    description="FastAPI backend with improved RAG and Text2SQL integration"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=1000)

    @validator('message')
    def validate_message(cls, v):
        if '<script>' in v.lower() or '<?php' in v.lower():
            raise ValueError('Malicious content detected')
        return v

class ChatResponse(BaseModel):
    response: str
    intent: str
    tools_used: List[str]
    timestamp: str

class Product(BaseModel):
    id: str
    name: str
    category: str
    description: str
    price: Optional[float] = None
    relevance_score: float

class ProductResponse(BaseModel):
    query: str
    results: List[Product]
    summary: str

class Outlet(BaseModel):
    id: int
    name: str
    location: str
    city: str
    hours_open: str
    hours_close: str
    address: str
    services: str

class OutletResponse(BaseModel):
    query: str
    results: List[Outlet]
    count: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

# ============================================================================
# PRODUCT KNOWLEDGE BASE
# ============================================================================

class ProductKB:
    PRODUCTS = [
        {
            "id": "prod_001",
            "name": "Glass Coffee Cup",
            "category": "Drinkware",
            "description": "Premium borosilicate glass coffee cup, heat-resistant, 350ml capacity",
            "price": 24.99,
            "keywords": ["glass", "cup", "coffee", "drinkware", "borosilicate", "350ml", "transparent"]
        },
        {
            "id": "prod_002",
            "name": "Ceramic Travel Mug",
            "category": "Drinkware",
            "description": "Insulated ceramic travel mug with leak-proof lid, 400ml",
            "price": 34.99,
            "keywords": ["ceramic", "travel", "mug", "insulated", "leak-proof", "400ml", "portable"]
        },
        {
            "id": "prod_003",
            "name": "Stainless Steel Thermos",
            "category": "Drinkware",
            "description": "Double-wall stainless steel thermos, keeps drinks hot/cold for 12 hours",
            "price": 44.99,
            "keywords": ["stainless", "steel", "thermos", "insulated", "hot", "cold", "flask"]
        },
        {
            "id": "prod_004",
            "name": "Eco-Friendly Bamboo Cup",
            "category": "Drinkware",
            "description": "Sustainable bamboo drinkware, biodegradable, 300ml",
            "price": 19.99,
            "keywords": ["bamboo", "eco", "sustainable", "green", "biodegradable", "300ml", "environment"]
        },
        {
            "id": "prod_005",
            "name": "French Press Coffee Maker",
            "category": "Drinkware",
            "description": "Classic French press for brewing, 1L capacity, stainless steel",
            "price": 39.99,
            "keywords": ["french", "press", "coffee", "maker", "brewing", "1l", "brewer"]
        },
    ]

    @staticmethod
    def search(query: str, top_k: int = 5) -> tuple:
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Enhanced: if very generic query, return all products
        generic_queries = {"product", "products", "all", "what", "show", "list", "have", "sell", "offer"}
        if query_words.issubset(generic_queries) or len(query_words) <= 2:
            logger.info("Generic product query detected, returning all products")
            results = [
                Product(
                    id=p["id"],
                    name=p["name"],
                    category=p["category"],
                    description=p["description"],
                    price=p["price"],
                    relevance_score=0.5
                )
                for p in ProductKB.PRODUCTS[:top_k]
            ]
            return results, [0.5] * len(results)

        scored_products = []
        for product in ProductKB.PRODUCTS:
            keywords_set = set(product["keywords"])
            matches = len(query_words & keywords_set)
            
            # Also check description
            desc_matches = sum(1 for word in query_words if word in product["description"].lower())
            total_matches = matches + desc_matches * 0.5
            
            if total_matches > 0:
                score = min(total_matches / max(len(keywords_set), 1), 0.99)
                scored_products.append((product, score))

        scored_products.sort(key=lambda x: x[1], reverse=True)

        results = [
            Product(
                id=p["id"],
                name=p["name"],
                category=p["category"],
                description=p["description"],
                price=p["price"],
                relevance_score=score
            )
            for p, score in scored_products[:top_k]
        ]

        return results, [p[1] for p in scored_products[:top_k]]

    @staticmethod
    def generate_summary(products: List[Product], query: str) -> str:
        if not products:
            return "We have drinkware products available. Try asking about: glass cups, travel mugs, thermoses, bamboo cups, or french press."

        if len(products) >= 4:
            return f"We have {len(products)} drinkware products:\n\n" + "\n".join([
                f"• {p.name} - ${p.price} - {p.description}"
                for p in products
            ])

        names = [f"{p.name} (${p.price})" for p in products]
        summary = f"Found {len(products)} product(s): {', '.join(names)}."

        if products[0].relevance_score > 0.7:
            summary += f"\n\nBest match: {products[0].name} - {products[0].description}"

        return summary

# ============================================================================
# OUTLETS DATABASE
# ============================================================================

class OutletsDB:
    DB_PATH = "outlets.db"

    @staticmethod
    def init_db():
        try:
            conn = sqlite3.connect(OutletsDB.DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outlets (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    city TEXT NOT NULL,
                    hours_open TEXT NOT NULL,
                    hours_close TEXT NOT NULL,
                    address TEXT NOT NULL,
                    services TEXT NOT NULL
                )
            """)

            # Check if already populated
            cursor.execute("SELECT COUNT(*) FROM outlets")
            if cursor.fetchone()[0] == 0:
                outlets_data = [
                    ("SS 2, Petaling Jaya", "SS 2", "Petaling Jaya", "09:00", "22:00",
                     "123 Jalan SS 2/45, 58000 Kuala Lumpur", "Dine-in, Takeaway, WiFi"),
                    ("Klang Main Branch", "Klang Main", "Klang", "08:00", "23:00",
                     "456 Jalan Sultan Sulaiman, 41000 Klang", "Dine-in, Takeaway, Drive-through"),
                    ("Shah Alam Outlet", "Shah Alam", "Shah Alam", "10:00", "21:00",
                     "789 Persiaran Sultan Salahuddin, 40000 Shah Alam", "Dine-in, Takeaway"),
                    ("Pavilion KL", "Pavilion", "Kuala Lumpur", "10:00", "22:00",
                     "168 Jalan Bukit Bintang, 55100 Kuala Lumpur", "Dine-in, Takeaway, WiFi"),
                    ("IOI Mall", "IOI Mall", "Putrajaya", "11:00", "21:00",
                     "Lot 1-A-1A & 1-A-1B, Level 1, IOI Mall, 62000 Putrajaya", "Dine-in, Takeaway"),
                ]

                for data in outlets_data:
                    cursor.execute("""
                        INSERT INTO outlets (name, location, city, hours_open, hours_close, address, services)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, data)

                conn.commit()
                logger.info("Outlets database initialized with 5 locations")

            return True
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            return False

    @staticmethod
    def text_to_sql(natural_query: str) -> str:
        query_lower = natural_query.lower()
        sql = "SELECT * FROM outlets"
        conditions = []

        if any(word in query_lower for word in ["petaling", "pj", "ss 2", "ss2"]):
            conditions.append("(city = 'Petaling Jaya' OR location LIKE '%SS 2%')")
        elif "klang" in query_lower:
            conditions.append("city = 'Klang'")
        elif "shah alam" in query_lower:
            conditions.append("city = 'Shah Alam'")
        elif "pavilion" in query_lower or "bukit bintang" in query_lower:
            conditions.append("location LIKE '%Pavilion%'")
        elif "ioi" in query_lower or "putrajaya" in query_lower:
            conditions.append("(location LIKE '%IOI%' OR city = 'Putrajaya')")
        elif "kuala lumpur" in query_lower or "kl" in query_lower:
            conditions.append("city = 'Kuala Lumpur'")

        if "dine" in query_lower or "seating" in query_lower:
            conditions.append("services LIKE '%Dine-in%'")
        if "takeaway" in query_lower or "takeout" in query_lower:
            conditions.append("services LIKE '%Takeaway%'")
        if "drive" in query_lower:
            conditions.append("services LIKE '%Drive-through%'")

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " LIMIT 10"
        logger.info(f"Generated SQL: {sql}")
        return sql

    @staticmethod
    def execute_query(sql: str) -> List[Outlet]:
        try:
            conn = sqlite3.connect(OutletsDB.DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if any(keyword in sql.upper() for keyword in ["DROP", "DELETE", "INSERT", "UPDATE"]):
                logger.warning(f"Blocked dangerous SQL: {sql}")
                raise ValueError("Only SELECT queries allowed")

            cursor.execute(sql)
            rows = cursor.fetchall()

            outlets = [
                Outlet(
                    id=row["id"],
                    name=row["name"],
                    location=row["location"],
                    city=row["city"],
                    hours_open=row["hours_open"],
                    hours_close=row["hours_close"],
                    address=row["address"],
                    services=row["services"]
                )
                for row in rows
            ]

            return outlets
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            raise

# ============================================================================
# CALCULATOR TOOL
# ============================================================================

class CalculatorTool:
    ALLOWED_CHARS = set("0123456789+-*/(). ")

    @staticmethod
    def extract_expression(text: str) -> Optional[str]:
        patterns = [
            r'(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*)',
            r'calculate\s+(.*?)(?:\?|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def calculate(expression: str) -> tuple:
        try:
            expression = expression.strip()
            if not expression or not all(c in CalculatorTool.ALLOWED_CHARS for c in expression):
                return None, "Invalid expression"
            if "/0" in expression.replace(" ", ""):
                return None, "Division by zero"
            
            result = eval(expression, {"__builtins__": {}})
            if not isinstance(result, (int, float)):
                return None, "Result is not numeric"
            
            return result, None
        except Exception as e:
            return None, str(e)

# ============================================================================
# ENHANCED CHAT CONTROLLER
# ============================================================================

CONVERSATIONS = {}

class EnhancedChatController:
    @staticmethod
    async def process_message(user_id: str, message: str) -> ChatResponse:
        # Initialize conversation
        if user_id not in CONVERSATIONS:
            CONVERSATIONS[user_id] = {
                "turns": [],
                "context": {},
                "slots": {}
            }

        conv = CONVERSATIONS[user_id]
        tools_used = []
        response_parts = []
        intent = "unknown"

        msg_lower = message.lower()

        # 1. CALCULATOR INTENT
        calc_keywords = ["calculate", "calc", "compute", "+", "-", "*", "/"]
        if any(kw in msg_lower for kw in calc_keywords):
            intent = "calculate"
            tools_used.append("calculator")
            
            expr = CalculatorTool.extract_expression(message)
            if expr:
                result, error = CalculatorTool.calculate(expr)
                if error:
                    response_parts.append(f"❌ Calculation error: {error}")
                else:
                    response_parts.append(f"✅ {expr} = {result}")
                    conv["context"]["last_calculation"] = result
            else:
                response_parts.append("I can help with calculations! Try: 'Calculate 15 + 25 * 2'")

        # 2. PRODUCT INTENT
        product_keywords = ["product", "drinkware", "cup", "mug", "glass", "thermos", "bamboo", "french press"]
        if any(kw in msg_lower for kw in product_keywords):
            intent = "product_inquiry"
            tools_used.append("product_search")
            
            try:
                results, scores = ProductKB.search(message)
                summary = ProductKB.generate_summary(results, message)
                response_parts.append(f"🛍️ {summary}")
            except Exception as e:
                response_parts.append(f"Product search temporarily unavailable: {str(e)}")

        # 3. OUTLET INTENT
        outlet_keywords = ["outlet", "branch", "location", "where", "address", "hours", "open", "close"]
        if any(kw in msg_lower for kw in outlet_keywords):
            intent = "outlet_inquiry"
            tools_used.append("outlet_search")
            
            try:
                sql = OutletsDB.text_to_sql(message)
                outlets = OutletsDB.execute_query(sql)
                
                if outlets:
                    if "hours" in msg_lower or "open" in msg_lower or "close" in msg_lower:
                        outlet_info = "\n".join([
                            f"• {o.name}: {o.hours_open} - {o.hours_close}"
                            for o in outlets
                        ])
                        response_parts.append(f"⏰ Operating hours:\n{outlet_info}")
                    elif "address" in msg_lower:
                        outlet_info = "\n".join([
                            f"• {o.name}: {o.address}"
                            for o in outlets
                        ])
                        response_parts.append(f"📍 Addresses:\n{outlet_info}")
                    else:
                        outlet_info = "\n".join([
                            f"• {o.name} ({o.city}) - {o.services}"
                            for o in outlets
                        ])
                        response_parts.append(f"📍 Found {len(outlets)} outlet(s):\n{outlet_info}")
                else:
                    response_parts.append("No outlets found. Try: Petaling Jaya, Klang, Shah Alam, Pavilion, or IOI Mall")
            except Exception as e:
                response_parts.append(f"Outlet search error: {str(e)}")

        # 4. GREETING
        if any(word in msg_lower for word in ["hello", "hi", "hey", "greetings"]):
            intent = "greeting"
            tools_used.append("general_response")
            response_parts.append("👋 Hello! I can help you with:\n• Finding outlets\n• Product information\n• Calculations\n\nWhat would you like to know?")

        # 5. DEFAULT
        if not response_parts:
            intent = "general"
            tools_used.append("general_response")
            response_parts.append("I can help you find outlets, search products, or perform calculations. What would you like to know?")

        # Build final response
        bot_response = "\n\n".join(response_parts)

        # Store turn
        conv["turns"].append({
            "user": message,
            "bot": bot_response,
            "tools": tools_used,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })

        return ChatResponse(
            response=bot_response,
            intent=intent,
            tools_used=tools_used,
            timestamp=datetime.now().isoformat()
        )

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/products", response_model=ProductResponse, tags=["RAG"])
async def search_products(query: str = Query(..., min_length=1)):
    try:
        if len(query) > 200:
            raise ValueError("Query too long")
        
        results, scores = ProductKB.search(query)
        summary = ProductKB.generate_summary(results, query)
        
        return ProductResponse(query=query, results=results, summary=summary)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Product search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Product search failed")

@app.get("/outlets", response_model=OutletResponse, tags=["Text2SQL"])
async def search_outlets(query: str = Query(..., min_length=1)):
    try:
        if len(query) > 200:
            raise ValueError("Query too long")
        
        # Enhanced SQL injection detection
        if any(word in query.lower() for word in ["drop", "delete", "insert", "update", "'", "--", ";"]):
            raise ValueError("Malicious query detected")
        
        sql = OutletsDB.text_to_sql(query)
        outlets = OutletsDB.execute_query(sql)
        
        return OutletResponse(query=query, results=outlets, count=len(outlets))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Outlet search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Outlet search failed")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def process_chat(req: ChatRequest):
    try:
        return await EnhancedChatController.process_message(req.user_id, req.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services={
            "chat": "operational",
            "products": "operational",
            "outlets": "operational"
        }
    )

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Mindhive AI Backend",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "products": "GET /products?query=...",
            "outlets": "GET /outlets?query=...",
            "health": "POST /health"
        }
    }

@app.on_event("startup")
async def startup():
    logger.info("Starting Enhanced Mindhive AI Backend")
    OutletsDB.init_db()
    logger.info("Backend ready version with better integration")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)