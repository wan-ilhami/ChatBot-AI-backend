# Mindhive AI Backend

## Table of Contents

1. [Setup & Run Instructions](#setup--run-instructions)
2. [Architecture Overview](#architecture-overview)
3. [Key Trade-offs](#key-trade-offs)

---

## Setup & Run Instructions

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install fastapi uvicorn pydantic pytest httpx
```

### Step 4: Create `__init__.py`

Add empty file in `app/` folder:

```bash
touch app/__init__.py
```
---

## Running Tests

### Run All Tests with Pytest

```bash
pytest tests/test_happy.py -v
pytest tests/test_unhappy.py -v
```

### Run with Output Capture

```bash
pytest tests/test_happy.py -v -s
pytest tests/test_unhappy.py -v -s
```

### Manual Testing (without pytest)

```bash
python tests/test_happy.py
python tests/test_unhappy.py
```

---

## Running the Server

```bash
python -m uvicorn app.backend:app --host 0.0.0.0 --port 8000
```

**API Documentation:** http://localhost:8000/docs

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────┐
│     FastAPI REST API Endpoints          │
├─────────────────────────────────────────┤
│  /chat  |  /products  |  /outlets       │
├─────────────────────────────────────────┤
│    Enhanced Chat Controller             │
│  (Intent Detection & Tool Routing)      │
├─────────────────────────────────────────┤
│  ProductKB  | OutletsDB | Calculator    │
│  (In-Memory) | (SQLite)  | (Safe Eval)  │
├─────────────────────────────────────────┤
│        Conversation Memory              │
│     (Per-user State Management)         │
└─────────────────────────────────────────┘
```

### Core Components

**1. FastAPI Application (`backend.py`)**
- REST API endpoints
- Request/response validation with Pydantic
- CORS middleware for cross-origin requests
- Health check endpoint

**2. Chat Controller (`main_brain.py`)**
- Multi-turn conversation orchestration
- Intent parsing (calculator, products, outlets, etc.)
- Slot-filling for missing information
- Action planning and tool routing

**3. Product Knowledge Base (`ProductKB`)**
- In-memory product catalog (5 drinkware items)
- Keyword-based search with relevance scoring
- Handles generic queries ("show all products")

**4. Outlets Database (`OutletsDB`)**
- SQLite database with 5 pre-populated locations
- Natural language to SQL conversion (Text2SQL)
- Location and service filtering
- SQL injection prevention

**5. Calculator Tool (`CalculatorTool`)**
- Safe mathematical expression evaluation
- Character whitelist (no code injection)
- Supports: +, -, *, /, (), decimals
- Division by zero protection

**6. Conversation Memory**
- Tracks multi-turn conversations per user
- Stores slots (location, outlet name, etc.)
- Context window management (5 recent turns)
- Turn history with timestamps

---

## Key Trade-offs

### ✅ Chosen: In-Memory Product KB

**Why:** Simple, fast, no database overhead
**Trade-off:** Only 5 products, doesn't scale
**Alternative:** Elasticsearch or PostgreSQL with full-text search

### ✅ Chosen: SQLite for Outlets

**Why:** Lightweight, zero setup, file-based persistence
**Trade-off:** Not ideal for high concurrency, limited query performance
**Alternative:** PostgreSQL for production

### ✅ Chosen: Keyword-Based Intent Detection

**Why:** Fast, deterministic, no ML model needed
**Trade-off:** Brittle with paraphrasing, limited language understanding
**Alternative:** NLP with spaCy or Hugging Face transformers

### ✅ Chosen: Text2SQL with Pattern Matching

**Why:** Simple, prevents SQL injection, no LLM dependency
**Trade-off:** Only handles predefined patterns
**Alternative:** LLM-based (Claude/GPT) for natural language flexibility

### ✅ Chosen: In-Memory Conversation State

**Why:** Fast, simple for single-server deployment
**Trade-off:** Resets on server restart, doesn't scale horizontally
**Alternative:** Redis for distributed, persistent state

### ✅ Chosen: Safe Calculator with eval()

**Why:** Simple to implement, sufficient for basic arithmetic
**Trade-off:** Potential security risk if whitelist bypassed
**Alternative:** Parse and evaluate AST instead of eval()

### ✅ Chosen: Per-User Conversation Storage

**Why:** Isolation and privacy per user
**Trade-off:** Memory grows with number of concurrent users
**Alternative:** Implement conversation cleanup/TTL

### ✅ Chosen: Pydantic Validation

**Why:** Built-in validation, catches malicious input early
**Trade-off:** Tight coupling to FastAPI
**Alternative:** Custom validation layer

---

## Folder Structure

```
backend/
├── app/
│   ├── __init__.py           # Package marker
│   ├── backend.py            # FastAPI app & endpoints
│   └── main_brain.py         # Conversation logic
├── tests/
│   ├── test_happy.py         # Happy path tests
│   └── test_unhappy.py       # Error handling tests
└── venv/                     # Virtual environment
```

---

## Environment Setup Note

`pyenv.cfg` is **auto-generated** when you create the virtual environment. It stores Python version info. You can safely ignore it.

---

## Quick API Examples

### Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "message": "Calculate 15 + 25 * 2"}'
```

### Product Search
```bash
curl http://localhost:8000/products?query=glass+cup
```

### Outlet Search
```bash
curl http://localhost:8000/outlets?query=Klang
```

---

**Version:** 2.0.0  
**Framework:** FastAPI + Uvicorn  
**Database:** SQLite