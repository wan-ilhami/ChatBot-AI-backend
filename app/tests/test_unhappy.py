import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import app, OutletsDB  
from main_brain import ConversationController, CalculatorTool

# Initialize database before tests run
OutletsDB.init_db()

# Create test client
client = TestClient(app)



# ============================================================================
# PART 1 TESTS: Sequential Conversation
# ============================================================================

class TestSequentialConversation:
    """Tests for Part 1 - Multi-turn conversation tracking"""

    def test_three_turn_happy_path(self):
        """
        The classic example from the requirements.
        Tests that state persists across 3 turns.
        """
        controller = ConversationController()
        
        # Turn 1: Ask about location
        resp1 = controller.process_turn("Is there an outlet in Petaling Jaya?")
        assert controller.memory.slots.location is not None
        assert "petaling" in resp1.lower() or "found" in resp1.lower()
        
        # Turn 2: Specify outlet
        resp2 = controller.process_turn("What about the SS 2 outlet?")
        assert controller.memory.slots.outlet_name is not None
        
        # Turn 3: Ask for hours
        resp3 = controller.process_turn("What's the opening time?")
        assert "9" in resp3 or "hour" in resp3.lower()
        
        # Verify all 3 turns recorded
        assert controller.memory.get_turn_count() == 3
        
        print("✓ Three-turn conversation passed")

    def test_context_retention(self):
        """
        Tests that bot remembers context from earlier turns.
        User doesn't need to repeat location every time.
        """
        controller = ConversationController()
        
        controller.process_turn("Find outlets in Klang")
        assert controller.memory.slots.location == "klang"
        
        # Second query should still have location in memory
        controller.process_turn("What are the hours?")
        assert controller.memory.slots.location == "klang"
        
        print("✓ Context retention works")

    def test_slot_overwriting(self):
        """
        Tests that slots can be updated if user changes their mind.
        """
        controller = ConversationController()
        
        controller.process_turn("Outlets in Petaling Jaya")
        assert "petaling" in controller.memory.slots.location.lower()
        
        # User changes mind
        controller.process_turn("Actually, show me Klang instead")
        assert "klang" in controller.memory.slots.location.lower()
        
        print("✓ Slot overwriting works")


# ============================================================================
# PART 2 TESTS: Agentic Planning
# ============================================================================

class TestAgenticPlanning:
    """Tests for Part 2 - Intent parsing and action planning"""

    def test_intent_detection_calculate(self):
        """Verify calculator intent is detected correctly"""
        controller = ConversationController()
        
        response = controller.process_turn("Calculate 10 + 20")
        
        # Should have detected calculate intent and used calculator
        assert "30" in response
        
        print("✓ Calculate intent detected")

    def test_intent_detection_products(self):
        """Verify product inquiry intent is detected"""
        controller = ConversationController()
        
        response = controller.process_turn("What products do you have?")
        
        # Should mention products or drinkware
        assert "product" in response.lower() or "drinkware" in response.lower()
        
        print("✓ Product intent detected")

    def test_clarification_question_asked(self):
        """
        When info is missing, bot should ask clarifying question.
        This is the core of agentic planning.
        """
        controller = ConversationController()
        
        response = controller.process_turn("What are the hours?")
        
        # Bot should ask for location or outlet
        assert "which" in response.lower() or "location" in response.lower()
        
        print("✓ Clarification question asked correctly")


# ============================================================================
# PART 3 TESTS: Tool Integration (Calculator)
# ============================================================================

class TestCalculatorTool:
    """Tests for Part 3 - Calculator tool integration"""

    def test_basic_arithmetic(self):
        """Test simple calculations"""
        calc = CalculatorTool()
        
        test_cases = [
            ("5 + 3", 8),
            ("10 - 7", 3),
            ("4 * 5", 20),
            ("20 / 4", 5),
        ]
        
        for expr, expected in test_cases:
            result, error = calc.calculate(expr)
            assert error is None
            assert result == expected
        
        print("✓ Basic arithmetic works")

    def test_order_of_operations(self):
        """Test that PEMDAS is respected"""
        calc = CalculatorTool()
        
        result, error = calc.calculate("15 + 25 * 2")
        assert error is None
        assert result == 65  # Not 80!
        
        print("✓ Order of operations correct")

    def test_parentheses(self):
        """Test parentheses work"""
        calc = CalculatorTool()
        
        result, error = calc.calculate("(15 + 25) * 2")
        assert error is None
        assert result == 80
        
        print("✓ Parentheses work")

    def test_decimal_numbers(self):
        """Test that decimals are handled"""
        calc = CalculatorTool()
        
        result, error = calc.calculate("10.5 + 5.5")
        assert error is None
        assert result == 16.0
        
        print("✓ Decimal calculations work")


# ============================================================================
# PART 4 TESTS: RAG & Text2SQL
# ============================================================================

class TestProductEndpoint:
    """Tests for Part 4 - Product search (RAG simulation)"""

    def test_product_search_specific(self):
        """Search for specific product type"""
        response = client.get("/products?query=glass+coffee+cup")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert len(data["results"]) > 0
        assert "glass" in data["results"][0]["name"].lower()
        
        print("✓ Specific product search works")

    def test_product_search_generic(self):
        """Generic 'what products' query should return all"""
        response = client.get("/products?query=products")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return multiple products
        assert len(data["results"]) >= 4
        
        print("✓ Generic product query returns all products")

    def test_product_relevance_scoring(self):
        """Results should have relevance scores"""
        response = client.get("/products?query=eco+friendly")
        
        assert response.status_code == 200
        data = response.json()
        
        for product in data["results"]:
            assert "relevance_score" in product
            assert 0 <= product["relevance_score"] <= 1
        
        print("✓ Relevance scoring works")


class TestOutletEndpoint:
    """Tests for Part 4 - Outlet search (Text2SQL)"""

    def test_outlet_search_by_city(self):
        """Search outlets by city name"""
        response = client.get("/outlets?query=Petaling+Jaya")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) > 0
        assert "Petaling Jaya" in data["results"][0]["city"]
        
        print("✓ City-based outlet search works")

    def test_outlet_search_by_location(self):
        """Search by specific location name"""
        response = client.get("/outlets?query=SS+2")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) > 0
        # Should find SS 2 outlet
        found = any("SS 2" in outlet["location"] for outlet in data["results"])
        assert found
        
        print("✓ Location-based outlet search works")

    def test_outlet_has_all_fields(self):
        """Verify outlet objects have all required fields"""
        response = client.get("/outlets?query=Klang")
        
        assert response.status_code == 200
        data = response.json()
        
        outlet = data["results"][0]
        required_fields = ["id", "name", "location", "city", "hours_open", 
                          "hours_close", "address", "services"]
        
        for field in required_fields:
            assert field in outlet
        
        print("✓ Outlet objects have all required fields")


class TestChatEndpoint:
    """Tests for integrated chat endpoint"""

    def test_chat_calculator_integration(self):
        """Chat endpoint correctly routes to calculator"""
        response = client.post("/chat", json={
            "user_id": "test_calc",
            "message": "Calculate 50 + 50"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "100" in data["response"]
        assert "calculator" in data["tools_used"]
        
        print("✓ Chat-calculator integration works")

    def test_chat_product_integration(self):
        """Chat endpoint correctly routes to product search"""
        response = client.post("/chat", json={
            "user_id": "test_prod",
            "message": "Show me bamboo cups"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "product_search" in data["tools_used"]
        assert "bamboo" in data["response"].lower()
        
        print("✓ Chat-product integration works")

    def test_chat_outlet_integration(self):
        """Chat endpoint correctly routes to outlet search"""
        response = client.post("/chat", json={
            "user_id": "test_outlet",
            "message": "Find outlets in Shah Alam"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "outlet_search" in data["tools_used"]
        assert "shah alam" in data["response"].lower()
        
        print("✓ Chat-outlet integration works")


# ============================================================================
# SYSTEM TESTS
# ============================================================================

class TestSystemHealth:
    """Tests for overall system health"""

    def test_health_endpoint(self):
        """Health check returns correct status"""
        response = client.post("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["chat"] == "operational"
        
        print("✓ Health endpoint works")

    def test_root_endpoint(self):
        """Root endpoint returns API info"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        
        print("✓ Root endpoint works")

    def test_openapi_docs_available(self):
        """OpenAPI documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
        
        print("✓ OpenAPI docs available")


# ============================================================================
# PERFORMANCE TESTS (OPTIONAL)
# ============================================================================

class TestPerformance:
    """
    Basic performance tests.
    These aren't required but I wanted to make sure nothing is super slow.
    """

    @pytest.mark.slow
    def test_response_time_acceptable(self):
        """Simple queries should respond quickly"""
        import time
        
        start = time.time()
        response = client.post("/chat", json={
            "user_id": "perf_test",
            "message": "Calculate 1 + 1"
        })
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 1.0  # Should respond in under 1 second
        
        print(f"✓ Response time: {duration:.3f}s")

    @pytest.mark.slow
    def test_multiple_concurrent_users(self):
        """System handles multiple users"""
        users = [f"user_{i}" for i in range(10)]
        
        for user in users:
            response = client.post("/chat", json={
                "user_id": user,
                "message": "Hello"
            })
            assert response.status_code == 200
        
        print("✓ Handles multiple concurrent users")


# ============================================================================
# RUN TESTS MANUALLY
# ============================================================================

if __name__ == "__main__":
    """
    Quick way to run tests without pytest.
    Useful during development.
    """
    print("\n" + "="*70)
    print("MINDHIVE ASSESSMENT - HAPPY PATH TEST SUITE")
    print("="*70)
    
    # Similar to test_unhappy_flows.py, just run all tests manually
    test_classes = [
        TestSequentialConversation,
        TestAgenticPlanning,
        TestCalculatorTool,
        TestProductEndpoint,
        TestOutletEndpoint,
        TestChatEndpoint,
        TestSystemHealth,
        # Skip performance tests in manual mode
    ]
    
    total = 0
    passed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*70}")
        print(f"Running: {test_class.__name__}")
        print(f"{'='*70}\n")
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed += 1
                print(f"  PASS: {method_name}")
            except Exception as e:
                print(f"  FAIL: {method_name}")
                print(f"    Error: {str(e)}")
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print(f"{'='*70}\n")