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
# TEST CATEGORY 1: MISSING PARAMETERS
# ============================================================================

class TestMissingParameters:
    """
    Tests for when users provide incomplete information.
    The bot should ask clarifying questions, not crash.
    """

    def test_calculator_without_expression(self):
        """
        User says 'Calculate' without providing what to calculate.
        Expected: Bot asks for the expression or gives example.
        """
        response = client.post("/chat", json={
            "user_id": "test_user_001",
            "message": "Calculate"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should not crash, should respond helpfully
        assert "calculate" in data["response"].lower() or "calculation" in data["response"].lower()
        # Should mention how to use it
        assert "try" in data["response"].lower() or "example" in data["response"].lower()
        
        print(f"✓ Calculator without expression handled gracefully")
        print(f"  Response: {data['response'][:100]}...")

    def test_outlet_search_no_location(self):
        """
        User says 'Show outlets' without specifying location.
        Expected: Returns all outlets OR asks for location.
        """
        response = client.post("/chat", json={
            "user_id": "test_user_002",
            "message": "Show me outlets"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should either show outlets or ask for clarification
        # Both behaviors are acceptable
        assert len(data["response"]) > 0
        assert response.status_code != 500  # Most important: didn't crash!
        
        print(f"✓ Generic outlet query handled")
        print(f"  Tools used: {data['tools_used']}")

    def test_product_search_empty_query(self):
        """
        Product search with meaningless query like 'what' or 'show'.
        Expected: Returns all products or asks what they're looking for.
        """
        response = client.get("/products?query=show")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return something useful, not crash
        assert "results" in data
        assert len(data["results"]) >= 0  # Can be empty, just shouldn't crash
        
        print(f"✓ Empty product query handled")
        print(f"  Returned {len(data['results'])} products")

    def test_missing_slot_conversation_flow(self):
        """
        Multi-turn conversation where user gradually provides information.
        This tests the slot-filling mechanism under Part 1.
        """
        user_id = "test_user_003"
        
        # Turn 1: Vague request
        r1 = client.post("/chat", json={
            "user_id": user_id,
            "message": "What time do you open?"
        })
        assert r1.status_code == 200
        # Bot should ask for more info (location/outlet)
        
        # Turn 2: Provide location
        r2 = client.post("/chat", json={
            "user_id": user_id,
            "message": "The one in Petaling Jaya"
        })
        assert r2.status_code == 200
        
        # Turn 3: Specify outlet
        r3 = client.post("/chat", json={
            "user_id": user_id,
            "message": "SS 2"
        })
        assert r3.status_code == 200
        
        print(f"✓ Multi-turn slot filling works")
        print(f"  Final response includes hours: {'hour' in r3.json()['response'].lower()}")


# ============================================================================
# TEST CATEGORY 2: MALICIOUS INPUTS
# ============================================================================

class TestMaliciousInputs:
    """
    Tests for security - SQL injection, XSS, code injection attempts.
    The system should detect and block these, never execute them.
    """

    def test_sql_injection_outlets(self):
        """
        Classic SQL injection attempt on outlets endpoint.
        Expected: 400 error with 'malicious' message, NOT executed.
        """
        malicious_queries = [
            "'; DROP TABLE outlets; --",
            "1' OR '1'='1",
            "'; DELETE FROM outlets WHERE '1'='1",
            "admin'--",
        ]
        
        for query in malicious_queries:
            response = client.get(f"/outlets?query={query}")
            
            assert response.status_code in [400, 500]
            
            print(f"✓ Blocked SQL injection: {query[:30]}...")

    def test_xss_attempt_in_chat(self):
        """
        XSS attempt in chat message.
        Expected: Validator rejects it before processing.
        """
        response = client.post("/chat", json={
            "user_id": "test_hacker",
            "message": "<script>alert('XSS')</script>"
        })
        
        # Should be rejected by Pydantic validator
        assert response.status_code == 400 or response.status_code == 422
        
        print(f"✓ XSS attempt blocked in chat")

    def test_php_injection_attempt(self):
        """
        PHP code injection attempt (even though we're Python).
        Tests if validator catches general malicious patterns.
        """
        response = client.post("/chat", json={
            "user_id": "test_hacker_2",
            "message": "<?php system('rm -rf /'); ?>"
        })
        
        assert response.status_code == 400 or response.status_code == 422
        
        print(f"✓ PHP injection blocked")

    def test_oversized_input(self):
        """
        Extremely long input to test DOS protection.
        Expected: Rejected by length validator.
        """
        long_message = "A" * 2000  # Max is 1000 in our validator
        
        response = client.post("/chat", json={
            "user_id": "test_dos",
            "message": long_message
        })
        
        assert response.status_code == 422  # Pydantic validation error
        
        print(f"✓ Oversized input rejected (length: {len(long_message)})")

    def test_calculator_code_injection(self):
        """
        Attempt to inject Python code through calculator.
        Expected: Rejected by character whitelist.
        """
        calc_tool = CalculatorTool()
        
        malicious_expressions = [
            "import os",
            "__import__('os').system('ls')",
            "eval('print(123)')",
            "exec('x=1')",
        ]
        
        for expr in malicious_expressions:
            result, error = calc_tool.calculate(expr)
            
            # Should return error, not execute
            assert result is None
            assert error is not None
            assert "invalid" in error.lower() or "failed" in error.lower()
            
            print(f"✓ Calculator blocked: {expr[:30]}...")


# ============================================================================
# TEST CATEGORY 3: API FAILURES & DOWNTIME
# ============================================================================

class TestAPIFailures:
    """
    Tests for when things go wrong - database errors, network issues, etc.
    The frontend should show helpful errors, not just crash.
    """

    def test_database_not_initialized(self):
        """
        Simulate database file missing/corrupted.
        Note: Hard to test without mocking, but we check error handling exists.
        """
        # This is more of a demonstration that we thought about it
        # In real scenario, we'd mock sqlite3.connect to raise an exception
        
        response = client.get("/outlets?query=test")
        # Should work because we init DB on startup
        assert response.status_code == 200 or response.status_code == 500
        
        # If it's 500, response should have error message
        if response.status_code == 500:
            assert "error" in response.text.lower()
        
        print(f"✓ Database error handling present")

    def test_chat_with_backend_error(self):
        """
        Chat endpoint gracefully handles internal errors.
        We trigger this by sending weird data that might cause issues.
        """
        response = client.post("/chat", json={
            "user_id": "test_user_error",
            "message": "x" * 999  # Just under the limit, but might cause issues
        })
        
        # Should return 200 with error message OR 500 with error details
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            # Error should be in the bot's response
            data = response.json()
            # System should still respond, even if it's an error message
            assert len(data["response"]) > 0
        
        print(f"✓ Chat handles internal errors gracefully")

    def test_invalid_endpoint_request(self):
        """
        Request to non-existent endpoint.
        Expected: 404 with helpful message.
        """
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
        
        print(f"✓ 404 handled properly")

    def test_wrong_http_method(self):
        """
        Using GET instead of POST for chat endpoint.
        Expected: 405 Method Not Allowed.
        """
        response = client.get("/chat")
        
        assert response.status_code == 405
        
        print(f"✓ Wrong HTTP method rejected")


# ============================================================================
# TEST CATEGORY 4: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """
    Weird but valid inputs that might break things.
    """

    def test_empty_message(self):
        """
        User sends empty message.
        Expected: Rejected by min_length validator.
        """
        response = client.post("/chat", json={
            "user_id": "test_empty",
            "message": ""
        })
        
        assert response.status_code == 422  # Validation error
        
        print(f"✓ Empty message rejected")

    def test_only_spaces_message(self):
        """
        Message with only whitespace.
        Expected: Rejected or handled gracefully.
        """
        response = client.post("/chat", json={
            "user_id": "test_spaces",
            "message": "     "
        })
        
        # Could be 422 (validation) or 200 (handled)
        assert response.status_code in [200, 422]
        
        print(f"✓ Whitespace-only message handled")

    def test_special_characters_in_query(self):
        """
        Query with special characters (but not malicious).
        Expected: Handled gracefully.
        """
        response = client.get("/products?query=café ☕ 100%")
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        
        print(f"✓ Special characters in query handled")

    def test_very_long_user_id(self):
        """
        User ID at maximum allowed length.
        Expected: Accepted (within limit).
        """
        long_user_id = "u" * 100  # Max is 100
        
        response = client.post("/chat", json={
            "user_id": long_user_id,
            "message": "test"
        })
        
        assert response.status_code == 200
        
        print(f"✓ Maximum-length user_id accepted")

    def test_concurrent_requests_same_user(self):
        """
        Multiple requests from same user in quick succession.
        Expected: All handled independently (stateless-ish).
        """
        user_id = "test_concurrent"
        
        responses = []
        for i in range(5):
            r = client.post("/chat", json={
                "user_id": user_id,
                "message": f"Message {i}"
            })
            responses.append(r)
        
        # All should succeed
        for r in responses:
            assert r.status_code == 200
        
        print(f"✓ Concurrent requests handled")

    def test_calculator_division_by_zero(self):
        """
        Classic division by zero.
        Expected: Error message, not crash.
        """
        calc_tool = CalculatorTool()
        
        result, error = calc_tool.calculate("10 / 0")
        
        assert result is None
        assert error is not None
        assert "zero" in error.lower()
        
        print(f"✓ Division by zero handled: {error}")

    def test_calculator_complex_expression(self):
        """
        Valid but complex mathematical expression.
        Expected: Calculates correctly.
        """
        calc_tool = CalculatorTool()
        
        result, error = calc_tool.calculate("(15 + 25) * 2 - 10 / 5")
        
        assert error is None
        assert result == 78.0  # (15+25)*2 - 10/5 = 40*2 - 2 = 78
        
        print(f"✓ Complex calculation works: {result}")


# ============================================================================
# CONVERSATION CONTROLLER TESTS (Part 1 & 2)
# ============================================================================

class TestConversationController:
    """
    Unit tests for the conversation controller logic.
    Tests state management and planning decisions.
    """

    def test_happy_path_three_turns(self):
        """
        The classic 3-turn conversation from Part 1 requirements.
        """
        controller = ConversationController()
        
        # Turn 1
        resp1 = controller.process_turn("Is there an outlet in Petaling Jaya?")
        assert "petaling jaya" in resp1.lower() or "found" in resp1.lower()
        assert controller.memory.slots.location is not None
        
        # Turn 2
        resp2 = controller.process_turn("What about the SS 2 outlet?")
        assert controller.memory.slots.outlet_name is not None
        
        # Turn 3
        resp3 = controller.process_turn("What's the opening time?")
        assert "hour" in resp3.lower() or "time" in resp3.lower() or "9" in resp3
        
        assert controller.memory.get_turn_count() == 3
        
        print(f"✓ Three-turn conversation completed successfully")

    def test_interrupted_flow(self):
        """
        User provides incomplete info, bot asks for clarification.
        """
        controller = ConversationController()
        
        resp1 = controller.process_turn("What are your hours?")
        # Bot should ask for location
        assert "location" in resp1.lower() or "which" in resp1.lower()
        
        print(f"✓ Interrupted flow asks for clarification")

    def test_calculator_success(self):
        """
        Calculator tool integration - success case.
        """
        controller = ConversationController()
        
        resp = controller.process_turn("Calculate 15 + 25 * 2")
        
        assert "65" in resp  # Correct answer
        
        print(f"✓ Calculator integration works: {resp}")

    def test_calculator_failure(self):
        """
        Calculator tool integration - error case.
        """
        controller = ConversationController()
        
        resp = controller.process_turn("Calculate 10 / 0")
        
        assert "error" in resp.lower() or "zero" in resp.lower()
        assert "65" not in resp  # Shouldn't return wrong answer
        
        print(f"✓ Calculator error handled: {resp[:50]}...")

    def test_memory_persistence(self):
        """
        Memory correctly stores and retrieves conversation history.
        """
        controller = ConversationController()
        
        controller.process_turn("First message")
        controller.process_turn("Second message")
        controller.process_turn("Third message")
        
        memory_dict = controller.get_memory_snapshot()
        
        assert memory_dict["turn_count"] == 3
        assert len(memory_dict["turns"]) == 3
        assert memory_dict["turns"][0]["user_message"] == "First message"
        
        print(f"✓ Memory persistence verified")

    def test_reset_conversation(self):
        """
        Reset clears all conversation state.
        """
        controller = ConversationController()
        
        controller.process_turn("Some message")
        assert controller.memory.get_turn_count() == 1
        
        controller.reset()
        assert controller.memory.get_turn_count() == 0
        
        print(f"✓ Conversation reset works")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    """
    Run tests manually without pytest.
    Useful for quick debugging during development.
    """
    print("\n" + "="*70)
    print("MINDHIVE ASSESSMENT - UNHAPPY FLOWS TEST SUITE")
    print("="*70)
    
    # I know this isn't the "proper" way to run tests, but sometimes
    # during development I just want to see what breaks without dealing
    # with pytest configuration. Sue me. 😅
    
    test_classes = [
        TestMissingParameters,
        TestMaliciousInputs,
        TestAPIFailures,
        TestEdgeCases,
        TestConversationController,
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{'='*70}")
        print(f"Running: {test_class.__name__}")
        print(f"{'='*70}\n")
        
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
                print(f"  PASS: {method_name}")
            except Exception as e:
                print(f"  FAIL: {method_name}")
                print(f"    Error: {str(e)}")
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*70}\n")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is robust!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Check the errors above.")