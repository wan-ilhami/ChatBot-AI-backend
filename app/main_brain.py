"""
Part 1: Sequential Conversation with State Management
Part 2: Agentic Planning & Controller Logic
Part 3: Calculator Tool Integration

"""

import json
import re
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: STATE MANAGEMENT & MEMORY
# ============================================================================

class ConversationState(Enum):
    """Conversation state machine"""
    IDLE = "idle"
    GATHERING_INFO = "gathering_info"
    PROCESSING_TOOL = "processing_tool"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ConversationSlot:
    """Extracted information slots across turns"""
    location: Optional[str] = None
    outlet_name: Optional[str] = None
    query_type: Optional[str] = None
    calculation_expression: Optional[str] = None
    product_search_term: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class Turn:
    """Single conversation turn"""
    user_message: str
    bot_response: str
    intent: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    slots_snapshot: Dict = field(default_factory=dict)
    action_taken: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class ConversationMemory:
    """
    Multi-turn conversation memory with context window management.
    Tracks turns, slots, and conversation state.
    """

    def __init__(self, context_window: int = 5, max_turns: int = 50):
        self.turns: List[Turn] = []
        self.slots = ConversationSlot()
        self.state = ConversationState.IDLE
        self.context_window = context_window
        self.max_turns = max_turns
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"New conversation session: {self.session_id}")

    def add_turn(
        self,
        user_msg: str,
        bot_response: str,
        intent: str,
        action: Optional[str] = None
    ) -> None:
        """Record a conversation turn"""
        turn = Turn(
            user_message=user_msg,
            bot_response=bot_response,
            intent=intent,
            slots_snapshot=self.slots.to_dict(),
            action_taken=action
        )
        self.turns.append(turn)

        # Enforce max turns limit
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
            logger.warning(f"Conversation exceeded {self.max_turns} turns, truncating")

    def get_context(self, depth: int = 3) -> str:
        """Get formatted context from last N turns for LLM"""
        if not self.turns:
            return "No previous context."

        recent_turns = self.turns[-depth:]
        context = "Recent conversation history:\n"
        for i, turn in enumerate(recent_turns, 1):
            context += f"{i}. User: {turn.user_message}\n"
            context += f"   Bot: {turn.bot_response}\n"
        return context

    def get_slots_summary(self) -> str:
        """Get summary of extracted slots"""
        slots = self.slots.to_dict()
        filled = {k: v for k, v in slots.items() if v}
        if not filled:
            return "No information extracted yet."
        return json.dumps(filled, indent=2)

    def update_slots(self, **kwargs) -> None:
        """Update conversation slots"""
        for key, value in kwargs.items():
            if hasattr(self.slots, key) and value is not None:
                setattr(self.slots, key, value)
                logger.info(f"Updated slot: {key}={value}")

    def set_state(self, state: ConversationState) -> None:
        """Update conversation state"""
        logger.info(f"State transition: {self.state.value} → {state.value}")
        self.state = state

    def reset(self) -> None:
        """Reset memory for new conversation"""
        logger.info(f"Resetting conversation {self.session_id}")
        self.turns = []
        self.slots = ConversationSlot()
        self.state = ConversationState.IDLE
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_turn_count(self) -> int:
        """Get number of turns in conversation"""
        return len(self.turns)

    def get_last_turn(self) -> Optional[Turn]:
        """Get the last turn"""
        return self.turns[-1] if self.turns else None

    def to_dict(self) -> Dict:
        """Serialize memory to dict"""
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "state": self.state.value,
            "slots": self.slots.to_dict(),
            "turns": [t.to_dict() for t in self.turns],
            "context": self.get_context()
        }


# ============================================================================
# PART 2: AGENTIC PLANNING & CONTROLLER LOGIC
# ============================================================================

class IntentType(Enum):
    """Intent classification"""
    FIND_OUTLET = "find_outlet"
    GET_HOURS = "get_hours"
    GET_ADDRESS = "get_address"
    CALCULATE = "calculate"
    PRODUCT_INQUIRY = "product_inquiry"
    COMPLAINT = "complaint"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class Action:
    """Planned action for the bot"""
    intent: IntentType
    required_slots: List[str]
    filled_slots: List[str]
    missing_slots: List[str]
    next_question: Optional[str] = None
    tool_to_call: Optional[str] = None
    confidence: float = 0.0

    def needs_clarification(self) -> bool:
        """Check if more info is needed before action"""
        return len(self.missing_slots) > 0


class IntentParser:
    """Parse user intent and required slots"""

    INTENT_KEYWORDS = {
        IntentType.FIND_OUTLET: ["outlet", "branch", "store", "location", "where"],
        IntentType.GET_HOURS: ["open", "close", "hour", "time", "operational", "outlet"],
        IntentType.GET_ADDRESS: ["address", "located", "where", "directions"],
        IntentType.CALCULATE: ["calculate", "calc", "compute", "math", "add", "subtract", "multiply", "divide"],
        IntentType.PRODUCT_INQUIRY: ["product", "menu", "item", "drink", "coffee", "tea", "price"],
        IntentType.COMPLAINT: ["problem", "issue", "complaint", "unhappy", "wrong", "broken"],
        IntentType.GREETING: ["hello", "hi", "hey", "greetings", "good morning"],
    }

    def parse(self, user_msg: str, memory: ConversationMemory) -> Tuple[IntentType, float]:
        """
        Parse intent from user message.
        Returns: (IntentType, confidence_score)
        """
        msg_lower = user_msg.lower()
        matches = {}

        for intent_type, keywords in self.INTENT_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw in msg_lower)
            if match_count > 0:
                matches[intent_type] = match_count

        if not matches:
            return IntentType.UNKNOWN, 0.3

        # Highest match count wins
        best_intent = max(matches, key=matches.get)
        confidence = min(matches[best_intent] * 0.25, 0.95)

        logger.info(f"Parsed intent: {best_intent.value} (confidence: {confidence:.2f})")
        return best_intent, confidence


class Planner:
    """Agentic planner - decides next action"""

    SLOT_REQUIREMENTS = {
        IntentType.FIND_OUTLET: ["location"],
        IntentType.GET_HOURS: ["location", "outlet_name"],
        IntentType.GET_ADDRESS: ["location", "outlet_name"],
        IntentType.CALCULATE: ["calculation_expression"],
        IntentType.PRODUCT_INQUIRY: ["product_search_term"],
        IntentType.COMPLAINT: [],
        IntentType.GREETING: [],
        IntentType.UNKNOWN: [],
    }

    CLARIFICATION_QUESTIONS = {
        "location": "Which location are you interested in? (e.g., Petaling Jaya, Klang, Shah Alam)",
        "outlet_name": "Which outlet would you like to know about?",
        "calculation_expression": "What calculation would you like me to perform?",
        "product_search_term": "What product are you looking for?",
    }

    def plan(self, intent: IntentType, memory: ConversationMemory) -> Action:
        """
        Create action plan based on intent and current slots.
        Decision point: clarify, tool_call, or complete.
        """
        required = self.SLOT_REQUIREMENTS.get(intent, [])
        filled = []
        missing = []

        for slot_name in required:
            value = getattr(memory.slots, slot_name, None)
            if value:
                filled.append(slot_name)
            else:
                missing.append(slot_name)

        action = Action(
            intent=intent,
            required_slots=required,
            filled_slots=filled,
            missing_slots=missing,
            confidence=0.8 if not missing else 0.5
        )

        # Decision logic
        if missing:
            # Need clarification
            action.next_question = self.CLARIFICATION_QUESTIONS.get(
                missing[0],
                "Could you provide more details?"
            )
            logger.info(f"Action: ASK for {missing[0]}")
        else:
            # Ready to execute
            if intent == IntentType.FIND_OUTLET:
                action.tool_to_call = "search_outlets"
            elif intent == IntentType.GET_HOURS:
                action.tool_to_call = "get_outlet_hours"
            elif intent == IntentType.GET_ADDRESS:
                action.tool_to_call = "get_address"
            elif intent == IntentType.CALCULATE:
                action.tool_to_call = "calculator"
            elif intent == IntentType.PRODUCT_INQUIRY:
                action.tool_to_call = "search_products"

            logger.info(f"Action: EXECUTE {action.tool_to_call}")

        return action


# ============================================================================
# PART 3: CALCULATOR TOOL
# ============================================================================

class CalculatorTool:
    """
    Safe calculator for arithmetic operations.
    Supports: +, -, *, /, %, parentheses
    """

    ALLOWED_CHARS = set("0123456789+-*/(). ")

    @staticmethod
    def is_safe(expression: str) -> bool:
        """Validate expression contains only safe characters"""
        return all(c in CalculatorTool.ALLOWED_CHARS for c in expression)

    @staticmethod
    def extract_expression(text: str) -> Optional[str]:
        """Extract mathematical expression from text"""
        # Look for patterns like "5 + 3", "calculate 10 * 2", etc.
        patterns = [
            r'(\d+\s*[+\-*/]\s*\d+(?:\s*[+\-*/]\s*\d+)*)',
            r'calculate\s+(.*?)(?:\?|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def calculate(expression: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Execute calculation safely.
        Returns: (result, error_message)
        """
        try:
            # Clean expression
            expression = expression.strip()

            # Validate
            if not expression:
                return None, "Empty expression provided"

            if not CalculatorTool.is_safe(expression):
                return None, "Expression contains invalid characters"

            # Prevent division by zero
            if "/0" in expression:
                return None, "Division by zero detected"

            # Evaluate
            result = eval(expression, {"__builtins__": {}})

            # Type check
            if not isinstance(result, (int, float)):
                return None, "Calculation resulted in non-numeric type"

            logger.info(f"Calculation successful: {expression} = {result}")
            return result, None

        except ZeroDivisionError:
            return None, "Division by zero error"
        except SyntaxError:
            return None, "Invalid mathematical expression"
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}")
            return None, f"Calculation failed: {str(e)}"


# ============================================================================
# CONVERSATION CONTROLLER
# ============================================================================

class ConversationController:
    """Main orchestrator for multi-turn conversations"""

    def __init__(self):
        self.memory = ConversationMemory()
        self.parser = IntentParser()
        self.planner = Planner()
        self.calculator = CalculatorTool()

    def process_turn(self, user_message: str) -> str:
        """
        Process one conversation turn end-to-end.
        Flow: Parse → Extract Entities → Plan → Execute → Respond
        """
        logger.info(f"Processing turn {self.memory.get_turn_count() + 1}: {user_message}")

        try:
            # Step 1: Parse intent
            intent, confidence = self.parser.parse(user_message, self.memory)
            
            # If UNKNOWN intent, try to use previous intent if available
            if intent == IntentType.UNKNOWN and self.memory.get_last_turn():
                previous_intent = self.memory.get_last_turn().intent
                # Try to map string back to IntentType
                try:
                    intent = IntentType(previous_intent)
                    confidence = 0.5  # Lower confidence for inferred intent
                    logger.info(f"Inferred intent from context: {intent.value}")
                except ValueError:
                    pass

            # Step 2: Extract entities
            self._extract_entities(user_message, intent)

            # Step 3: Plan action
            action = self.planner.plan(intent, self.memory)

            # Step 4: Execute or ask for clarification
            if action.needs_clarification():
                bot_response = action.next_question
                action_taken = "ask_clarification"
            else:
                bot_response, action_taken = self._execute_action(action, intent, user_message)

            # Step 5: Update state
            self.memory.set_state(ConversationState.COMPLETED)

            # Step 6: Store turn in memory
            self.memory.add_turn(
                user_msg=user_message,
                bot_response=bot_response,
                intent=intent.value,
                action=action_taken
            )

            return bot_response

        except Exception as e:
            logger.error(f"Error processing turn: {str(e)}")
            self.memory.set_state(ConversationState.ERROR)
            return f"I encountered an error: {str(e)}. Please try again."

    def _extract_entities(self, user_msg: str, intent: IntentType) -> None:
        """Extract slots from user message"""
        msg_lower = user_msg.lower()

        # Location extraction
        locations = ["petaling jaya", "pj", "klang", "shah alam"]
        for loc in locations:
            if loc in msg_lower:
                self.memory.update_slots(location=loc)
                break

        # Outlet name extraction
        outlet_keywords = {"ss 2": "SS 2", "klang main": "Klang Main", "shah alam": "Shah Alam"}
        for keyword, name in outlet_keywords.items():
            if keyword in msg_lower:
                self.memory.update_slots(outlet_name=name)
                break

        # Calculation extraction
        if intent == IntentType.CALCULATE:
            expr = CalculatorTool.extract_expression(user_msg)
            if expr:
                self.memory.update_slots(calculation_expression=expr)

        # Product search extraction
        if intent == IntentType.PRODUCT_INQUIRY:
            self.memory.update_slots(product_search_term=user_msg)

    def _execute_action(self, action: Action, intent: IntentType, user_msg: str) -> Tuple[str, str]:
        """Execute the planned action"""

        if intent == IntentType.FIND_OUTLET:
            location = self.memory.slots.location
            return f"Found outlets in {location}! Which one would you like to know more about?", "search_outlets"

        elif intent == IntentType.GET_HOURS:
            outlet = self.memory.slots.outlet_name
            if outlet == "SS 2":
                return f"The {outlet} outlet opens at 9:00 AM - 10:00 PM daily.", "get_outlet_hours"
            else:
                return f"Hours for {outlet}: Please contact us for specific times.", "get_outlet_hours"

        elif intent == IntentType.GET_ADDRESS:
            outlet = self.memory.slots.outlet_name
            return f"Address for {outlet}: Please contact us for directions.", "get_address"

        elif intent == IntentType.CALCULATE:
            expr = self.memory.slots.calculation_expression
            result, error = self.calculator.calculate(expr)
            if error:
                return f"Calculation failed: {error}", "calculator_error"
            return f"The result of {expr} is {result}.", "calculator_success"

        elif intent == IntentType.PRODUCT_INQUIRY:
            return "We offer a variety of products! What specifically are you looking for?", "product_inquiry"

        elif intent == IntentType.COMPLAINT:
            return "I'm sorry to hear that. Could you tell me more about the issue so we can help?", "complaint_received"

        elif intent == IntentType.GREETING:
            return "Hello! How can I help you today? You can ask about our outlets, hours, or products.", "greeting"

        else:
            return "I'm here to help! Ask me about outlets, hours, calculations, or products.", "unknown_intent"

    def get_memory_snapshot(self) -> Dict:
        """Get current conversation state"""
        return self.memory.to_dict()

    def reset(self) -> None:
        """Reset conversation"""
        self.memory.reset()
        logger.info("Conversation reset")


# ============================================================================
# TESTING
# ============================================================================

def test_happy_path():
    """Test happy path: multi-turn conversation"""
    print("\n" + "="*70)
    print("TEST 1: HAPPY PATH - Sequential Conversation (3 turns)")
    print("="*70)

    controller = ConversationController()

    # Turn 1
    print("\n📝 Turn 1:")
    msg1 = "Is there an outlet in Petaling Jaya?"
    resp1 = controller.process_turn(msg1)
    print(f"User: {msg1}")
    print(f"Bot:  {resp1}")
    print(f"Memory: location={controller.memory.slots.location}")

    # Turn 2
    print("\n📝 Turn 2:")
    msg2 = "What about the SS 2 outlet?"
    resp2 = controller.process_turn(msg2)
    print(f"User: {msg2}")
    print(f"Bot:  {resp2}")
    print(f"Memory: outlet={controller.memory.slots.outlet_name}")

    # Turn 3
    print("\n📝 Turn 3:")
    msg3 = "What's the opening time?"
    resp3 = controller.process_turn(msg3)
    print(f"User: {msg3}")
    print(f"Bot:  {resp3}")

    print(f"\n✅ Conversation complete. Total turns: {controller.memory.get_turn_count()}")


def test_calculator():
    """Test calculator tool integration"""
    print("\n" + "="*70)
    print("TEST 2: CALCULATOR TOOL")
    print("="*70)

    controller = ConversationController()

    # Successful calculation
    print("\n✓ Success Case:")
    msg = "Can you calculate 15 + 25 * 2?"
    resp = controller.process_turn(msg)
    print(f"User: {msg}")
    print(f"Bot:  {resp}")

    # Failed calculation (division by zero)
    print("\n✗ Error Case - Division by Zero:")
    controller.reset()
    msg = "Calculate 10 / 0"
    resp = controller.process_turn(msg)
    print(f"User: {msg}")
    print(f"Bot:  {resp}")


def test_interrupted_flow():
    """Test handling of ambiguous/incomplete requests"""
    print("\n" + "="*70)
    print("TEST 3: INTERRUPTED/AMBIGUOUS FLOW")
    print("="*70)

    controller = ConversationController()

    print("\n🔄 User asks vague question:")
    msg1 = "Show me the opening times"
    resp1 = controller.process_turn(msg1)
    print(f"User: {msg1}")
    print(f"Bot:  {resp1}")
    print("Note: Bot asks for clarification (location needed)")

    print("\n📝 User provides missing info:")
    msg2 = "For SS 2 in Petaling Jaya"
    resp2 = controller.process_turn(msg2)
    print(f"User: {msg2}")
    print(f"Bot:  {resp2}")


if __name__ == "__main__":
    test_happy_path()
    test_calculator()
    test_interrupted_flow()

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)