# ─── LangGraph ReAct Agent ────────────────────────────────────────────────────
import os
import math as mathlib
import requests
from typing import TypedDict, List, Any

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END


# ─── 1. State Definition ──────────────────────────────────────────────────────
class AgentState(TypedDict):
    input: str
    agent_scratchpad: List[Any]   # running list of AI + Tool messages
    final_answer: str
    steps: List[str]              # human-readable action/observation log


# ─── Tool: Weather ────────────────────────────────────────────────────────────
_CITY_COORDS = {
    "london":     (51.5074,  -0.1278),
    "paris":      (48.8566,   2.3522),
    "new york":   (40.7128, -74.0060),
    "tokyo":      (35.6762, 139.6503),
    "karachi":    (24.8607,  67.0011),
    "lahore":     (31.5204,  74.3587),
    "islamabad":  (33.6844,  73.0479),
    "rawalpindi": (33.5651,  73.0169),
    "dubai":      (25.2048,  55.2708),
    "berlin":     (52.5200,  13.4050),
    "sydney":    (-33.8688, 151.2093),
    "chicago":    (41.8781, -87.6298),
}


@tool
def get_current_weather(city: str) -> str:
    """Get real-time current weather for a city using the free Open-Meteo API.
    Returns temperature, wind speed, humidity, and sky condition.
    Available cities: London, Paris, New York, Tokyo, Karachi, Lahore,
    Islamabad, Rawalpindi, Dubai, Berlin, Sydney, Chicago."""
    coords = _CITY_COORDS.get(city.lower().strip())
    if not coords:
        available = ", ".join(c.title() for c in _CITY_COORDS)
        return f"City '{city}' not found. Available cities: {available}"
    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=relativehumidity_2m,apparent_temperature"
    )
    try:
        data    = requests.get(url, timeout=5).json()
        cw      = data.get("current_weather", {})
        temp    = cw.get("temperature", "N/A")
        wind    = cw.get("windspeed",   "N/A")
        wcode   = cw.get("weathercode",  0)
        cond    = "Sunny" if wcode < 3 else "Cloudy" if wcode < 50 else "Rainy"
        humidity = data.get("hourly", {}).get("relativehumidity_2m",  ["N/A"])[0]
        feels    = data.get("hourly", {}).get("apparent_temperature", ["N/A"])[0]
        return (
            f"Current weather in {city.title()}:\n"
            f"  Condition : {cond}\n"
            f"  Temp      : {temp}°C\n"
            f"  Feels like: {feels}°C\n"
            f"  Wind      : {wind} km/h\n"
            f"  Humidity  : {humidity}%"
        )
    except requests.Timeout:
        return f"Weather API timed out for '{city}'"
    except Exception as exc:
        return f"Weather API error: {exc}"


# ─── Tool: Web Search ─────────────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web for real-time information about people, facts, news,
    or general knowledge. Use this for anything factual."""
    try:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return (
                f"Search unavailable: TAVILY_API_KEY environment variable not set. "
                f"Cannot search for: {query}"
            )
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        results  = response.get("results", [])
        if not results:
            return f"No results found for: '{query}'"
        return "\n\n".join(
            f"[{i+1}] {r['title']}\n    {r['content']}"
            for i, r in enumerate(results)
        )
    except ImportError:
        return "Search unavailable: tavily package not installed."
    except Exception as exc:
        return f"Search error: {exc}"


# ─── Tool: Calculator ─────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, pi, e, abs, round.
    Examples: '2025 - 1958', 'sqrt(144)', '15 * 8 + 20'"""
    try:
        safe_globals = {
            "__builtins__": {},
            "sqrt":  mathlib.sqrt,
            "log":   mathlib.log,
            "log2":  mathlib.log2,
            "log10": mathlib.log10,
            "sin":   mathlib.sin,
            "cos":   mathlib.cos,
            "tan":   mathlib.tan,
            "ceil":  mathlib.ceil,
            "floor": mathlib.floor,
            "pi":    mathlib.pi,
            "e":     mathlib.e,
            "abs":   abs,
            "round": round,
            "pow":   pow,
        }
        result = eval(expression, safe_globals)  # noqa: S307
        return f"{expression} = {round(float(result), 6)}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except NameError as err:
        return f"Error: Unknown function — {err}"
    except SyntaxError:
        return f"Error: Invalid syntax in '{expression}'"
    except Exception as err:
        return f"Error evaluating '{expression}': {err}"


# ─── Tool Registry ────────────────────────────────────────────────────────────
TOOLS     = [get_current_weather, search_web, calculator]
TOOLS_MAP = {t.name: t for t in TOOLS}


# ─── LLM Setup ────────────────────────────────────────────────────────────────
REACT_SYSTEM = """You are a ReAct agent. Strictly follow this loop:
Thought → Action (tool call) → Observation → Thought → ...

RULES:
1. ALWAYS use a tool for factual information — never answer from memory.
2. For multi-part questions, make one tool call per fact.
3. ALWAYS use the calculator tool for any arithmetic — never compute in your head.
4. Only give a Final Answer AFTER all required tool calls are complete."""

_llm_with_tools = None


def _get_llm_with_tools():
    """Lazily initialise the LLM so the module can be imported without an API key."""
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )
        _llm_with_tools = llm.bind_tools(TOOLS)
    return _llm_with_tools


# ─── 2. ReAct Node (Reasoning + Action) ──────────────────────────────────────
def react_node(state: AgentState) -> AgentState:
    """Call the LLM with the current state. Produces either a tool call or
    a final answer, and updates the scratchpad accordingly."""
    messages = [
        SystemMessage(content=REACT_SYSTEM),
        HumanMessage(content=state["input"]),
    ]
    messages.extend(state.get("agent_scratchpad", []))

    response = _get_llm_with_tools().invoke(messages)

    scratchpad = list(state.get("agent_scratchpad", []))
    scratchpad.append(response)

    if not response.tool_calls:
        # No more tool calls → final answer reached
        return {**state, "agent_scratchpad": scratchpad, "final_answer": response.content}

    return {**state, "agent_scratchpad": scratchpad}


# ─── 3. Tool Execution Node ───────────────────────────────────────────────────
def tool_node(state: AgentState) -> AgentState:
    """Execute every tool call found in the last AI message and append the
    Observations to the scratchpad so the LLM can reason further."""
    scratchpad = list(state.get("agent_scratchpad", []))
    steps      = list(state.get("steps", []))
    last_msg   = scratchpad[-1]

    for tc in last_msg.tool_calls:
        print(f"   Tool      : [{tc['name']}]")
        print(f"   Args      : {tc['args']}")
        result      = TOOLS_MAP[tc["name"]].invoke(tc["args"])
        observation = str(result)
        print(f"   Observation: {observation[:300]}")
        steps.append(
            f"Action: {tc['name']}({tc['args']})\n"
            f"Observation: {observation}"
        )
        scratchpad.append(
            ToolMessage(content=observation, tool_call_id=tc["id"])
        )

    return {**state, "agent_scratchpad": scratchpad, "steps": steps}


# ─── 4 & 5. Conditional Router ────────────────────────────────────────────────
def router(state: AgentState) -> str:
    """Route to 'tool_node' when the last AI message contains tool calls,
    otherwise route to END (final answer)."""
    scratchpad = state.get("agent_scratchpad", [])
    if scratchpad:
        last_msg = scratchpad[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tool_node"   # is_action
    return END                   # is_final


# ─── Graph Construction ───────────────────────────────────────────────────────
def build_graph():
    """Build and compile the LangGraph ReAct workflow.

    Flow:
        START → react_node → [Conditional Edge]
            • Action      → tool_node → react_node  (loop)
            • Final Answer → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("react_node", react_node)
    workflow.add_node("tool_node",  tool_node)

    workflow.add_edge(START, "react_node")
    workflow.add_conditional_edges(
        "react_node",
        router,
        {"tool_node": "tool_node", END: END},
    )
    workflow.add_edge("tool_node", "react_node")

    return workflow.compile()


graph = build_graph()
