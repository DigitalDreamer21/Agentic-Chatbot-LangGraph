"""
Run from the project root with the virtual environment activated:
    python verify_project.py

This script checks configured integrations. It makes live API calls and may consume
free-tier quotas. It never prints API key values.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def status(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

required = ["GOOGLE_API_KEY", "TAVILY_API_KEY", "OPENWEATHER_API_KEY", "ALPHA_VANTAGE_API_KEY"]
for key in required:
    status(f"Environment variable {key}", bool(os.getenv(key)))

status(
    "LangSmith configuration",
    bool(os.getenv("LANGSMITH_API_KEY")) and os.getenv("LANGSMITH_TRACING", "").lower() == "true",
    "Tracing enabled" if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true" else "Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY"
)

try:
    import backend
    status("Backend imports", True)
except Exception as exc:
    status("Backend imports", False, repr(exc))
    sys.exit(1)

# Calculator is local and does not consume an external API quota.
try:
    result = backend.calculator.invoke({"expression": "125 * 48 + 250"})
    status("Calculator", "6250" in str(result), str(result))
except Exception as exc:
    status("Calculator", False, repr(exc))

# Gemini live test.
try:
    answer = backend.llm.invoke("Reply with exactly: GEMINI_OK")
    content = getattr(answer, "content", answer)
    status("Gemini API", bool(content), str(content)[:160])
except Exception as exc:
    status("Gemini API", False, repr(exc))

# Tavily live test.
try:
    result = backend.search_tool.invoke({"query": "LangGraph official documentation"})
    status("Tavily search", bool(result), str(result)[:180])
except Exception as exc:
    status("Tavily search", False, repr(exc))

# Weather live test.
try:
    result = backend.get_current_weather.invoke({"location": "Patna, Bihar, India"})
    ok = "missing" not in str(result).lower() and "invalid" not in str(result).lower() and "error" not in str(result).lower()
    status("OpenWeather", ok, str(result)[:180])
except Exception as exc:
    status("OpenWeather", False, repr(exc))

# Alpha Vantage live test.
try:
    result = backend.get_stock_price.invoke({"symbol": "IBM"})
    ok = "error" not in result
    status("Alpha Vantage", ok, str(result)[:180])
except Exception as exc:
    status("Alpha Vantage", False, repr(exc))

# RAG availability check (actual retrieval requires an uploaded PDF).
try:
    retriever = backend.get_retriever()
    status("FAISS PDF index", True, "Index found; upload/query a PDF in the app to test retrieval.")
except FileNotFoundError as exc:
    status("FAISS PDF index", False, "No PDF indexed yet. Upload a PDF in the app.")
except Exception as exc:
    status("FAISS PDF index", False, repr(exc))

print("\nManual tests still required:")
print("1. Upload a PDF and ask a question whose answer is present in it.")
print("2. Ask to simulate a purchase of 5 IBM shares; approve once, then repeat and reject.")
print("3. In chat A, say 'My name is Pratik'; ask 'What is my name?' in chat A.")
print("4. Create chat B and ask 'What is my name?' It should not inherit chat A's thread memory.")
print("5. Open LangSmith > Agentic-Chatbot and confirm traces arrive after a Gemini request.")
