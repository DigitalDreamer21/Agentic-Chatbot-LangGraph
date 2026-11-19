import ast
import math
import os
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

load_dotenv()

# ----------------------------- Configuration -----------------------------

BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "faiss_db"
DATABASE_PATH = BASE_DIR / "chatbot.db"

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to the .env file in the project root."
    )

# GPT-OSS 20B handles chat; local Sentence Transformers handles PDF embeddings.
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    max_retries=2,
    api_key=groq_api_key,
)

# Downloads the model on first run, then reuses the local Hugging Face cache.
# No Google/Groq embedding API key is required.
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ),
    model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
    encode_kwargs={"normalize_embeddings": True},
)


def _require_embeddings():
    """Kept as a compatibility guard for the existing PDF RAG functions."""
    if embeddings is None:
        raise RuntimeError("The local embedding model could not be initialized.")


# ----------------------------- PDF RAG -----------------------------

def ingest_rag_document(file_path: str) -> int:
    """Load a PDF, split it, and add its chunks to the persistent FAISS index."""
    _require_embeddings()

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("The uploaded PDF contains no readable pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be extracted from this PDF.")

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    if (FAISS_DIR / "index.faiss").exists() and (FAISS_DIR / "index.pkl").exists():
        vector_store = FAISS.load_local(
            folder_path=str(FAISS_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(str(FAISS_DIR))
    return len(chunks)


def get_retriever():
    """Load the local FAISS index or return a clear error if no PDF was indexed."""
    _require_embeddings()

    index_file = FAISS_DIR / "index.faiss"
    pickle_file = FAISS_DIR / "index.pkl"

    if not index_file.exists() or not pickle_file.exists():
        raise FileNotFoundError(
            "No PDF has been indexed yet. Upload a PDF before asking document questions."
        )

    vector_store = FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=embeddings,
        # Safe only because this index is created locally by this application.
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant passages from the PDF documents uploaded to this chatbot."""
    try:
        documents = get_retriever().invoke(query)
    except (FileNotFoundError, RuntimeError) as error:
        return str(error)
    except Exception as error:
        return f"Document retrieval failed: {error}"

    if not documents:
        return "No relevant information was found in the uploaded PDF documents."

    formatted = []
    for index, document in enumerate(documents, start=1):
        source = Path(document.metadata.get("source", "Unknown source")).name
        page = document.metadata.get("page", "Unknown page")
        # PyPDFLoader page metadata is zero-based.
        if isinstance(page, int):
            page += 1

        formatted.append(
            f"Passage {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted)


# ----------------------------- Tools -----------------------------

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


def _safe_calculate(node):
    if isinstance(node, ast.Expression):
        return _safe_calculate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
        left = _safe_calculate(node.left)
        right = _safe_calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")
        return _ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_calculate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTANTS:
        return _ALLOWED_CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _ALLOWED_FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError("That function is not allowed.")
        return function(*[_safe_calculate(arg) for arg in node.args])
    raise ValueError(
        "Unsupported expression. Use basic arithmetic and sqrt/abs/round/min/max/sum."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression, for example: 2 + 2 or sqrt(81)."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_calculate(parsed)
        return str(result)
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest available stock quote from Alpha Vantage for a ticker symbol."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is missing from the .env file."}

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.strip().upper(),
                "apikey": api_key,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}
        if "Note" in data or "Information" in data:
            return {"error": data.get("Note") or data.get("Information")}

        quote = data.get("Global Quote")
        if not quote:
            return {"error": f"No quote was returned for symbol {symbol}."}
        return quote
    except requests.RequestException as error:
        return {"error": f"Stock service request failed: {error}"}
    except ValueError:
        return {"error": "Stock service returned an invalid JSON response."}


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate a stock purchase after explicit human approval.
    This tool does not place a real trade with a brokerage.
    """
    if quantity <= 0:
        return {"status": "rejected", "message": "Quantity must be greater than zero."}

    normalized_symbol = symbol.strip().upper()
    decision = interrupt(
        f"Approve simulated purchase of {quantity} shares of {normalized_symbol}? (yes/no)"
    )

    if isinstance(decision, str) and decision.strip().lower() in {"yes", "y"}:
        return {
            "status": "success",
            "message": (
                f"Simulated purchase approved for {quantity} shares of "
                f"{normalized_symbol}. No real trade was placed."
            ),
            "symbol": normalized_symbol,
            "quantity": quantity,
        }

    return {
        "status": "cancelled",
        "message": f"Simulated purchase of {quantity} shares of {normalized_symbol} was declined.",
        "symbol": normalized_symbol,
        "quantity": quantity,
    }


@tool
def get_current_weather(location: str) -> str:
    """Get current weather conditions for a city or location using OpenWeather."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing. Set OPENWEATHER_API_KEY in the .env file."

    try:
        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geo_response.raise_for_status()
        locations = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        place = locations[0]
        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": place["lat"],
                "lon": place["lon"],
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        weather = weather_response.json()

        visibility = weather.get("visibility")
        visibility_km = round(visibility / 1000, 1) if visibility is not None else "N/A"
        place_name = ", ".join(
            part
            for part in (
                place.get("name"),
                place.get("state"),
                place.get("country"),
            )
            if part
        )

        return (
            f"Current weather in {place_name}:\n"
            f"- Condition: {weather['weather'][0]['description'].title()}\n"
            f"- Temperature: {weather['main']['temp']}°C\n"
            f"- Feels like: {weather['main']['feels_like']}°C\n"
            f"- Humidity: {weather['main']['humidity']}%\n"
            f"- Pressure: {weather['main']['pressure']} hPa\n"
            f"- Wind speed: {weather.get('wind', {}).get('speed', 'N/A')} m/s\n"
            f"- Visibility: {visibility_km} km"
        )
    except requests.Timeout:
        return "The weather service request timed out. Please try again."
    except requests.HTTPError as error:
        code = error.response.status_code if error.response is not None else "unknown"
        if code == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned HTTP error {code}."
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        return f"Could not retrieve weather information: {error}"


tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
]

llm_with_tools = llm.bind_tools(tools)


# ----------------------------- LangGraph -----------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are a helpful Agentic AI chatbot.\n"
        "Use rag_tool for questions about uploaded PDFs. If no PDF is indexed, "
        "tell the user to upload one. Never invent PDF content.\n"
        "Use search_tool for current or recent information that needs web search.\n"
        "Use calculator for mathematical calculations.\n"
        "Use get_stock_price for current stock quotes.\n"
        "Use purchase_stock only when the user explicitly asks to simulate a stock purchase. "
        "It requires human approval and does not place a real trade.\n"
        "Use get_current_weather for current weather questions.\n"
        "For general questions, answer directly. After using a tool, explain its result clearly. "
        "If a tool returns an error, explain it instead of inventing a result."
    )
)


def chat_node(state: ChatState):
    messages = [SYSTEM_MESSAGE, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# SQLite checkpoint persistence for conversation memory.
_connection = sqlite3.connect(
    database=str(DATABASE_PATH),
    check_same_thread=False,
)
checkpoint = SqliteSaver(_connection)
checkpoint.setup()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads() -> list[str]:
    """Return thread IDs currently saved in the SQLite checkpointer."""
    thread_ids = set()
    for checkpoint_tuple in checkpoint.list(None):
        configurable = checkpoint_tuple.config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if thread_id:
            thread_ids.add(thread_id)
    return sorted(thread_ids)
