from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)
import os

import sys

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from src.RAG.agent import build_agent

from langsmith import utils, Client
print("TRACING:", os.getenv("LANGSMITH_TRACING"), "/", os.getenv("LANGCHAIN_TRACING_V2"))
print("KEY set:", bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")))
print("tracing_is_enabled:", utils.tracing_is_enabled())
c = Client()
print("endpoint:", c.api_url)
print("key works, projects:", [p.name for p in c.list_projects(limit=3)])



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(BASE_DIR, "src", "RAG", "server.py")
assert os.path.exists(SERVER_PATH), f"server.py not found at {SERVER_PATH}"
TEMPLATES_DIR = os.path.join(BASE_DIR, "src", "templates")  # adjust if needed


@asynccontextmanager
async def lifespan(app: FastAPI):
    params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=os.environ.copy(),
    )

    # AsyncExitStack keeps the MCP connection open for the whole app lifetime
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools = await load_mcp_tools(session)
        app.state.tools_by_name = {t.name: t for t in tools}
        app.state.agent = build_agent(tools, MemorySaver())
        app.state.pending = {}   # session_id -> number of actions awaiting approval

        print(f"MCP connected, {len(tools)} tools loaded")
        yield
    # leaving the block closes the MCP session and stops server.py


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class ChatIn(BaseModel):
    session_id: str
    message: str


class DecisionIn(BaseModel):
    session_id: str
    decision: str  # "approve" | "reject"


async def run_agent(app: FastAPI, session_id: str, payload):
    config = {"configurable": {"thread_id": session_id}}
    result = await app.state.agent.ainvoke(payload, config=config)

    # Paused on set_returns: ask the UI for approval
    if result.get("__interrupt__"):
        value = result["__interrupt__"][0].value
        actions = value.get("action_requests", []) if isinstance(value, dict) else []
        app.state.pending[session_id] = max(len(actions), 1)
        return {
            "type": "approval",
            "actions": [
                {
                    "name": a.get("name"),
                    "args": a.get("args"),
                    "description": a.get("description"),
                }
                for a in actions
            ] or [{"name": "set_returns", "args": {}, "description": str(value)}],
        }

    final = result["messages"][-1].content

    # Same as your CLI loop: record the interaction
    record_tool = app.state.tools_by_name.get("record_response")
    if record_tool:
        last_user = next(
            (m.content for m in reversed(result["messages"]) if m.type == "human"),
            "",
        )
        try:
            await record_tool.ainvoke(
                {"user_query": last_user, "agent_response": final}
            )
        except Exception as e:
            print(f"Warning: could not record response: {e}")

    return {"type": "message", "content": final}


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/chat")
async def chat(body: ChatIn):
    try:
        payload = {"messages": [HumanMessage(content=body.message)]}
        return await run_agent(app, body.session_id, payload)
    except Exception as e:
        return JSONResponse({"type": "error", "content": str(e)}, status_code=500)


@app.post("/api/decision")
async def decision(body: DecisionIn):
    count = app.state.pending.pop(body.session_id, 0)
    if not count:
        return JSONResponse(
            {"type": "error", "content": "No pending approval."}, status_code=400
        )

    if body.decision == "approve":
        decisions = [{"type": "approve"} for _ in range(count)]
    else:
        decisions = [
            {"type": "reject", "message": "Return request rejected by the user."}
            for _ in range(count)
        ]

    try:
        return await run_agent(
            app, body.session_id, Command(resume={"decisions": decisions})
        )
    except Exception as e:
        return JSONResponse({"type": "error", "content": str(e)}, status_code=500)