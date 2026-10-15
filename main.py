import asyncio
import os
import sys
from typing_extensions import final
import uuid

from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import find_dotenv, load_dotenv

# Load environment variables before importing components
# that depend on them.
load_dotenv(find_dotenv(usecwd=True), override=True)

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic import BaseModel

from src.RAG.agent import build_agent
from src.RAG.upload import handle_file_upload, ingest_text
from src.RAG.session_memory import SessionRecorder
from src.RAG.cache.semantic_cache import SemanticCache
from login import signin, signup
from src.RAG.safety import check_input, check_output



# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVER_PATH = os.path.join(
    BASE_DIR,
    "src",
    "RAG",
    "server.py"
)

TEMPLATES_DIR = os.path.join(
    BASE_DIR,
    "src",
    "templates"
)

assert os.path.exists(
    SERVER_PATH
), f"server.py not found at {SERVER_PATH}"



# ============================================================
# FastAPI lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=os.environ.copy(),
    )

    # Keep the MCP connection alive for the entire
    # FastAPI application lifetime.
    async with AsyncExitStack() as stack:

        read, write = await stack.enter_async_context(
            stdio_client(params)
        )

        session = await stack.enter_async_context(
            ClientSession(read, write)
        )

        await session.initialize()

        tools = await load_mcp_tools(session)

        app.state.tools = tools
        app.state.tools_by_name = {
            tool.name: tool
            for tool in tools
        }

        app.state.checkpointer = MemorySaver()      # one shared checkpointer
        app.state.recorder = SessionRecorder(
            ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )
        app.state.cache = SemanticCache()
        app.state.threads = {}      # session_id -> current thread_id

        # session_id -> number of pending approval actions
        app.state.pending = {}

        # summarise journals left by a crashed run BEFORE building the agent
        await app.state.recorder.close_all()

        app.state.agent = build_agent(
            tools,
            app.state.checkpointer,
        )

        print(
            f"MCP connected, {len(tools)} tools loaded"
        )

        yield

        # system shutdown: summarise every active session
        await app.state.recorder.close_all()

    # MCP connection closes here.


# ============================================================
# Application
# ============================================================

app = FastAPI(
    lifespan=lifespan
)

templates = Jinja2Templates(
    directory=TEMPLATES_DIR
)


# ============================================================
# Request models
# ============================================================

class ChatIn(BaseModel):
    session_id: str
    message: str


class DecisionIn(BaseModel):
    session_id: str
    decision: str


# ============================================================
# Cache helpers
# ============================================================

IGNORED_TOOLS = {"set_returns", "write_reports"}

CACHEABLE_TOOLS = {
    "search_knowledge_base",
    "get_stock_and_compatible_lookup",
    "get_order_status",
    "record_response",
}

MIN_CACHE_WORDS = 5


def is_self_contained(q: str) -> bool:
    # Very short messages ("and shipping?", "what about it?") rely on earlier turns.
    return len(q.split()) >= MIN_CACHE_WORDS


def tools_used_this_turn(messages) -> set:
    last_human = max(
        (i for i, m in enumerate(messages) if m.type == "human"),
        default=-1,
    )
    return {
        tc["name"]
        for m in messages[last_human + 1:]
        for tc in (getattr(m, "tool_calls", None) or [])
    }


# ============================================================
# Agent execution
# ============================================================

async def run_agent(
    app: FastAPI,
    session_id: str,
    payload,
    cache_query: str | None = None,
):

    thread_id = app.state.threads.get(session_id, session_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = await app.state.agent.ainvoke(
        payload,
        config=config
    )

    # --------------------------------------------------------
    # Human approval interrupt
    # --------------------------------------------------------

    if result.get("__interrupt__"):

        value = result["__interrupt__"][0].value

        actions = (
            value.get("action_requests", [])
            if isinstance(value, dict)
            else []
        )

        app.state.pending[session_id] = max(
            len(actions),
            1
        )

        return {
            "type": "approval",
            "actions": [
                {
                    "name": action.get("name"),
                    "args": action.get("args"),
                    "description": action.get("description"),
                }
                for action in actions
            ]
            or [
                {
                    "name": "set_returns",
                    "args": {},
                    "description": str(value),
                }
            ],
        }

    # --------------------------------------------------------
    # Normal final response
    # --------------------------------------------------------

    final = result["messages"][-1].content

    last_user = next(
        (
            message.content
            for message in reversed(result["messages"])
            if message.type == "human"
        ),
        "",
    )

    allowed, final = await check_output(str(final))

    if not allowed:
        return {"type": "message", "content": final}

    # Session log used for the end-of-session summary.
    app.state.recorder.log(session_id, "user", last_user)
    app.state.recorder.log(session_id, "assistant", final)

    # Cache answers that only used cacheable tools.
    used = tools_used_this_turn(result["messages"]) - IGNORED_TOOLS
    print(f"[cache] cache_query={cache_query!r} tools_used={used}")

    if allowed and cache_query and used and used <= CACHEABLE_TOOLS:
        try:
            await asyncio.to_thread(
                app.state.cache.put,
                cache_query,
                final,
            )
        except Exception as e:
            print(
                f"Warning: cache write failed: {e}"
            )

    # Record interaction using MCP tool.
    record_tool = app.state.tools_by_name.get(
        "record_response"
    )

    if record_tool:

        try:
            await record_tool.ainvoke(
                {
                    "user_query": last_user,
                    "agent_response": final,
                }
            )

        except Exception as e:
            print(
                f"Warning: could not record response: {e}"
            )

    return {
        "type": "message",
        "content": final,
    }


# ============================================================
# Main page
# ============================================================
@app.get("/login.js")
async def get_login_js():
    return FileResponse(
        os.path.join(TEMPLATES_DIR, "login.js"),
        media_type="application/javascript"
    )
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
    )
def read_root(request: Request):

    return templates.TemplateResponse(
        request=request,
        name=["index.html", "login.html"],
    )

# ============================================================
# login and signup
# ============================================================
@app.post("/api/signin")
async def login_user(email: str, password: str):
    result = signin(email, password)
    return {"message": result}
@app.post("/api/signup")
async def register_user(name: str, email: str, password: str):
    result = signup(name, email, password)
    return {"message": result}


# ============================================================
# Chat
# ============================================================

@app.post("/api/chat")
async def chat(body: ChatIn):

    try:

        recorder = app.state.recorder
        allowed, message = await check_input(body.message)
        if not allowed:
            return {"type": "message", "content": message}

        cacheable = is_self_contained(message)

        if cacheable:
            try:
                hit = await asyncio.to_thread(
                    app.state.cache.get,
                    body.message,
                )
            except Exception as e:      # Ollama down etc.: skip the cache
                print(f"Warning: cache lookup failed: {e}")
                hit = None

            print(f"[cache] cacheable=True hit={bool(hit)}")

            if hit:
                recorder.log(body.session_id, "user", body.message)
                recorder.log(body.session_id, "assistant", hit)

                # Keep the agent's conversation history in sync, so a later
                # follow-up knows this exchange happened.
                thread_id = app.state.threads.get(
                    body.session_id,
                    body.session_id,
                )
                try:
                    await app.state.agent.aupdate_state(
                        {"configurable": {"thread_id": thread_id}},
                        {
                            "messages": [
                                HumanMessage(content=body.message),
                                AIMessage(content=hit),
                            ]
                        },
                    )
                except Exception as e:
                    print(
                        f"Warning: could not sync cached turn into history: {e}"
                    )

                return {
                    "type": "message",
                    "content": hit,
                }

        payload = {
            "messages": [
                HumanMessage(
                    content=body.message
                )
            ]
        }

        return await run_agent(
            app,
            body.session_id,
            payload,
            cache_query=body.message if cacheable else None,
        )

    except Exception as e:

        return JSONResponse(
            {
                "type": "error",
                "content": str(e),
            },
            status_code=500,
        )


# ============================================================
# Return approval
# ============================================================

@app.post("/api/decision")
async def decision(body: DecisionIn):

    count = app.state.pending.pop(
        body.session_id,
        0,
    )

    if not count:

        return JSONResponse(
            {
                "type": "error",
                "content": "No pending approval.",
            },
            status_code=400,
        )

    if body.decision == "approve":

        decisions = [
            {
                "type": "approve"
            }
            for _ in range(count)
        ]

    elif body.decision == "reject":

        decisions = [
            {
                "type": "reject",
                "message": (
                    "Return request rejected by the user."
                ),
            }
            for _ in range(count)
        ]

    else:

        return JSONResponse(
            {
                "type": "error",
                "content": (
                    "Decision must be 'approve' or 'reject'."
                ),
            },
            status_code=400,
        )

    try:

        return await run_agent(
            app,
            body.session_id,
            Command(
                resume={
                    "decisions": decisions
                }
            ),
        )

    except Exception as e:

        return JSONResponse(
            {
                "type": "error",
                "content": str(e),
            },
            status_code=500,
        )


# ============================================================
# Clear chat session
# ============================================================

@app.post("/api/chat/reset")
async def reset_chat(session_id: str):

    app.state.pending.pop(session_id, None)

    # Summarise this session into Agents.md.
    await app.state.recorder.close(session_id)

    # Fresh conversation: new checkpointer thread.
    app.state.threads[session_id] = (
        f"{session_id}-{uuid.uuid4().hex[:8]}"
    )

    # Rebuild so the new summary is in the prompt. The checkpointer
    # is shared, so other users' sessions are not lost.
    app.state.agent = build_agent(
        app.state.tools,
        app.state.checkpointer,
    )

    return {
        "status": "success",
        "message": f"Session {session_id} summarised and cleared.",
    }


# ============================================================
# Knowledge-base upload
# ============================================================

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    try:

        # Read uploaded file.
        raw = await file.read()

        # Validate and save it under Data/.
        name, text = handle_file_upload(
            file.filename,
            raw,
        )

        # BGE-M3 embedding and Qdrant operations are blocking,
        # so don't block FastAPI's event loop.
        count = await asyncio.to_thread(
            ingest_text,
            text,
            name,
        )

        # Cached answers may now be stale.
        app.state.cache.clear()

    except ValueError as e:

        return JSONResponse(
            {
                "type": "error",
                "content": str(e),
            },
            status_code=400,
        )

    except Exception as e:

        return JSONResponse(
            {
                "type": "error",
                "content": f"Ingestion failed: {e}",
            },
            status_code=500,
        )

    return {
        "type": "success",
        "content": (
            f"'{name}' added to the knowledge base "
            f"({count} chunks)."
        ),
    }