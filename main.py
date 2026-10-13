import asyncio
import os
import sys

from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import find_dotenv, load_dotenv

# Load environment variables before importing components
# that depend on them.
load_dotenv(find_dotenv(usecwd=True), override=True)

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools

from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.types import Command

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic import BaseModel

from src.RAG.agent import build_agent
from src.RAG.upload import handle_file_upload, ingest_text




# ============================================================
# Paths
# ============================================================
session_memories = {} 

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

        app.state.tools_by_name = {
            tool.name: tool
            for tool in tools
        }

        app.state.agent = build_agent(
            tools,
            MemorySaver()
        )

        # session_id -> number of pending approval actions
        app.state.pending = {}

        print(
            f"MCP connected, {len(tools)} tools loaded"
        )

        yield

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
# Agent execution
# ============================================================

async def run_agent(
    app: FastAPI,
    session_id: str,
    payload,
):

    config = {
        "configurable": {
            "thread_id": session_id
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

    # Record interaction using MCP tool.
    record_tool = app.state.tools_by_name.get(
        "record_response"
    )

    if record_tool:

        last_user = next(
            (
                message.content
                for message in reversed(result["messages"])
                if message.type == "human"
            ),
            "",
        )

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

@app.get(
    "/",
    response_class=HTMLResponse
)
def read_root(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


# ============================================================
# Chat
# ============================================================

@app.post("/api/chat")
async def chat(body: ChatIn):

    try:

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
    if session_id in session_memories:
        # Option A: If using a standard LangChain memory object
        # session_memories[session_id].clear()
        
        # Option B: Completely delete or overwrite the session's history
        session_memories[session_id] = [] 
        
        return {"status": "success", "message": f"Memory cleared for session {session_id}"}
    
    return {"status": "success", "message": "No active session memory to clear."}
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