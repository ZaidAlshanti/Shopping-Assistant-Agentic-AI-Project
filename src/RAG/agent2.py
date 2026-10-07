import asyncio
import os
import sys
import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
import dotenv

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Initialize Model
model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.2
)

# Define State
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Load environment variables
dotenv.load_dotenv()

async def main():
    # =========================================================
    # Start MCP server
    # =========================================================
    server_path = os.path.join(
        os.path.dirname(__file__),
        "server.py",
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path],
        env=os.environ.copy(),
    )

    print("Connecting to MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # =================================================
            # Initialize MCP
            # =================================================
            await session.initialize()

            # =================================================
            # Load & Verify MCP tools
            # =================================================
            response = await session.list_tools()
            print(f"Connected. Found {len(response.tools)} MCP tools.")
            
            # FIXED: Moved inside the async session context block
            tools = load_mcp_tools(session)

            print("\nAvailable tools:")
            for t in tools:
                print(f"- {t.name}")

            # Create a dictionary of tools by name
            tools_by_name = {t.name: t for t in tools}
            model_with_tools = model.bind_tools(tools_by_name.values())

            # =================================================
            # Define Graph Nodes & Conditional Logic
            # =================================================
            def llm_call(state: dict):
                """LLM decides whether to call a tool or not"""
                sys_msg = SystemMessage(
                    content="""You are AuraTech's main shopping assistant.

Use only the information from the get_stock_and_compatible_lookup, get_order_status, set_returns, search_knowledge_base, and write_reports tools.

You can answer questions in Arabic and English.

You have access to product, order, knowledge-base, reporting, and specialist subagents.

Use the appropriate tools or subagents whenever information must come from AuraTech's systems.

Never invent:
- Product information
- Prices
- Stock
- Specifications
- Reviews
- Policies
- Orders
- Return information"""
                )
                
                response_msg = model_with_tools.invoke([sys_msg] + state["messages"])
                return {
                    "messages": [response_msg],
                    "llm_calls": state.get('llm_calls', 0) + 1
                }

            def tool_node(state: dict):
                """Performs the tool call"""
                result = []
                for tool_call in state["messages"][-1].tool_calls:
                    selected_tool = tools_by_name[tool_call["name"]]
                    observation = selected_tool.invoke(tool_call["args"])
                    result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
                return {"messages": result}            

            def should_continue(state: MessagesState) -> Literal["tool_node", END]:
                """Decide if we should continue the loop or stop"""
                last_message = state["messages"][-1]
                if last_message.tool_calls:
                    return "tool_node"
                return END            

            # =================================================
            # Build & Compile Workflow
            # =================================================
            agent_builder = StateGraph(MessagesState)

            # Add nodes
            agent_builder.add_node("llm_call", llm_call)
            agent_builder.add_node("tool_node", tool_node)

            # Add edges to connect nodes
            agent_builder.add_edge(START, "llm_call")
            agent_builder.add_conditional_edges(
                "llm_call",
                should_continue,
                ["tool_node", END]
            )
            agent_builder.add_edge("tool_node", "llm_call")

            # Compile the agent
            agent = agent_builder.compile()

            # =================================================
            # Invoke Agent Workflow
            # =================================================
            initial_messages = [HumanMessage(content="how can I clean my laptop?")]
            output_state = agent.invoke({"messages": initial_messages, "llm_calls": 0})
            
            print("\n--- Execution History ---")
            for m in output_state["messages"]:
                m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())
