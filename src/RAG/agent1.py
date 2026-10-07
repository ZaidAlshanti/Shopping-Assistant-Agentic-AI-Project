import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from deepagents import create_deep_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

           
            # =================================================
            # Main agent
            # =================================================
def build_agent(tools,checkpointer):
                tools_by_name = {
                                tool.name: tool
                                for tool in tools
                            }
                
                            # =================================================
                            # Main model
                            # =================================================
                
                model = ChatOpenAI(
                                model="gpt-4o-mini",
                                temperature=0.2,
                            )
                comparing_subagent = {
                
                                "name": "comparer-agent",
                
                                "description": (
                                    "Compares products using product specifications, "
                                    "prices, stock information, and customer reviews."
                                ),
                
                                "system_prompt": """
                
                You are AuraTech's product comparison specialist.
                
                Use only the information from the get_stock_and_compatible_lookup
                and search_knowledge_base tools.
                
                Your job is to compare products accurately.
                
                For product specifications, price, stock, and compatibility,
                use get_stock_and_compatible_lookup.
                
                For customer reviews, use search_knowledge_base.
                
                If comparing multiple products, retrieve information for each
                product separately.
                
                Do not invent specifications, prices, reviews, or other product
                information.
                
                Base your comparison only on retrieved information.
                
                Return a clear comparison containing:
                
                - Product information
                - Important specifications
                - Price
                - Stock when relevant
                - Customer review findings
                - Advantages
                - Disadvantages
                - Which product is better for different types of users
                - Overall recommendation
                
                Do not write a file or report yourself.
                Return your findings to the parent agent.
                
                """,
                
                                "tools": [
                                    tools_by_name[
                                        "get_stock_and_compatible_lookup"
                                    ],
                                    tools_by_name[
                                        "search_knowledge_base"
                                    ],
                                ],
                
                                "model": model,
                            }
                
                            # =================================================
                            # Subagent: Report writer
                            # =================================================
                
                writer_subagent = {
                
                                "name": "writer-agent",
                
                                "description": (
                                    "Turns comparison findings into a structured "
                                    "report and provides recommendations."
                                ),
                
                                "system_prompt": """
                
                You are AuraTech's report-writing specialist.
                
                Use only the information from the
                get_stock_and_compatible_lookup and
                search_knowledge_base tools.
                
                You receive findings from another agent.
                
                Turn those findings into a clear, professional report.
                
                The report should contain:
                
                1. Title
                2. Executive summary
                3. Product comparison
                4. Important findings
                5. Advantages and disadvantages
                6. Recommendations
                7. Final conclusion
                
                Do not invent information.
                
                Use only the findings provided to you.
                
                After preparing the report, use write_reports to save it.
                
                The report must contain the actual findings and recommendations,
                not a placeholder.
                
                """,
                
                                "tools": [
                                    tools_by_name[
                                        "search_knowledge_base"
                                    ],
                                    tools_by_name[
                                        "get_stock_and_compatible_lookup"
                                    ],
                                    tools_by_name[
                                        "write_reports"
                                    ],
                                ],
                
                                "model": model,
                            }
                
                            # =================================================
                            # Subagent: Orders / Returns
                            # =================================================
                
                orders_subagent = {
                
                                "name": "orders-agent",
                
                                "description": (
                                    "Handles order status, return policies, orders "
                                    "creation and return/refund requests."
                                ),
                
                                "system_prompt": """
                
                You are AuraTech's order and returns specialist.
                
                Use only the information from the get_order_status,
                set_returns, and search_knowledge_base tools.
                
                You handle:
                
                - Order status
                - Return policies
                - Refund policies
                - Return requests
                
                For order status, use get_order_status.
                
                For return or refund policy questions, use
                search_knowledge_base first.
                
                Never approve or create a return before checking the
                relevant policy when the policy is required.
                
                Only use set_returns when the user's request and the
                available information justify creating a return/refund
                request.
                
                Never invent order information or policies.
                
                IMPORTANT HUMAN APPROVAL RULE:
                
                When the customer requests a return or refund, prepare
                the return request using the available information.
                
                The customer must approve the return before the return
                is actually submitted.
                
                Do not assume customer approval.
                
                """,
                
                                "tools": [
                                    tools_by_name[
                                        "search_knowledge_base"
                                    ],
                                    tools_by_name[
                                        "get_order_status"
                                    ],
                                    tools_by_name[
                                        "set_returns"
                                    ],
                                ],
                
                                "model": model,
                            }
                
                            # =================================================
                            # Subagents
                            # =================================================
                
                subagents = [
                                comparing_subagent,
                                writer_subagent,
                                orders_subagent,
                            ]
                
                            # =================================================
                            # Main agent system prompt
                            # =================================================
                
                system_prompt = """
                
                You are AuraTech's main shopping assistant.
                
                Never answer questions about products, orders, returns,
                or policies based on your own knowledge or assumptions.
                
                Always use the information from:
                
                - get_stock_and_compatible_lookup
                - get_order_status
                - set_returns
                - search_knowledge_base
                
                and nothing else.
                
                You can answer questions in Arabic and English.
                
                You have access to product, order, knowledge-base,
                reporting, and specialist subagents.
                
                Use the appropriate tools or subagents whenever
                information must come from AuraTech's systems.
                
                Never invent:
                
                - Product information
                - Prices
                - Stock
                - Specifications
                - Reviews
                - Policies
                - Orders
                - Return information
                
                For product comparison requests, delegate to
                comparer-agent.
                
                For report-writing requests, use the comparison findings
                and delegate report creation to writer-agent when appropriate.
                
                For order and return requests, delegate to orders-agent.
                
                For policies, shipping, reviews, and help articles,
                use the knowledge base.
                
                When the knowledge base provides citations,
                preserve them in the final answer.
                
                Always answer the user's actual question.
                
                The application automatically records the final response
                using record_response, so you do not need to call that
                tool yourself.
                
                """
                
                return create_deep_agent(

                    model=model,

                    system_prompt=system_prompt,

                    tools=[
                        tools_by_name[tool.name]
                        for tool in tools
                        if tool.name in tools_by_name
                    ],

                    memory=[
                        "./AGENTS.md"
                    ],

                    subagents=subagents,

                    # =================================================
                    # HUMAN-IN-THE-LOOP
                    # =================================================

                    interrupt_on={
                        "set_returns": {

                        

                            "allowed_decisions": [
                                "approve",
                                "reject"
                            ],
                        },
                    },

                    checkpointer=MemorySaver(),
                )

            # =========================================================
            # Interactive loop
            # =========================================================

