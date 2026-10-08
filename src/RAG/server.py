import os
import sys
import json

import pandas as pd
import oracledb


from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel


# Create a project (called a "Workspace Dataset" or "Project" in the UI)

# ============================================================
# MCP
# ============================================================

mcp = FastMCP("ShopAssistant")


# ============================================================
# Qdrant
# ============================================================

print(
    "Loading Embedding Model...",
    file=sys.stderr,
    flush=True,
)

embedding_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "shopassist_knowledge",
)

qdrant_client = QdrantClient(
    url=QDRANT_URL
)


# ============================================================
# Product catalogue
# ============================================================



# ============================================================
# Product lookup
# ============================================================

@mcp.tool()
def get_stock_and_compatible_lookup(
    product_name: str,
) -> str:
    """
        Get the products info.
    """
    
    df=pd.read_csv("Data\\productCatalouge.csv",index_col="product_id")
    return df.loc[df['product_name'] == product_name].to_string()


# ============================================================
# Order status
# ============================================================

@mcp.tool()
def get_order_status(
    customer_id: str,
) -> str:

    """
    Get the order status for a customer.
    """

    customer_id = (
        customer_id
        .strip()
        .strip("\"'")
        .upper()
    )


    try:

        with open(
            "Data\\Orders.json",
            "r",
            encoding="utf-8",
        ) as f:

            orders_data = json.load(f)

    except Exception as e:

        return (
            f"Could not read order data: {e}"
        )

    customer_orders = [
        order
        for order in orders_data.get(
            "orders",
            [],
        )
        if order.get("customer_id") == customer_id
    ]

    if not customer_orders:

        return (
            "No order found for the "
            "specified customer."
        )

    lines = []

    for order in customer_orders:

        lines.append(
            f"Order for Customer ID: "
            f"{order['customer_id']}"
        )

        lines.append(
            f"Status: {order['status']}"
        )

        items = order.get("items")

        if items:

            lines.append("Items:")

            for item in items:
                lines.append(
                    f"- {item}"
                )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# Return request
# ============================================================

@mcp.tool()
def set_returns(
    customer_id: str,
    reason: str
) -> str:

    """
    Set a return or refund request.

    This is currently a placeholder implementation.
    """
    

    customer_id = (
        customer_id
        .strip()
        .strip("\"'")
        .upper()
    )

    reason = reason.strip()

    return_dir = os.path.join(
        os.path.dirname(__file__),
        "ReturnRequests",
    )

    os.makedirs(
        return_dir,
        exist_ok=True,
    )

    return_path = os.path.join(
        return_dir,
        "Returns.txt",
    )

    with open(
        return_path,
        "a",
        encoding="utf-8",
    ) as log_file:

        log_file.write(
            (
                "Return/Refund Request - "
                f"Customer ID: {customer_id}, "
                f"Reason: {reason}\n"
            )
        )

    return (
        f"Return/refund request for "
        f"Customer ID {customer_id} "
        f"has been recorded. "
        f"Reason: {reason}"
    )


# ============================================================
# Knowledge base
# ============================================================

@mcp.tool()
def search_knowledge_base(
    query: str,
    top_k: int = 4,
) -> str:

    """
    Search the AuraTech knowledge base for:

    - return policies
    - shipping information
    - customer reviews
    - help articles
    """

    try:

        query_vector = embedding_model.encode(
            query,
            batch_size=1,
            max_length=512,
        )["dense_vecs"].tolist()

        search_results = (
            qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
            )
            .points
        )

    except Exception as e:

        print(
            f"Knowledge-base search failed: {e}",
            file=sys.stderr,
            flush=True,
        )

        return (
            "The knowledge base could not be "
            "searched."
        )

    if not search_results:

        return (
            "No relevant information found "
            "in the knowledge base."
        )

    formatted_results = []

    for i, hit in enumerate(search_results):

        payload = hit.payload

        citation = (
            f"[{payload.get('source', 'Unknown')} "
            f"> "
            f"{payload.get('section', 'Unknown')}]"
        )

        chunk_info = (
            f"--- Document {i + 1} ---\n"
            f"Citation: {citation}\n"
            f"Type: "
            f"{payload.get('type', 'Unknown').upper()}\n"
            f"Content: "
            f"{payload.get('text', '')}\n"
        )

        formatted_results.append(
            chunk_info
        )

    system_instruction = (
        "When answering using this data, "
        "preserve the exact Citation string "
        "provided for relevant information."
    )

    return (
        "\n".join(formatted_results)
        + "\n\n"
        + system_instruction
    )


# ============================================================
# Record response
# ============================================================

@mcp.tool()
def record_response(
    user_query: str,
    agent_response: str,
) -> str:

    """
    Record the user's query and the agent's response.
    """

    evaluation_dir = os.path.join(
        os.path.dirname(__file__),
        "Evaluation",
    )

    os.makedirs(
        evaluation_dir,
        exist_ok=True,
    )

    response_path = os.path.join(
        evaluation_dir,
        "Responses.txt",
    )

    with open(
        response_path,
        "a",
        encoding="utf-8",
    ) as log_file:

        log_file.write(
            f"User Query: {user_query}\n"
        )

        log_file.write(
            f"Agent Response: {agent_response}\n"
        )

        log_file.write(
            "-------------------------------------------\n"
        )

    return (
        "Interaction recorded successfully."
    )


# ============================================================
# Write reports
# ============================================================

@mcp.tool()
def write_reports(
    title: str,
    content: str,
) -> str:

    """
    Write a report based on the provided content.
    """

    reports_dir = os.path.join(
        os.path.dirname(__file__),
        "Reports",
    )

    os.makedirs(
        reports_dir,
        exist_ok=True,
    )

    # Make title safe for Windows filenames
    safe_title = "".join(
        c if c.isalnum() or c in " _-"
        else "_"
        for c in title
    ).strip()

    if not safe_title:
        safe_title = "report"

    report_path = os.path.join(
        reports_dir,
        f"{safe_title}.txt",
    )

    try:

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(content)

    except OSError as e:

        return (
            f"Failed to write report: {e}"
        )

    return (
        f"Report generated successfully: "
        f"{report_path}"
    )


# ============================================================
# Start MCP server
# ============================================================

if __name__ == "__main__":
    mcp.run()