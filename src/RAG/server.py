import os
import sys
import json

from openai import models
import pandas as pd
import psycopg2
import psycopg2.extras

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient,models
from FlagEmbedding import BGEM3FlagModel


# Create a project (called a "Workspace Dataset" or "Project" in the UI)

# ============================================================
# Connect to Postgres Database
# ============================================================
DB_CONFIG = {
    "host":     os.getenv("PGHOST", "localhost"),
    "port":     int(os.getenv("PGPORT", "5432")),
    "dbname":   os.getenv("PGDATABASE", "postgres"),
    "user":     os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Elephant"),
}


def get_pg_connection():
    """Open a fresh PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)
# ============================================================
# MCP
# ============================================================

mcp = FastMCP("ShopAssistant")

embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)   # fp16 is for GPUs

COLLECTION_NAME = "shopassist_knowledge_v2"          # must match the ingestion script exactly

qdrant_client = QdrantClient(url="http://localhost:6333")



# ============================================================
# Product catalogue
# ============================================================



# ============================================================
# Product lookup
# ============================================================

@mcp.tool()
def get_stock_and_compatible_lookup(product_name: str) -> str:
    """
    Look up a product by its exact product name in the catalogue.
    Use the required argument product_name, for example:
    {"product_name": "AuraBuds Pro"}
    """

    product_name = product_name.strip().strip("\"'")

    if not product_name:
        return "Please provide a product name."

    try:
        with get_pg_connection() as conn:
            with conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:

                # Case-insensitive exact match first
                cur.execute(
                    """
                    SELECT product_id,
                           product_name,
                           specifications,
                           price,
                           stock,
                           compatible_with
                    FROM prods
                    WHERE LOWER(product_name) = LOWER(%s)
                    """,
                    (product_name,),
                )
                rows = cur.fetchall()

                # Fall back to fuzzy match if nothing found
                if not rows:
                    cur.execute(
                        """
                        SELECT product_id,
                               product_name,
                               specifications,
                               price,
                               stock,
                               compatible_with
                        FROM prods
                        WHERE product_name ILIKE %s
                        ORDER BY product_name
                        """,
                        (f"%{product_name}%",),
                    )
                    rows = cur.fetchall()

    except psycopg2.Error as e:
        print(
            f"PostgreSQL lookup failed: {e}",
            file=sys.stderr,
            flush=True,
        )
        return f"Product lookup failed: {e}"

    if not rows:
        return f"No product found matching '{product_name}'."

    lines = []

    for row in rows:
        lines.append(
            f"Product ID:     {row['product_id']}"
        )
        lines.append(
            f"Product Name:   {row['product_name']}"
        )
        lines.append(
            f"Specifications: {row['specifications']}"
        )
        lines.append(
            f"Price:          ${row['price']:.2f}"
        )
        lines.append(
            f"Stock:          {row['stock']} units"
        )
        lines.append(
            f"Compatible With:"
            f" {row['compatible_with'] or 'None'}"
        )
        lines.append("")

    return "\n".join(lines).strip()

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
def search_knowledge_base(query: str) -> str:
    """
    Search the AuraTech knowledge base using hybrid retrieval.

    Searches return policies, shipping information, customer
    reviews, help articles, and other indexed documents.

    Uses BGE-M3 dense and learned sparse vectors with RRF fusion.
    """

    top_k = 3

    try:
        qry = query.strip()

        if not qry:
            return "Please provide a search query."

        # Use the SAME BGE-M3 model used during ingestion.
        encoded = embedding_model.encode(
            [qry],
            batch_size=1,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vector = encoded["dense_vecs"][0].tolist()
        sparse_weights = encoded["lexical_weights"][0]

        sparse_vector = models.SparseVector(
            indices=[
                int(token_id)
                for token_id in sparse_weights.keys()
            ],
            values=[
                float(weight)
                for weight in sparse_weights.values()
            ],
        )

        # Retrieve candidates from both vector indexes and fuse
        # their rankings using Reciprocal Rank Fusion.
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="bm25",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=top_k,
            with_payload=True,
        )

        search_results = response.points

    except Exception as e:
        print(
            f"Knowledge-base search failed: {e}",
            file=sys.stderr,
            flush=True,
        )
        return "The knowledge base could not be searched."

    if not search_results:
        return "No relevant information found in the knowledge base."

    formatted_results = []

    for i, hit in enumerate(search_results, start=1):
        payload = hit.payload or {}

        source = payload.get("source", "Unknown")
        text = payload.get("text", "")
        parent_text = payload.get("parent_text", "")

        citation = f"[{source}]"

        chunk_info = (
            f"--- Document {i} ---\n"
            f"Citation: {citation}\n"
            f"Relevance score: {hit.score:.4f}\n"
            f"Content:\n{text}\n"
        )

        # Neighboring context is available because ingestion stores it.
        if parent_text and parent_text != text:
            chunk_info += (
                f"Additional context:\n{parent_text}\n"
            )

        formatted_results.append(chunk_info)

    system_instruction = (
        "Answer using the retrieved evidence. "
        "Preserve the exact Citation string when citing a source. "
        "Do not invent policies, product details, or citations. "
        "If the retrieved evidence does not answer the question, "
        "state that the available knowledge base does not provide "
        "enough information."
    )

    return (
        "\n\n".join(formatted_results)
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