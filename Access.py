"""
Access control for the order tools.

The customer ID never comes from the model. main.py stores the logged-in
customer in `current_customer` for the duration of each request, and the
wrapped tools below read it from there. The model can't pick, change or
see another customer's ID, whatever the user types.
"""

from contextvars import ContextVar

from langchain_core.tools import StructuredTool

current_customer: ContextVar[str | None] = ContextVar("current_customer", default=None)

# Tools that need a logged-in customer.
PROTECTED = {"get_order_status", "create_order", "set_returns"}


def secure_tools(tools):
    """Return the same tools, with the PROTECTED ones wrapped."""

    def wrap(tool):
        if tool.name not in PROTECTED:
            return tool

        schema = dict(tool.args_schema)  # MCP tools carry a JSON-schema dict
        takes_id = "customer_id" in schema.get("properties", {})
        # hide customer_id from the model
        schema["properties"] = {
            k: v for k, v in schema.get("properties", {}).items() if k != "customer_id"
        }
        schema["required"] = [r for r in schema.get("required", []) if r != "customer_id"]

        async def run(**kwargs):
            customer_id = current_customer.get()
            if not customer_id:
                return "The customer is not logged in. Ask them to log in first."
            if takes_id:
                kwargs["customer_id"] = customer_id
            return await tool.ainvoke(kwargs)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=schema,
            coroutine=run,
        )

    return [wrap(t) for t in tools]