"""
Agent Tools: Tool definitions for the LangGraph agent.
Includes retriever, table extractor, calculator, and citation formatter.
"""

import json
import math
import re
from typing import Optional, Annotated

from langchain_core.tools import tool
from simpleeval import simple_eval

# These will be set during initialization
_vector_store = None
_tracer = None


def init_tools(vector_store, tracer):
    """Initialize tools with the vector store and tracer instances."""
    global _vector_store, _tracer
    _vector_store = vector_store
    _tracer = tracer


@tool
def search_documents(query: str, top_k: int = 8) -> str:
    """
    Search the Cyber Ireland 2022 Report for relevant information.
    Use this tool to find facts, statistics, tables, and data from the document.
    Returns ranked results with page numbers and content snippets.

    Args:
        query: The search query - be specific about what data you need.
        top_k: Number of results to return (default: 8).
    """
    if not _vector_store:
        return "Error: Vector store not initialized."

    if _tracer:
        _tracer.log_step(
            "search_documents",
            "retrieval",
            input_data={"query": query, "top_k": top_k},
        )

    results = _vector_store.hybrid_search(query, top_k=top_k)

    if not results:
        return "No relevant documents found for this query."

    formatted = []
    for i, r in enumerate(results):
        page = r["metadata"]["page_number"]
        chunk_type = r["metadata"]["chunk_type"]
        section = r["metadata"].get("section", "")
        score = r.get("score", 0)
        content = r["content"]

        formatted.append(
            f"[Result {i+1}] (Page {page}, Type: {chunk_type}, "
            f"Section: {section}, Score: {score:.3f})\n{content}"
        )

    output = "\n\n---\n\n".join(formatted)

    if _tracer:
        _tracer.log_step(
            "search_documents_result",
            "retrieval",
            output_data={
                "num_results": len(results),
                "top_pages": [r["metadata"]["page_number"] for r in results[:5]],
                "top_scores": [r.get("score", 0) for r in results[:5]],
            },
        )

    return output


@tool
def search_tables(query: str, top_k: int = 5) -> str:
    """
    Search specifically for tables in the Cyber Ireland 2022 Report.
    Use this when you need specific data from tables, regional breakdowns,
    or numerical comparisons.

    Args:
        query: Describe the table data you're looking for.
        top_k: Number of table results to return.
    """
    if not _vector_store:
        return "Error: Vector store not initialized."

    if _tracer:
        _tracer.log_step(
            "search_tables",
            "retrieval",
            input_data={"query": query, "top_k": top_k},
        )

    results = _vector_store.hybrid_search(query, top_k=top_k, filter_type="table")

    # Also search table summaries
    summary_results = _vector_store.hybrid_search(
        query, top_k=top_k, filter_type="table_summary"
    )

    all_results = results + summary_results
    # Deduplicate by table_id
    seen = set()
    unique = []
    for r in all_results:
        tid = r["metadata"].get("table_id", r["id"])
        if tid not in seen:
            seen.add(tid)
            unique.append(r)

    if not unique:
        return "No relevant tables found. Try using search_documents instead."

    formatted = []
    for i, r in enumerate(unique[:top_k]):
        page = r["metadata"]["page_number"]
        caption = r["metadata"].get("caption", "")
        content = r["content"]

        formatted.append(
            f"[Table Result {i+1}] (Page {page}, Caption: {caption})\n{content}"
        )

    output = "\n\n---\n\n".join(formatted)

    if _tracer:
        _tracer.log_step(
            "search_tables_result",
            "retrieval",
            output_data={"num_results": len(unique[:top_k])},
        )

    return output


@tool
def get_page_content(page_number: int) -> str:
    """
    Get ALL content from a specific page of the report.
    Use this when you know the exact page number and need the full context.

    Args:
        page_number: The page number to retrieve (1-indexed).
    """
    if not _vector_store:
        return "Error: Vector store not initialized."

    if _tracer:
        _tracer.log_step(
            "get_page_content",
            "retrieval",
            input_data={"page_number": page_number},
        )

    results = _vector_store.get_page_content(page_number)

    if not results:
        return f"No content found for page {page_number}."

    formatted = []
    for r in results:
        formatted.append(
            f"[Page {page_number}, Type: {r['metadata']['chunk_type']}]\n{r['content']}"
        )

    return "\n\n---\n\n".join(formatted)


@tool
def calculate(expression: str) -> str:
    """
    Perform a mathematical calculation. Use this for any arithmetic operations,
    CAGR calculations, percentages, comparisons, etc.

    Supports: +, -, *, /, **, (), sqrt(), round(), abs()

    Examples:
    - "7351 * 1.10 ** 8" for compound growth
    - "(17333 / 7351) ** (1/8) - 1" for CAGR calculation
    - "37 / 734 * 100" for percentage calculation
    - "round((17333/7351)**(1/8) - 1, 4)" to round a result

    Args:
        expression: Mathematical expression to evaluate.
    """
    if _tracer:
        _tracer.log_step(
            "calculate",
            "calculation",
            input_data={"expression": expression},
        )

    try:
        # Add safe math functions
        functions = {
            "sqrt": math.sqrt,
            "round": round,
            "abs": abs,
            "pow": pow,
            "log": math.log,
            "log10": math.log10,
            "ceil": math.ceil,
            "floor": math.floor,
        }

        result = simple_eval(expression, functions=functions)

        # Format output
        if isinstance(result, float):
            # Check if it looks like a percentage (small decimal)
            if -1 < result < 1 and result != 0:
                percentage = result * 100
                output = f"Result: {result}\nAs percentage: {percentage:.2f}%\nExpression: {expression}"
            else:
                output = f"Result: {result}\nFormatted: {result:,.2f}\nExpression: {expression}"
        else:
            output = f"Result: {result}\nExpression: {expression}"

        if _tracer:
            _tracer.log_step(
                "calculate_result",
                "calculation",
                output_data={"expression": expression, "result": str(result)},
            )

        return output

    except Exception as e:
        error_msg = f"Calculation error: {str(e)}\nExpression: {expression}\nPlease check the expression syntax."
        if _tracer:
            _tracer.log_step(
                "calculate_error",
                "calculation",
                output_data={"error": str(e)},
            )
        return error_msg


@tool
def calculate_cagr(
    beginning_value: float,
    ending_value: float,
    num_years: float,
) -> str:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    CAGR = (Ending_Value / Beginning_Value) ^ (1 / Num_Years) - 1

    Args:
        beginning_value: The starting/baseline value (e.g., 7351 jobs in 2022).
        ending_value: The target/ending value (e.g., 17333 jobs in 2030).
        num_years: Number of years between start and end (e.g., 8).
    """
    if _tracer:
        _tracer.log_step(
            "calculate_cagr",
            "calculation",
            input_data={
                "beginning_value": beginning_value,
                "ending_value": ending_value,
                "num_years": num_years,
            },
        )

    try:
        if beginning_value <= 0:
            return "Error: Beginning value must be positive."
        if ending_value <= 0:
            return "Error: Ending value must be positive."
        if num_years <= 0:
            return "Error: Number of years must be positive."

        cagr = (ending_value / beginning_value) ** (1 / num_years) - 1
        cagr_pct = cagr * 100

        output = (
            f"CAGR Calculation:\n"
            f"  Formula: (Ending / Beginning) ^ (1 / Years) - 1\n"
            f"  = ({ending_value:,.0f} / {beginning_value:,.0f}) ^ (1 / {num_years:.0f}) - 1\n"
            f"  = {ending_value/beginning_value:.6f} ^ {1/num_years:.6f} - 1\n"
            f"  = {cagr:.6f}\n"
            f"  = {cagr_pct:.2f}%\n\n"
            f"CAGR: {cagr_pct:.2f}% per year\n"
            f"This means a ~{cagr_pct:.1f}% compound annual growth rate is required."
        )

        if _tracer:
            _tracer.log_step(
                "calculate_cagr_result",
                "calculation",
                output_data={
                    "cagr_decimal": round(cagr, 6),
                    "cagr_percentage": round(cagr_pct, 2),
                },
            )

        return output

    except Exception as e:
        return f"CAGR calculation error: {str(e)}"


# List of all tools for the agent
ALL_TOOLS = [
    search_documents,
    search_tables,
    get_page_content,
    calculate,
    calculate_cagr,
]
