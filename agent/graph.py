"""
LangGraph Agent: Autonomous multi-step reasoning agent for the Cyber Ireland report.
Uses Groq (Llama 3.3 70B) as the LLM with tool calling capabilities.
"""

import os
import re
import json
import time
from typing import TypedDict, Annotated, Sequence, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS, init_tools
from utils.logger import AgentTracer


# --- State definition ---
class AgentState(TypedDict):
    """State maintained across the agent graph."""
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    query: str
    trace_id: str
    iteration: int


# --- Configuration ---
MAX_ITERATIONS = 15  # Safety limit to prevent infinite loops


def _normalize_content(content) -> str:
    """Normalize message content - handles both string and list-of-parts formats.
    Gemini 2.5 returns content as list: [{'type': 'text', 'text': '...'}]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                parts.append(part['text'])
            elif isinstance(part, str):
                parts.append(part)
        return '\n'.join(parts)
    return str(content) if content else ""


class CyberIrelandAgent:
    """
    Autonomous LangGraph agent for answering queries about the Cyber Ireland 2022 Report.
    """

    def __init__(
        self,
        vector_store,
        tracer: Optional[AgentTracer] = None,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
    ):
        self.vector_store = vector_store
        self.tracer = tracer or AgentTracer()

        # Initialize tools with dependencies
        init_tools(vector_store, self.tracer)

        # Initialize LLM with tool binding
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
        )

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""

        # --- Node functions ---
        def agent_node(state: AgentState) -> dict:
            """The main agent reasoning node. Calls LLM with tools."""
            messages = state["messages"]
            iteration = state.get("iteration", 0)

            if self.tracer:
                self.tracer.log_step(
                    f"agent_reasoning_iter_{iteration}",
                    "reasoning",
                    input_data={
                        "iteration": iteration,
                        "num_messages": len(messages),
                    },
                )

            # Retry with exponential backoff for rate limiting
            response = None
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.llm_with_tools.invoke(messages)
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        # Extract retry delay if available
                        delay_match = re.search(r'retry.*?(\d+\.?\d*)\s*s', error_str.lower())
                        wait_time = float(delay_match.group(1)) if delay_match else (2 ** attempt * 10)
                        wait_time = min(wait_time + 2, 120)  # Cap at 2 minutes
                        if self.tracer:
                            self.tracer.log_step(
                                f"rate_limit_retry_{attempt}",
                                "reasoning",
                                metadata={"wait_seconds": wait_time, "attempt": attempt+1},
                            )
                        print(f"[AGENT] Rate limited. Waiting {wait_time:.0f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        raise

            if response is None:
                raise Exception("Max retries exceeded due to rate limiting.")

            if self.tracer:
                tool_calls = getattr(response, 'tool_calls', [])
                self.tracer.log_step(
                    f"agent_response_iter_{iteration}",
                    "reasoning",
                    output_data={
                        "has_tool_calls": bool(tool_calls),
                        "num_tool_calls": len(tool_calls) if tool_calls else 0,
                        "tool_names": [tc.get("name", "") for tc in tool_calls] if tool_calls else [],
                        "content_preview": str(response.content)[:300] if response.content else "",
                    },
                )

            return {
                "messages": [response],
                "iteration": iteration + 1,
            }

        def tool_executor(state: AgentState) -> dict:
            """Execute tool calls from the agent."""
            last_message = state["messages"][-1]
            tool_calls = getattr(last_message, 'tool_calls', [])

            if self.tracer:
                self.tracer.log_step(
                    "tool_execution",
                    "tool_call",
                    input_data={
                        "tools": [tc.get("name", "") for tc in tool_calls],
                    },
                )

            # Use ToolNode to execute
            tool_node = ToolNode(ALL_TOOLS)
            result = tool_node.invoke(state)

            if self.tracer:
                tool_messages = result.get("messages", [])
                for tm in tool_messages:
                    if isinstance(tm, ToolMessage):
                        self.tracer.log_step(
                            f"tool_result_{tm.name}",
                            "tool_call",
                            output_data={
                                "tool": tm.name,
                                "result_preview": str(tm.content)[:500],
                            },
                        )

            return result

        # --- Routing logic ---
        def should_continue(state: AgentState) -> str:
            """Determine whether to continue with tools or end."""
            messages = state["messages"]
            iteration = state.get("iteration", 0)

            # Safety check: max iterations
            if iteration >= MAX_ITERATIONS:
                if self.tracer:
                    self.tracer.log_step(
                        "max_iterations_reached",
                        "reasoning",
                        metadata={"max_iterations": MAX_ITERATIONS},
                    )
                return "end"

            # Check if last message has tool calls
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

            return "end"

        # --- Build graph ---
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_executor)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def query(self, user_query: str) -> dict:
        """
        Process a user query through the agent.

        Returns:
            dict with keys: answer, citations, trace_id, steps
        """
        # Start trace
        trace_id = self.tracer.start_trace(user_query)

        try:
            # Prepare initial messages
            system_msg = SystemMessage(content=SYSTEM_PROMPT)
            human_msg = HumanMessage(content=user_query)

            initial_state: AgentState = {
                "messages": [system_msg, human_msg],
                "query": user_query,
                "trace_id": trace_id,
                "iteration": 0,
            }

            # Run the graph
            self.tracer.log_step(
                "graph_execution_start",
                "reasoning",
                input_data={"query": user_query},
            )

            final_state = self.graph.invoke(initial_state)

            # Extract final answer from last AI message
            answer = ""
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    answer = _normalize_content(msg.content)
                    break

            # Extract citations from the answer
            citations = self._extract_citations(answer)

            # Build step summary
            steps = self._summarize_steps(final_state)

            self.tracer.log_step(
                "final_answer",
                "reasoning",
                output_data={
                    "answer_length": len(answer),
                    "num_citations": len(citations),
                    "total_iterations": final_state.get("iteration", 0),
                },
            )

            # End trace
            trace_data = self.tracer.end_trace(
                final_answer=answer,
                citations=citations,
                status="completed",
            )

            return {
                "answer": answer,
                "citations": citations,
                "trace_id": trace_id,
                "steps": steps,
                "trace": trace_data,
            }

        except Exception as e:
            error_msg = f"Agent error: {str(e)}"
            self.tracer.log_step("error", "reasoning", output_data={"error": error_msg})
            trace_data = self.tracer.end_trace(
                status="error",
                error=error_msg,
            )
            return {
                "answer": f"Error processing query: {str(e)}",
                "citations": [],
                "trace_id": trace_id,
                "steps": [],
                "trace": trace_data,
            }

    def _extract_citations(self, text) -> list[dict]:
        """Extract page citations from the answer text."""
        text_str = _normalize_content(text) if not isinstance(text, str) else text
        citations = []
        # Match patterns like "(Page 12)", "(Page 27)", "page 5"
        matches = re.findall(r'[Pp]age\s+(\d+)', text_str)
        seen_pages = set()
        for match in matches:
            page_num = int(match)
            if page_num not in seen_pages:
                seen_pages.add(page_num)
                citations.append({"page": page_num})

        return citations

    def _summarize_steps(self, final_state: dict) -> list[dict]:
        """Create a summary of agent steps from messages."""
        steps = []
        for msg in final_state.get("messages", []):
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, 'tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        steps.append({
                            "type": "tool_call",
                            "tool": tc.get("name", ""),
                            "args": {k: str(v)[:100] for k, v in tc.get("args", {}).items()},
                        })
                elif msg.content:
                    steps.append({
                        "type": "reasoning",
                        "content_preview": _normalize_content(msg.content)[:200],
                    })
            elif isinstance(msg, ToolMessage):
                steps.append({
                    "type": "tool_result",
                    "tool": msg.name,
                    "result_preview": str(msg.content)[:200],
                })
        return steps
