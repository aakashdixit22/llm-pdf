"""
Observability: Structured Agent Trace Logger
Captures the full reasoning chain of the agent for each query.
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class TraceStep:
    """A single step in the agent's trace."""
    step_id: str
    step_name: str
    step_type: str  # "reasoning", "tool_call", "retrieval", "calculation", "verification"
    timestamp: str
    input_data: Any
    output_data: Any
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class QueryTrace:
    """Full trace of a single query execution."""
    trace_id: str
    query: str
    start_time: str
    end_time: str = ""
    total_duration_ms: float = 0.0
    steps: list = field(default_factory=list)
    final_answer: str = ""
    citations: list = field(default_factory=list)
    status: str = "in_progress"  # "in_progress", "completed", "error"
    error: str = ""

    def to_dict(self):
        d = asdict(self)
        d["steps"] = [s.to_dict() if hasattr(s, 'to_dict') else s for s in self.steps]
        return d


class AgentTracer:
    """Captures and stores agent execution traces."""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._current_trace: Optional[QueryTrace] = None
        self._step_start_time: float = 0.0
        self.traces: list[QueryTrace] = []

    def start_trace(self, query: str) -> str:
        """Start a new trace for a query. Returns trace_id."""
        trace_id = str(uuid.uuid4())[:8]
        self._current_trace = QueryTrace(
            trace_id=trace_id,
            query=query,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        self._log(f"[TRACE:{trace_id}] Starting trace for query: {query[:100]}...")
        return trace_id

    def log_step(
        self,
        step_name: str,
        step_type: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict = None,
    ):
        """Log a step in the current trace."""
        if not self._current_trace:
            return

        step = TraceStep(
            step_id=f"{self._current_trace.trace_id}_{len(self._current_trace.steps)}",
            step_name=step_name,
            step_type=step_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_data=self._safe_serialize(input_data),
            output_data=self._safe_serialize(output_data),
            metadata=metadata or {},
        )
        self._current_trace.steps.append(step)

        # Console logging
        self._log(
            f"[TRACE:{self._current_trace.trace_id}] "
            f"Step {len(self._current_trace.steps)}: {step_name} ({step_type})"
        )
        if output_data:
            output_str = str(output_data)[:200]
            self._log(f"  Output: {output_str}")

    def end_trace(
        self,
        final_answer: str = "",
        citations: list = None,
        status: str = "completed",
        error: str = "",
    ) -> dict:
        """End the current trace and return the trace dict."""
        if not self._current_trace:
            return {}

        self._current_trace.end_time = datetime.now(timezone.utc).isoformat()
        self._current_trace.final_answer = final_answer
        self._current_trace.citations = citations or []
        self._current_trace.status = status
        self._current_trace.error = error

        # Calculate total duration
        start = datetime.fromisoformat(self._current_trace.start_time)
        end = datetime.fromisoformat(self._current_trace.end_time)
        self._current_trace.total_duration_ms = (end - start).total_seconds() * 1000

        trace_dict = self._current_trace.to_dict()
        self.traces.append(self._current_trace)

        self._log(
            f"[TRACE:{self._current_trace.trace_id}] "
            f"Completed in {self._current_trace.total_duration_ms:.0f}ms "
            f"({len(self._current_trace.steps)} steps) - Status: {status}"
        )

        # Save to file
        self._save_trace(trace_dict)
        self._current_trace = None

        return trace_dict

    def _save_trace(self, trace_dict: dict):
        """Save a trace to JSON file."""
        filename = f"trace_{trace_dict['trace_id']}_{int(time.time())}.json"
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_dict, f, indent=2, ensure_ascii=False, default=str)
        self._log(f"[TRACE] Saved to {filepath}")

    def save_all_traces(self, filename: str = "all_traces.json"):
        """Save all traces to a single file."""
        filepath = os.path.join(self.log_dir, filename)
        all_data = [t.to_dict() for t in self.traces]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)
        self._log(f"[TRACE] Saved {len(all_data)} traces to {filepath}")

    def _safe_serialize(self, data: Any) -> Any:
        """Safely serialize data for JSON storage."""
        if data is None:
            return None
        try:
            json.dumps(data, default=str)
            return data
        except (TypeError, ValueError):
            return str(data)[:2000]

    def _log(self, message: str):
        """Print log message to console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
