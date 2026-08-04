# qa_framework/telemetry_tracers.py
import json
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler


class FrameworkQATracer(BaseCallbackHandler):
    """Crash-proof local telemetry collector engineered specifically for QA verification pipelines."""

    def __init__(self):
        self.traces = {
            "query_input": None,
            "internal_steps": [],
            "final_agent_output": None,
            "metrics": {
                "start_time": datetime.now(),
                "execution_duration_sec": 0,
                "database_calls_count": 0
            }
        }

    def on_chain_start(self, serialized, prompts, **kwargs):
        if prompts and not self.traces["query_input"]:
            self.traces["query_input"] = str(prompts)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.traces["metrics"]["database_calls_count"] += 1
        tool_identity = serialized.get("name", "Tool") if isinstance(serialized, dict) else "SQL_Tool"
        self.traces["internal_steps"].append({
            "step_index": self.traces["metrics"]["database_calls_count"],
            "tool_utilized": tool_identity,
            "dispatched_input": str(input_str),
            "captured_output": None
        })

    def on_tool_end(self, output, **kwargs):
        if self.traces["internal_steps"]:
            self.traces["internal_steps"][-1]["captured_output"] = str(output)

    def on_chain_end(self, outputs, **kwargs):
        self.traces["metrics"]["execution_duration_sec"] = (
                    datetime.now() - self.traces["metrics"]["start_time"]).total_seconds()
        if outputs and isinstance(outputs, dict):
            self.traces["final_agent_output"] = str(outputs.get("output", outputs))
        else:
            self.traces["final_agent_output"] = str(outputs)
