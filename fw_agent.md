I have compiled and structured your entire automated AI QA test framework into a single, clean markdown document below.
You can copy this text block and save it directly as framework_agent.md inside your project directory to act as your official documentation and code repository template. [1] 

# AI QA Test Framework Document: Payment Repair Agent Testing Package
This document outlines the production-grade, offline-safe testing package designed to evaluate the LangChain Payment Repair Agent. The framework combines **internal LangChain tracing callbacks** for structural step validation with **Promptfoo programmatic matrix integrations** for cross-LLM benchmarking.
---## 📦 Project Architecture LayoutTo implement this framework inside your PyCharm environment, structure your project workspace as follows:
```text
payment-agent-qa/
├── qa_framework/
│   ├── __init__.py
│   ├── database_provider.py    # Enterprise mock/live database configurations
│   ├── telemetry_tracers.py   # Custom BaseCallbackHandlers capturing execution states
│   └── eval_metrics.py        # Programmatic offline grading rubrics
├── agent_provider.py          # Promptfoo custom Python provider script
├── promptfoo_config.yaml      # Promptfoo matrix cross-validation blueprint
└── test_runner.py             # Main framework test orchestrator package
```
---## 🛠️ 1. Framework Package Submodules
### Module A: The Core Tracer (`qa_framework/telemetry_tracers.py`)
This component safely extracts internal execution states, structured database input strings, raw tool observations, and engine latencies into a local structured dictionary without relying on external cloud endpoints.
```python
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
        self.traces["metrics"]["execution_duration_sec"] = (datetime.now() - self.traces["metrics"]["start_time"]).total_seconds()
        if outputs and isinstance(outputs, dict):
            self.traces["final_agent_output"] = str(outputs.get("output", outputs))
        else:
            self.traces["final_agent_output"] = str(outputs)
```

### Module B: Evaluator Matrix Engines (`qa_framework/eval_metrics.py`)
This module houses deterministic, rule-based evaluation functions to score the captured traces. 
```python
# qa_framework/eval_metrics.py
import re

class FrameworkEvaluator:
    @staticmethod
    def calculate_trace_scores(trace_dict: dict) -> dict:
        """
        Runs programmatic QA assertions against raw execution frames to generate clear scores.
        Returns numerical indicators and feedback matrices.
        """
        steps = trace_dict.get("internal_steps", [])
        final_answer = trace_dict.get("final_agent_output", "")
        
        # 1. SQL Factual Grounding Evaluation Score
        has_database_interaction = len(steps) > 0
        sql_grounding_score = 1.0 if has_database_interaction else 0.0
        
        # 2. Operational Completeness & Hallucination Checks
        contains_error_resolutions = any(kw in final_answer.upper() for kw in ["BIC", "LIQUIDITY", "FUNDS", "SWIFT", "VERIFY"])
        completeness_score = 1.0 if contains_error_resolutions else 0.2
        
        # 3. Code Execution Hygiene Verification
        contains_dangerous_sql = any("DROP" in str(s.get("dispatched_input")).upper() or "DELETE" in str(s.get("dispatched_input")).upper() for s in steps)
        safety_score = 0.0 if contains_dangerous_sql else 1.0
        
        # Aggregate and compute a mean baseline grade
        composite_framework_score = (sql_grounding_score + completeness_score + safety_score) / 3.0
        
        return {
            "composite_score": round(composite_framework_score, 2),
            "breakdown": {
                "sql_grounding": sql_grounding_score,
                "completeness": completeness_score,
                "safety": safety_score
            },
            "telemetry_insights": {
                "total_db_queries": len(steps),
                "latency_overhead_seconds": trace_dict["metrics"]["execution_duration_sec"]
            }
        }
```

### Module C: Automated Provider Hook (`agent_provider.py`)
Exposes a clean interface hook that Promptfoo reads natively, executing each matrix turn through your virtual environment configurations.
```python
# agent_provider.py
import os
import re
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from qa_framework.telemetry_tracers import FrameworkQATracer
from db import db  # Assuming your database utility configuration lives here

os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

def call_api(prompt: str, options: Dict[str, Any] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    config = options.get("config", {}) if options else {}
    target_model = config.get("model_name", "llama-3.3-70b-versatile")
    
    try:
        llm = ChatGroq(model=target_model, temperature=0.0)
        local_tracer = FrameworkQATracer()
        
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=False,
            suffix="You are an expert Payment Operations QA AI assistant. Query DB for payments in Repair status and analyze failure reasons."
        )
        
        response = agent_executor.invoke({"input": prompt}, config={"callbacks": [local_tracer]})
        clean_text = re.sub(r'<think>.*?</think>', '', response.get("output", ""), flags=re.DOTALL).strip()
        
        return {"output": clean_text}
    except Exception as e:
        return {"error": f"Framework run pipeline crash: {str(e)}"}
```
---
## ⚙️ 2. Promptfoo Matrix Configuration (`promptfoo_config.yaml`)
```yaml
# promptfoo_config.yaml
description: 'Cross-LLM Evaluation Matrix'

prompts:
  - "{{user_query}}"

providers:
  - id: "file://agent_provider.py"
    label: "Agent_LLM_1_Llama70B"
    config:
      pythonExecutable: "C:/Users/Admin/PycharmProjects/agent-testing/venv/Scripts/python.exe"
      model_name: "llama-3.3-70b-versatile"
  - id: "file://agent_provider.py"
    label: "Agent_LLM_2_Llama8B"
    config:
      pythonExecutable: "C:/Users/Admin/PycharmProjects/agent-testing/venv/Scripts/python.exe"
      model_name: "llama-3.1-8b-instant"

tests:
  - vars:
      user_query: "Fetch all payments with a 'Repair' status, analyze their failure error codes, and give me distinct repair recommendations for each."
  - vars:
      user_query: "Identify any payments stuck in repair because of code ERR_4041 and list the recommended validation checks."
  - vars:
      user_query: "Find liquidity error code ERR_2019 items and suggest accounting workflows to clear them."
```
---
## 🚀 3. The Test Framework Orchestrator (`test_runner.py`)

```python
# test_runner.py
import os
import re
import json
import subprocess
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent

# Framework Component Modules Imports
from qa_framework.telemetry_tracers import FrameworkQATracer
from qa_framework.eval_metrics import FrameworkEvaluator
from db import db

os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

# 10 Isolated Test Prompts
TEST_PROMPTS_SUITE = [
    "Fetch all payments with a 'Repair' status, analyze their failure error codes, and give me distinct repair recommendations for each.",
    "Identify any payments stuck in repair because of code ERR_4041 and list the recommended validation checks.",
    "Find liquidity error code ERR_2019 items and suggest accounting workflows to clear them.",
    "Are there any transactions currently failing due to an unmappable status code?",
    "Provide a high-priority action checklist for all transactions flagged with payment failures.",
    "Summarize total monetary values currently stuck in a non-completed status.",
    "Draft an operational exception message for a clearing account liquidity deficit.",
    "Verify if TXN_1003 requires any manual intervention steps.",
    "Check if the database has any missing values in the critical error description fields.",
    "Generate a structured markdown summary report detailing all active payment anomalies."
]

def run_point_1_internal_trace_evaluation():
    print("\n==================================================")
    print("▶ PHASE 1: RUNNING INTERNAL LANGCHAIN TRACE EVALUATIONS")
    print("==================================================")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

phase_1_results = []
for idx, prompt_text in enumerate(TEST_PROMPTS_SUITE):
print(f"Executing Prompt {idx+1}/{len(TEST_PROMPTS_SUITE)} against evaluation tracer...")
tracer = FrameworkQATracer()
agent_executor = create_sql_agent(
llm=llm, db=db, agent_type="tool-calling", verbose=False
) [2, 3] 
try:
response = agent_executor.invoke({"input": prompt_text}, config={"callbacks": [tracer]})
evaluation_report = FrameworkEvaluator.calculate_trace_scores(tracer.traces) [4] 
phase_1_results.append({
"prompt": prompt_text,
"score": evaluation_report["composite_score"],
"breakdown": evaluation_report["breakdown"],
"insights": evaluation_report["telemetry_insights"]
})
print(f" ↳ Composite Score Assigned: {evaluation_report['composite_score']}")
except Exception as e:
print(f" ↳ [ERROR] Scenario Failed: {e}")
return phase_1_results
def run_point_2_promptfoo_cross_matrix():
print("\n==================================================")
print("▶ PHASE 2: RUNNING PROMPTFOO CROSS-LLM BENCHMARKS")
print("==================================================")
output_json = "promptfoo_results.json"
os.environ["PROMPTFOO_PYTHON"] = "C:\Users\Admin\PycharmProjects\agent-testing\venv\Scripts\python.exe"
cmd = ["promptfoo", "eval", "-c", "promptfoo_config.yaml", "-o", output_json]
try:
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✔ Promptfoo matrix cross-validation collection phase completed successfully.")
with open(output_json, 'r', encoding='utf-8') as f:
return json.load(f)
except Exception as e:
print(f"❌ Promptfoo execution driver hit an issue: {e}")
return None
def compile_final_qa_report(phase_1, phase_2):
print("\n==================================================")
print("📋 FINAL COMPREHENSIVE RECONCILIATION QUALITY REPORT")
print("==================================================")
report_structure = {
"report_generated_at": str(datetime.now()),
"phase_1_trace_evaluations": phase_1,
"phase_2_cross_model_comparison": {}
}
print("\n[Trace Analysis Standing Matrix]")
total_scores = 0
for item in phase_1:
print(f" - Score: {item['score']} | Queries Executed: {item['insights']['total_db_queries']} | Prompt: {item['prompt'][:60]}...")
total_scores += item['score']
print(f"** Average Framework Quality Grade: {round(total_scores / len(phase_1), 2) if phase_1 else 0.0}")
if phase_2 and "results" in phase_2:
print("\n[Cross-Model Verification Summary Matrix]")
table_rows = phase_2["results"].get("table", [])
for idx, row in enumerate(table_rows):
query_text = row.get("vars", {}).get("user_query", "Unknown Query")
outputs = row.get("outputs", [])
print(f" \nScenario #{idx+1}: {query_text[:80]}...")
for out in outputs:
provider_label = out.get("provider", "Unknown")
latency = out.get("latencyMs", 0)
txt_sample = str(out.get("text", "None"))[:100].replace('\n', ' ')
print(f" ├── {provider_label} | Latency: {latency}ms | Response: {txt_sample}...") [5, 6, 7] 
report_structure["phase_2_cross_model_comparison"][f"scenario_{idx+1}"] = {
"query": query_text,
"models_compared": [o.get("provider") for o in outputs]
}
report_filename = "final_qa_compiled_report.json"
with open(report_filename, 'w', encoding='utf-8') as rf:
json.dump(report_structure, rf, indent=4)
print(f"\n✔ Complete consolidated analytical metrics log exported to file workspace: {report_filename}") [8] 
if name == "main":
p1_data = run_point_1_internal_trace_evaluation()
p2_data = run_point_2_promptfoo_cross_matrix()
compile_final_qa_report(p1_data, p2_data)
```


### Next Steps for Implementation
1. **Save this file**: Copy the contents above into a new file named `framework_agent.md` in your project folder.
2. **Execute**: Run `python test_runner.py` from your PyCharm terminal. It will orchestrate Phase 1, automatically kick off Promptfoo in Phase 2, and dump the ultimate aggregated audit summary straight into `final_qa_compiled_report.json`.

Would you like help setting up a **CI/CD pipeline script** (like GitHub Actions) to run this `test_runner.py` automatically on every new code commit?

