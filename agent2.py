import os
import re
from datetime import datetime
from typing import Any, Dict, List

# 1. ENVIRONMENT CONFIGURATION
os.environ["DEEPEVAL_LOCAL_MODE"] = "True"
os.environ["DEEPEVAL_API_KEY"] = "local_development_smoke_test_key"

# FIXED: Native modern core imports only (Bypasses broken legacy modules completely)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from deepeval.integrations.langchain import CallbackHandler

# Import your database file object utility
from db import db


# 2. DEFINE NATIVE SQL DATABASE WRAPPER TOOLS
def run_query(sql_query: str) -> str:
    """Executes a SQL query against the payments table and returns the row data string."""
    try:
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        return str(db.run(sql_query))
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"


def get_schema(table_name: str) -> str:
    """Returns the schema design definitions for validation checking."""
    try:
        return db.get_table_info([table_name])
    except Exception as e:
        return f"Error retrieving schema layout: {str(e)}"


db_tools = [
    Tool(
        name="sql_db_query",
        func=run_query,
        description="Input must be a valid SQLite query statement string. Use this to fetch data records."
    ),
    Tool(
        name="sql_db_schema",
        func=get_schema,
        description="Input is a table name string. Use this to check table columns before querying."
    )
]


# 3. CRASH-PROOF LOCAL LOG TRACER
class LocalQATracer(BaseCallbackHandler):
    def __init__(self):
        self.traces = {"user_input": None, "steps": [], "final_output": None}

    def on_chain_start(self, serialized, prompts, **kwargs):
        if prompts and not self.traces["user_input"]:
            self.traces["user_input"] = str(prompts)

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Tool") if isinstance(serialized, dict) else "Database Tool"
        self.traces["steps"].append({"name": tool_name, "input": str(input_str)})

    def on_tool_end(self, output, **kwargs):
        if self.traces["steps"]:
            self.traces["steps"][-1]["output"] = str(output)

    def on_chain_end(self, outputs, **kwargs):
        if outputs and isinstance(outputs, dict):
            self.traces["final_output"] = str(outputs.get("output", outputs))


local_tracker = LocalQATracer()
deepeval_callback = CallbackHandler()

# 4. INITIALIZE THE REASONING MODEL
from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0  # Set to 0.0 for deterministic, factual SQL queries
)


# 5. EXPLICIT REACT INSTRUCTION TEMPLATE WITH COMPACT FALLBACK RUNNER
# We wrap the tools directly into a clean execution loop to bypass AgentExecutor.
template = """You are a Payment Operations QA AI assistant.
Your goal is to inspect database structures and analyze transactions in 'Repair' status.

You have access to the following tool functions:
- sql_db_query: Input must be a valid SQLite query string.
- sql_db_schema: Input must be a table name string.

To get the records, your first step should be running a query like: SELECT * FROM payments WHERE status = 'Repair';

Once you have gathered all data and have your final breakdown ready, use this format:
Final Answer: [Your complete structured analysis of the payment failures and distinct repair recommendations here]

Question: {input}"""

prompt = PromptTemplate.from_template(template)

# Modern chain implementation via LCEL (LangChain Expression Language)
chain = (
        {"input": RunnablePassthrough()}
        | prompt
        | llm
)

# 6. RUN THE PIPELINE VIA CUSTOM AGENT RUNNER LOOP
query = "Fetch all payments with a 'Repair' status, analyze their failure error codes, and give me distinct repair recommendations for each."

print(f"--- Launching Restructured Payment Repair Agent ---")
try:
    # First turn: Ask the model to generate the required database plan or query
    response = chain.invoke(query, config={"callbacks": [local_tracker, deepeval_callback]})
    raw_text = response.content

    # Check if the model requested or inferred an immediate query check
    if "SELECT" in raw_text.upper():
        # Cleanly capture query and pass it straight to our native tool runner
        extracted_sql = re.search(r"SELECT.*?;", raw_text, re.IGNORECASE | re.DOTALL)
        sql_to_run = extracted_sql.group(0) if extracted_sql else "SELECT * FROM payments WHERE status = 'Repair';"

        print(f"Executing extracted tool query: {sql_to_run}")
        # Log tool step manually to keep your QA tracer records functional
        local_tracker.on_tool_start({"name": "sql_db_query"}, sql_to_run)
        db_output = run_query(sql_to_run)
        local_tracker.on_tool_end(db_output)

        # Second turn: Pass results back to model for synthesis and final analysis
        synthesis_prompt = f"""Based on the following database records:
        {db_output}

        Provide a structured analysis on why the error occurred and distinct repair recommendations for the operations team.
        """
        final_response = llm.invoke(synthesis_prompt, config={"callbacks": [local_tracker]})
        final_output = final_response.content
    else:
        final_output = raw_text

    # Clear out internal deepseek thinking chains for a clean response display
    clean_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()

    print("\n--- AGENT OUTPUT RESPONSE ---")
    print(clean_output)

except Exception as e:
    print(f"Agent execution encountered an error: {e}")

# 7. DISPLAY CAPTURED LOCAL MEMORY TRACES
print("\n=== EXTRACTING LOCAL QA TRACES ===")
captured_data = local_tracker.traces
print(f"Total Database Tool Interactions: {len(captured_data['steps'])}")
for idx, step in enumerate(captured_data['steps']):
    print(f"  ├── Step {idx + 1} [Tool: {step['name']}] -> Input: {step['input'].strip()}")
    print(f"  │    └── DB Output: {step.get('output', 'None').strip()[:140]}...")
