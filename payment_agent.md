================================================================================
AI QA AGENT DEVELOPMENT CHAT HISTORY - PAYMENT REPAIR ARCHITECTURE
================================================================================

--------------------------------------------------------------------------------
1. CORE ARCHITECTURE DEFINITION (LANGCHAIN + DEEPEVAL PRODUCTION SETUP)
--------------------------------------------------------------------------------
The initial production-grade framework connects a Python LangChain SQL agent 
to an enterprise payment schema using DeepEval for cloud-based telemetry tracing.

Code Implementation:
```python
import os
from datetime import datetime
from typing import Any, Dict, List
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, insert
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from deepeval.callbacks import DeepEvalLangchainCallbackHandler

# Setup Environment Variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["DEEPEVAL_API_KEY"] = "your-deepeval-api-key"

# In-Memory SQLite Payment Database Configuration
engine = create_engine("sqlite:///:memory:")
metadata = MetaData()

payments_table = Table(
    "payments",
    metadata,
    Column("payment_id", String(50), primary_key=True),
    Column("amount", Integer),
    Column("currency", String(3)),
    Column("status", String(20)),            # e.g., 'Completed', 'Repair'
    Column("error_code", String(20)),        # e.g., 'ERR_001', 'ERR_002'
    Column("error_description", Text)
)
metadata.create_all(engine)

# Insert Mock QA Data Matrix
with engine.connect() as connection:
    connection.execute(
        insert(payments_table),
        [
            {"payment_id": "TXN_1001", "amount": 5000, "currency": "USD", "status": "Repair", "error_code": "ERR_4041", "error_description": "Beneficiary BIC bank code not found in SWIFT directory."},
            {"payment_id": "TXN_1002", "amount": 12500, "currency": "EUR", "status": "Repair", "error_code": "ERR_2019", "error_description": "Insufficient funds in client clearing account for structural settlement."},
            {"payment_id": "TXN_1003", "amount": 750, "currency": "USD", "status": "Completed", "error_code": None, "error_description": None}
        ]
    )
    connection.commit()

db = SQLDatabase(engine)

# Core LLM & Tracer Hooks
deepeval_callback = DeepEvalLangchainCallbackHandler()
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

agent_suffix = """
You are an expert Payment Operations QA AI assistant. 
When asked about payments in 'Repair' status:
1. Query the database to find all records where status is 'Repair'.
2. Extract the error_code and error_description.
3. Perform an analysis on why the error occurred.
4. Provide structured, actionable repair recommendations for the operations team.
"""

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
    suffix=agent_suffix
)

# Run Query
query = "Fetch all payments with a 'Repair' status, analyze their failure error codes, and give me distinct repair recommendations for each."
response = agent_executor.invoke({"input": query}, config={"callbacks": [deepeval_callback]})
deepeval_callback.flush()
```

--------------------------------------------------------------------------------
2. DATABASE PIPELINE VERIFICATION (LOCAL SMOKE TESTING)
--------------------------------------------------------------------------------
To test that your database connection is active before running expensive or complex model generations, run these verification steps at the bottom of the script:

```python
print("\n--- 1. Testing LangChain SQLDatabase Wrapper ---")
print("Discovered Tables:", db.get_usable_table_names())
print("\nTable Schema Layout:\n", db.get_table_info(["payments"]))

print("\nQuerying Data via LangChain:")
test_query_result = db.run("SELECT * FROM payments WHERE status = 'Repair';")
print(test_query_result)
```

--------------------------------------------------------------------------------
3. DEEPEVAL PACKAGE DEPRECATION FIXES (OFFLINE MODE RESOLUTION)
--------------------------------------------------------------------------------
Issue Encountered:
`ImportError: cannot import name 'DeepEvalLangchainCallbackHandler' from 'deepeval.callbacks'`

Resolution:
DeepEval's tracking namespace was updated. To execute entirely offline without a cloud API key dashboard, use the modern integration path and set `DEEPEVAL_LOCAL_MODE`.

```python
import os
# Force DeepEval to completely skip cloud validation checks
os.environ["DEEPEVAL_LOCAL_MODE"] = "True"
os.environ["DEEPEVAL_API_KEY"] = "local_development_smoke_test_key" # Suppresses token skipping warnings

# Correct Modern Import Path
from deepeval.integrations.langchain import CallbackHandler
deepeval_callback = CallbackHandler()
```

--------------------------------------------------------------------------------
4. LANGCHAIN PACKAGE DEPRECATION & AGENT EXECUTION ERRORS
--------------------------------------------------------------------------------
Issue Encountered:
`ImportError: cannot import name 'ChatOllama' from 'langchain_community.chat_models'`
`ModuleNotFoundError: No module named 'langchain_core.memory'`

Root Cause Analysis:
Older versions of LangChain community classes lacked support for structured tool calling or reasoning output formats from open-source models like `deepseek-r1:8b`. Importing any object from the deprecated `langchain.agents` pathway triggers missing dependency crashes in modernized virtual environments.

Resolution (The Native LCEL Framework Approach):
Bypass the legacy `create_sql_agent` and `AgentType` modules completely. Instead, use LangChain Expression Language (LCEL) paired with a manual loop selector to extract queries and pass data directly to your target database driver functions.

```python
import os
import re
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Initialize Modern Local LLM
llm = ChatOllama(model="deepseek-r1:8b", temperature=0.0)

# Functional Tool Extraction Wrapper Definitions
def run_query(sql_query: str) -> str:
    try:
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        return str(db.run(sql_query))
    except Exception as e:
        return f"Error: {str(e)}"

# Crash-Proof Custom Local Memory Tracer
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

# Construct Version-Safe ReAct Evaluation Prompt Template
template = """You are a Payment Operations QA AI assistant.
Your goal is to inspect database structures and analyze transactions in 'Repair' status.

To get the records, your first step should be running a query like: SELECT * FROM payments WHERE status = 'Repair';
Once you have gathered all data and have your final breakdown ready, use this format:
Final Answer: [Your complete structured analysis of the payment failures and distinct repair recommendations here]

Question: {input}"""

prompt = PromptTemplate.from_template(template)
chain = {"input": RunnablePassthrough()} | prompt | llm

# Execute LCEL Flow Turn with Manual Tool Orchestration
query = "Fetch all payments with a 'Repair' status, analyze their failure error codes, and give me distinct repair recommendations for each."
response = chain.invoke(query, config={"callbacks": [local_tracker]})
raw_text = response.content

if "SELECT" in raw_text.upper():
    extracted_sql = re.search(r"SELECT.*?;", raw_text, re.IGNORECASE | re.DOTALL)
    sql_to_run = extracted_sql.group(0) if extracted_sql else "SELECT * FROM payments WHERE status = 'Repair';"
    
    local_tracker.on_tool_start({"name": "sql_db_query"}, sql_to_run)
    db_output = run_query(sql_to_run)
    local_tracker.on_tool_end(db_output)
    
    synthesis_prompt = f"Based on the following data: {db_output}\nProvide distinct payment repair recommendations."
    final_response = llm.invoke(synthesis_prompt, config={"callbacks": [local_tracker]})
    final_output = final_response.content
else:
    final_output = raw_text

# Strip out DeepSeek internal raw <think> tags for clean logging output display
clean_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
print("Clean Output:\n", clean_output)
```

--------------------------------------------------------------------------------
5. ENTERPRISE ARCHITECTURE: CHOOSE THE RIGHT MODEL FROM SYSTEM KEYS
--------------------------------------------------------------------------------
Model Evaluated for Agent Optimization:
- **`llama-3.3-70b-versatile`** -> **WINNER (Primary Production Pick)**
