import os
os.environ["DEEPEVAL_LOCAL_MODE"] = "True"
from datetime import datetime
from typing import Any, Dict, List

# ADD THIS LINE INSTEAD
from langchain_ollama import ChatOllama
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, insert
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
# Change your import statement to this:
from deepeval.integrations.langchain import CallbackHandler


# Initialize it like this:

from db import db

# 1. SETUP ENVIRONMENT VARIABLES
# Replace with your actual API keys
#os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
#os.environ["DEEPEVAL_API_KEY"] = "your-deepeval-api-key"


# 2. SETUP MOCK SQL DATABASE
# Creating an in-memory SQLite database representing your payment system

# 3. INITIALIZE LLM & DEEPEVAL CALLBACK TRACER
# DeepEval callback hooks directly into the LangChain execution lifecycle to trace spans
#deepeval_callback = DeepEvalLangchainCallbackHandler()
deepeval_callback = CallbackHandler()
llm = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0.0
)
#llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# 4. CONSTRUCT CUSTOM SYSTEM INSTRUCTIONS FOR REPAIR AGENT
# Adding domain-specific guidelines enforces reliable analysis and recommendation mapping
agent_suffix = """
You are an expert Payment Operations QA AI assistant. 
When asked about payments in 'Repair' status:
1. Query the database to find all records where status is 'Repair'.
2. Extract the error_code and error_description.
3. Perform an analysis on why the error occurred.
4. Provide structured, actionable repair recommendations for the operations team.

Example Matrix for Recommendations:
- ERR_4041 (SWIFT/BIC issues): Recommend verifying the routing code via SWIFT Ref, or contact the beneficiary for correct instructions.
- ERR_2019 (Funding issues): Recommend triggering a liquidity alert or queuing the transaction until the clearing account is topped up.
"""

# 5. INITIALIZE THE LANGCHAIN SQL AGENT
# create_sql_agent provides out-of-the-box tools to inspect schemas and run validated SQL queries
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",
    verbose=True,
    suffix=agent_suffix
)

# 6. RUN THE AGENT WITH THE TRACER CALLBACK
query = "Fetch all payments with a 'Repair' status, analyze their failure error codes, and give me distinct repair recommendations for each."

print(f"--- Launching Payment Repair Agent Execution ---")
try:
    response = agent_executor.invoke(
        {"input": query},
        config={"callbacks": [deepeval_callback]}  # Passes DeepEval tracer down the execution chain
    )

    print("\n--- AGENT OUTPUT RESPONSE ---")
    print(response["output"])
except Exception as e:
    print(f"Agent execution encountered an error: {e}")

finally:
    # Clean, local-friendly termination
    print("\n--- Local agent run complete. Traces retained in memory for local QA analysis. ---")
    # 8. EXTRACT AND DISPLAY DEEPEVAL TRACES FROM MEMORY
    print("\n=== EXTRACTING LOCAL DEEPEVAL TRACES ===")

    # DeepEval stores traces in a private attribute list named '_traces'
    captured_traces = deepeval_callback._traces

    print(f"Total root level traces captured: {len(captured_traces)}")

    for index, trace in enumerate(captured_traces):
        print(f"\n[Trace #{index + 1}] Type: {trace.type} | Name: {trace.name}")
        print(f"Execution Duration: {trace.execution_time}s")
        print(f"Query Input: {trace.input}")
        print(f"Agent Output: {trace.output}")

        # Iterate through child spans (e.g., individual tool calls, SQL database lookups, LLM generations)
        if hasattr(trace, 'traces') and trace.traces:
            print(f"  └── Internal Steps Taken ({len(trace.traces)} steps):")
            for sub_step in trace.traces:
                print(f"      ├── Step Type: {sub_step.type} | Name: {sub_step.name}")
                print(f"      │   ├── Duration: {sub_step.execution_time}s")
                # Truncating inputs/outputs to keep the console clean
                print(f"      │   ├── Step Input: {str(sub_step.input)[:120]}...")
                print(f"      │   └── Step Output: {str(sub_step.output)[:120]}...")

#finally:
    # 7. FLUSH TRACES TO DEEPEVAL PLATFORM
    # This sends the complete LLM token usage, prompt inputs, tool calls, and latency data to your DeepEval dashboard
 #   deepeval_callback.flush()
  #  print("\n--- DeepEval Traces successfully synchronized to cloud dashboard. ---")
