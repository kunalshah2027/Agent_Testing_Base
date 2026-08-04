import os
from langchain_core.callbacks import BaseCallbackHandler

os.environ["DEEPEVAL_LOCAL_MODE"] = "True"
# ADD THIS LINE INSTEAD
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits import create_sql_agent
# Change your import statement to this:
from deepeval.integrations.langchain import CallbackHandler
# Initialize it like this:
from db import db

# 1. SETUP ENVIRONMENT VARIABLES
# Replace with your actual API keys
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["DEEPEVAL_API_KEY"] = "your-deepeval-api-key"
os.environ["GROQ_API_KEY"] = "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"
os.environ["DEEPEVAL_API_KEY"] = "local_development_smoke_test_key"

# 2. SETUP MOCK SQL DATABASE
# Creating an in-memory SQLite da   tabase representing your payment system

# 3. INITIALIZE LLM & DEEPEVAL CALLBACK TRACER
# DeepEval callback hooks directly into the LangChain execution lifecycle to trace spans
# deepeval_callback = DeepEvalLangchainCallbackHandler()
# llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

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


class LocalQATracer(BaseCallbackHandler):
    def __init__(self):
        self.traces = {"user_input": None, "steps": [], "final_output": None}

    def on_chain_start(self, serialized, prompts, **kwargs):
        # Safely parse prompts even if LangChain wraps them in complex schema objects
        if prompts:
            try:
                # If it's a list, safely extract the string version of the items
                if isinstance(prompts, list):
                    self.traces["user_input"] = str(prompts[0])
                else:
                    self.traces["user_input"] = str(prompts)
            except Exception:
                self.traces["user_input"] = str(prompts)

    def on_tool_start(self, serialized, input_str, **kwargs):
        # Record the database tool name and query input safely
        tool_name = serialized.get("name", "Tool") if isinstance(serialized, dict) else "SQL Database Tool"
        self.traces["steps"].append({"name": tool_name, "input": str(input_str)})

    def on_tool_end(self, output, **kwargs):
        # Capture raw database table rows returned to the agent
        if self.traces["steps"]:
            self.traces["steps"][-1]["output"] = str(output)

    def on_chain_end(self, outputs, **kwargs):
        # Safely handle outputs whether they arrive as dicts, strings, or AgentFinish items
        if outputs:
            try:
                if isinstance(outputs, dict):
                    self.traces["final_output"] = str(outputs.get("output", outputs))
                else:
                    self.traces["final_output"] = str(outputs)
            except Exception:
                self.traces["final_output"] = str(outputs)


local_tracker = LocalQATracer()
# 4. INITIALIZE THE LOCAL LLM
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0  # Set to 0.0 for deterministic, factual SQL queries
)
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
        config={"callbacks": [local_tracker]}  # Passes DeepEval tracer down the execution chain
    )
    raw_output = response["output"]
    import re

    clean_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()

    print("\n--- AGENT OUTPUT RESPONSE ---")
    print(response["output"])
except Exception as e:
    print(f"Agent execution encountered an error: {e}")

# 6. EXTRACT LOCAL MEMORY TRACES
print("\n=== EXTRACTING LOCAL QA TRACES ===")
captured_data = local_tracker.traces
print(f"Total Database Tool Interactions: {len(captured_data['steps'])}")
for idx, step in enumerate(captured_data['steps']):
    print(f"  ├── Step {idx + 1} [Tool: {step['name']}] -> Input: {step['input']}")
