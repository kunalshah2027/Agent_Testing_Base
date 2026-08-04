import os
import re
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from deepeval.integrations.langchain import CallbackHandler
from langchain_core.callbacks import BaseCallbackHandler

# 1. ENFORCE DEEPEVAL OFFLINE ENVIRONMENT SECURITY PLACEHOLDERS
os.environ["DEEPEVAL_LOCAL_MODE"] = "True"
os.environ["DEEPEVAL_API_KEY"] = "local_development_smoke_test_key"
os.environ["GROQ_API_KEY"] = "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"

# Import your shared enterprise database wrapper directly from your workspace
from db import db

# 2. SHARED REPAIR AGENT SYSTEM INSTRUCTIONS
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


# 3. EXPOSE THE PROMPTFOO DYNAMIC ENTRYPOINT HOOK
def call_api(prompt: str, options: Dict[str, Any] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Promptfoo invokes this function automatically for every iteration matrix block scenario.
    """
    # Extract structural configuration parameters safely passed down from your YAML definitions
    config = options.get("config", {}) if options else {}
    model_name = config.get("model_name", "llama-3.3-70b-versatile")

    try:
        # Initialize the specific model being evaluated on this loop turn
        llm = ChatGroq(
            model=model_name,
            temperature=0.0  # Strict determinism for functional QA test consistency
        )

        # Instantiate localized evaluation tracers
        #deepeval_callback = CallbackHandler()

        # Build the exact agent architecture blueprint
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=False,  # Set to False to keep Promptfoo terminal reporting grids clean
            suffix=agent_suffix
        )

        # Dispatch user query down the agent execution chain safely
        # Promptfoo sends the full formatted string, we isolate the core text payload
        response = agent_executor.invoke(
            {"input": prompt},
            config={"callbacks": [local_tracker]}
        )

        raw_output = response.get("output", str(response))
        # Safely clean out internal local model chain-of-thought blocks if present
        clean_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()

        # Return structured tracking output blocks straight back to Promptfoo core matrices
        return {
            "output": clean_output
        }

    except Exception as e:
        return {
            "error": f"Agent execution runtime pipeline crashed: {str(e)}"
        }
