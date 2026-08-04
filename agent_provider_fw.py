# agent_provider.py
import os
import re
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from qa_framework.telemetry_tracers import FrameworkQATracer
from db import db  # Assuming your database utility module configuration lives here

os.environ["GROQ_API_KEY"] = "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"


def call_api(prompt: str, options: Dict[str, Any] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    config = options.get("config", {}) if options else {}
    target_model = config.get("model_name", "llama-3.3-70b-versatile")
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
    try:
        llm = ChatGroq(model=target_model, temperature=0.0)
        local_tracer = FrameworkQATracer()

        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=False,
            suffix=agent_suffix
        )

        response = agent_executor.invoke({"input": prompt}, config={"callbacks": [local_tracer]})
        clean_text = re.sub(r'<think>.*?</think>', '', response.get("output", ""), flags=re.DOTALL).strip()

        return {"output": clean_text}
    except Exception as e:
        return {"error": f"Framework run pipeline crash: {str(e)}"}
