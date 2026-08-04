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

os.environ["GROQ_API_KEY"] = "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"

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

    # Initialize the baseline test validation engine configuration
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    phase_1_results = []

    for idx, prompt_text in enumerate(TEST_PROMPTS_SUITE):
        print(f"Executing Prompt {idx + 1}/{len(TEST_PROMPTS_SUITE)} against evaluation tracer...")
        tracer = FrameworkQATracer()

        agent_executor = create_sql_agent(
            llm=llm, db=db, agent_type="tool-calling", verbose=False
        )

        try:
            response = agent_executor.invoke({"input": prompt_text}, config={"callbacks": [tracer]})

            # Pass memory trace tree straight to the programmatic metrics grader module
            evaluation_report = FrameworkEvaluator.calculate_trace_scores(tracer.traces)

            phase_1_results.append({
                "prompt": prompt_text,
                "score": evaluation_report["composite_score"],
                "breakdown": evaluation_report["breakdown"],
                "insights": evaluation_report["telemetry_insights"]
            })
            print(f"   ↳ Composite Score Assigned: {evaluation_report['composite_score']}")
        except Exception as e:
            print(f"   ↳ [ERROR] Scenario Failed: {e}")

    return phase_1_results


def run_point_2_promptfoo_cross_matrix():
    print("\n==================================================")
    print("▶ PHASE 2: RUNNING PROMPTFOO CROSS-LLM BENCHMARKS")
    print("==================================================")

    output_json = "promptfoo_results.json"
    # Target execution parameters path mapping configuration
    os.environ["PROMPTFOO_PYTHON"] = "C:\\Users\\Admin\\PycharmProjects\\agent-testing\\venv\\Scripts\\python.exe"

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

    # Print Trace Metric Analysis Standings
    print("\n[Trace Analysis Standing Matrix]")
    total_scores = 0
    for item in phase_1:
        print(
            f" - Score: {item['score']} | Queries Executed: {item['insights']['total_db_queries']} | Prompt: {item['prompt'][:60]}...")
        total_scores += item['score']
    print(f"** Average Framework Quality Grade: {round(total_scores / len(phase_1), 2) if phase_1 else 0.0}")

    # Process and map Promptfoo comparison matrix blocks
    if phase_2 and "results" in phase_2:
        print("\n[Cross-Model Verification Summary Matrix]")
        table_rows = phase_2["results"].get("table", [])

        for idx, row in enumerate(table_rows):
            query_text = row.get("vars", {}).get("user_query", "Unknown Query")
            outputs = row.get("outputs", [])

            print(f" \nScenario #{idx + 1}: {query_text[:80]}...")
            for out in outputs:
                provider_label = out.get("provider", "Unknown")
                latency = out.get("latencyMs", 0)
                txt_sample = str(out.get("text", "None"))[:100].replace('\n', ' ')
                print(f"   ├── {provider_label} | Latency: {latency}ms | Response: {txt_sample}...")

            # Populate programmatic export JSON tree structural nodes
            report_structure["phase_2_cross_model_comparison"][f"scenario_{idx + 1}"] = {
                "query": query_text,
                "models_compared": [o.get("provider") for o in outputs]
            }

    # Persist the final consolidated report to your drive directory
    report_filename = "final_qa_compiled_report.json"
    with open(report_filename, 'w', encoding='utf-8') as rf:
        json.dump(report_structure, rf, indent=4)
    print(f"\n✔ Complete consolidated analytical metrics log exported to file workspace: {report_filename}")


if __name__ == "__main__":
    # Execute the testing package steps sequentially
    p1_data = run_point_1_internal_trace_evaluation()
    p2_data = run_point_2_promptfoo_cross_matrix()
    compile_final_qa_report(p1_data, p2_data)
