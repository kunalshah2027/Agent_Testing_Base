import os
import json
import subprocess

# 1. SETUP AUTHENTICATION KEYS
os.environ["GROQ_API_KEY"] = "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"


def execute_promptfoo_matrix():
    print("--- 1. Launching Promptfoo Evaluation Matrix Across 10 Prompts ---")

    # Executes promptfoo evaluation run over your yaml file configuration matrix
    # Specifying output as json allows your Python script to parse the cross-verification variables natively
    result_file = "eval_results.json"
    cmd = ["promptfoo", "eval", "-c", "promptfoo.config.yaml", "-o", result_file]

    try:
        # Fires command execution block
        subprocess.run(cmd, check=True)
        print("\n--- 2. Promptfoo Evaluation Run Completed Successfully ---")

        # Read and parse the output JSON results file for custom QA cross-checks
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("\n=== CROSS-VERIFICATION ANALYSIS SUMMARY ===")
        results = data.get("results", {})
        table = results.get("table", [])

        # Loop through each evaluation query scenario turn inside the results tree
        for row in table:
            user_prompt = row.get("vars", {}).get("user_query", "Unknown Prompt")
            response_outputs = row.get("response", {})

            print(f"\n[Scenario Prompt]: {user_prompt}")

            # Isolate model execution outputs from the generated table structures
            for idx, provider_res in enumerate(row.get("outputs", [])):
                provider_label = provider_res.get("provider", "Unknown Model")
                output_text = provider_res.get("text", "No Output")
                latency_ms = provider_res.get("latencyMs", 0)

                print(f"  ├── Provider: {provider_label} (Latency: {latency_ms}ms)")
                # Truncating responses to keep console output readable
                # Create the clean single-line string first
                single_line_text = output_text[:140].replace('\n', ' ')

                # Print the pre-cleaned variable safely inside the f-string
                print(f"  │   └── Extracted Text: {single_line_text}...")

                #print(f"  │   └── Extracted Text: {output_text[:140].replace('\n', ' ')}...")
                print(f"  │   └── Extracted Text: {single_line_text}...")
        # 3. OPEN THE WEB INTERFACE VISUAL REPORT PANELS
        print("\nLaunching Promptfoo Interactive Web Viewer Dashboard to visually cross-verify grids...")
        subprocess.Popen(["promptfoo", "view"])

    except FileNotFoundError:
        print(
            "\n[CRITICAL ERROR]: The 'promptfoo' CLI is missing or not configured in your system's environmental PATH variables.")
        print("Please ensure you run: npm install -g promptfoo")
    except Exception as e:
        print(f"\nExecution Matrix interrupted: {e}")


if __name__ == "__main__":
    execute_promptfoo_matrix()
