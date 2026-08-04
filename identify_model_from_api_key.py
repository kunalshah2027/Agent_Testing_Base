import os
from groq import Groq

# Initialize the client with your key
client = Groq(api_key="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")

# Fetch and print all available models
models = client.models.list()
for model in models.data:
    print(model.id)
