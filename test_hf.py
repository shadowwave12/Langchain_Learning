import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env
load_dotenv()

# Create Hugging Face client
client = InferenceClient(
    api_key=os.getenv("HF_TOKEN"),
    provider="featherless-ai"
)

# Send chat request
response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Nepal?"
        }
    ],
    max_tokens=100
)

# Print response
print(response.choices[0].message.content)