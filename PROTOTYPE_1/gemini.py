from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()

client = InferenceClient(token=os.getenv("HF_TOKEN"))

response = client.chat_completion(
    model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    messages=[{"role": "user", "content": "Say hello"}],
    max_tokens=50,
)
print(response.choices[0].message.content)