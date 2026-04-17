import os
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import InferenceClient

client = InferenceClient(provider="featherless-ai", api_key=os.getenv("HF_TOKEN"))

try:
    r = client.chat_completion(model="google/gemma-2-9b-it", messages=[{"role":"user","content":"Hi"}], max_tokens=5)
    print("OK:", r.choices[0].message.content)
except Exception as e:
    print("FAIL:", str(e)[:200])