"""Quick test: which models work on HF router?"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BASE = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

models_to_test = [
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

client = OpenAI(base_url=BASE, api_key=TOKEN)

for model in models_to_test:
    print(f"  Testing {model}...", end=" ", flush=True)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5, temperature=0.0,
        )
        print(f"OK -> {r.choices[0].message.content.strip()}")
    except Exception as e:
        err = str(e)[:120]
        print(f"FAIL -> {err}")
