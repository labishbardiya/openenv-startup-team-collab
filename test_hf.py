from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

resp = client.responses.create(
    model=os.getenv("MODEL_NAME"),
    input=[{"role": "user", "content": "Say hello"}],
)

print(resp)
