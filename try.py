import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()
models = client.models.list()
print("---YOUR ACTIVE GROQ MODELS---")
for model in models.data:
    print(model.id)