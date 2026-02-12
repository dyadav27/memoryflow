<<<<<<< HEAD
from transformers import pipeline

chat_model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct",
    max_new_tokens=200
)

def generate_response(prompt):
    return chat_model(prompt)[0]["generated_text"]
=======
from transformers import pipeline

chat_model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct",
    max_new_tokens=200
)

def generate_response(prompt):
    return chat_model(prompt)[0]["generated_text"]
>>>>>>> ddf6092 (Initial MemoryFlow submission)
