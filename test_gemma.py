from ollama import chat

response = chat(
    model="gemma4",
    messages=[
        {
            "role": "user",
            "content": "Explain traffic volume prediction in simple words."
        }
    ]
)

print(response.message.content)