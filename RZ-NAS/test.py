import os
import openai
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()

# 2. Set the API key (Standard OpenAI doesn't need api_base or api_type)
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    # 3. Call the API using the 'model' parameter
    # Note: Use 'model' for Standard OpenAI, not 'engine'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4o"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Are we connected?"}
        ],
        max_tokens=10
    )
    
    # 4. Print the result (Dictionary style for v0.28.x)
    print("Connection Successful!")
    print("Response:", response['choices'][0]['message']['content'])

except Exception as e:
    print(f"Connection Failed: {e}")