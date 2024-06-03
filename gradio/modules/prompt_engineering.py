import os
import json
from pathlib import Path
import openai

BASE_DIR = Path().resolve()

with open(BASE_DIR/'secrets.json') as f:
    secrets = json.loads(f.read())
    
SECRET_KEY = secrets['openai_key']


def gen(x):
  gpt_prompt = [{
      "role": "system",
      "content": ("You're an artificial intelligence chatbot with a lot of imagination."
                  "Look at the words you're presented with and imagine what you look like and describe them in detail.\n\n"
                  "Example:\n"
                  "Input: 귀여운 아기 공룡"
                  "Output: bady dinosaur, adorable, pink, spotted, short neck, four legs, two small wings"),
  }]

  gpt_prompt.append({
      "role":"user",
      "content": ("Imagine the word below and describe their appearance in English in about 20 words,"
                  f"using mainly none and adjectives, separted by commas:\n\n{x}")
  })

  gpt_response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=gpt_prompt
  )
  return gpt_response["choices"][0]["message"]["content"]