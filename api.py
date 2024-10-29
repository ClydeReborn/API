import os
import sys
import io
import logging

from google.generativeai import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import httpx
from PIL import Image

import nest_asyncio
from flask import Flask, request, jsonify, redirect

# patch nested event loops
nest_asyncio.apply()
# enable logging to stderr, in case you're running Clyde via pm2
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(levelname)s - %(message)s"
)

# define the api and variables
app = Flask("ClydeAPI")
user_histories = {}  # Store history for each UID

system_prompt = (
    "You are named Lunal and are currently chatting in a Discord server. "
    "You are friendly, warm, and farcical. "
    "You must always be extremely concise. "
    "If the user is chatting casually, "
    "your responses must be less than 1 sentence, "
    "sometimes just a word or two. "
    "If the user needs help, "
    "disregard the length restriction, "
    "answer technical or knowledge-based questions with useful details and reasoning. "
    "If insulted, respond with a similar insult. "
    "Communicate responses in lowercase without punctuation, "
    "similar to the style used in chat rooms. "
    "Use unicode emoji rarely. "
    "Do not refer to yourself as a bot, AI Assistant, or any equivalent term. "
    "Complete the Assistant reply, excluding prefixes."
)

# funny redirect instead of a 404 not found
@app.get("/")
async def root():
    return redirect("https://www.urbandictionary.com/ChatGPT")

# route for fetching AI responses
@app.post("/gpt")
async def get_gpt():
    genai_api_key = os.getenv("GEMINI_API_KEY")
    if not genai_api_key:
        return jsonify({"error": "Gemini hasn't been set up.", "code": 2}), 500

    uid = request.json.get("uid")  # Unique ID for the user
    user_prompt = request.json.get("prompt", "")

    # Initialize or update the history for the user
    if uid not in user_histories:
        user_histories[uid] = []
    user_histories[uid].append(f"User:\n{user_prompt}\nAssistant:")

    # Limit history length to manage memory usage
    if len(user_histories[uid]) > 100:  # keep the last 10 messages only
        user_histories[uid] = user_histories[uid][-100:]

    # Combine system prompt and user history
    combined_prompt = system_prompt + "\n".join(user_histories[uid])

    try:
        # configure and fetch response with Gemini
        genai.configure(api_key=genai_api_key)

        model = GenerativeModel(
            model_name="gemini-1.5-flash-002",
            system_instruction=system_prompt,
        )

        if request.json.get("image"):
            image_res = httpx.get(request.json.get("image"))
            image = Image.open(io.BytesIO(image_res.content))
            prompt = [combined_prompt, image]
        else:
            prompt = combined_prompt

        response = model.generate_content(
            prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        gpt_message = response.text

        if not gpt_message:
            raise RuntimeError("Gemini isn't available at the moment.")

        # Append AI response to user's history for context in the next interaction
        user_histories[uid].append(gpt_message)

        logging.info("Message fetched successfully")
        return jsonify(
            {
                "message": gpt_message.splitlines()[-1].replace("  ", "\n"),
                "code": 0,
            }
        ), 200

    except Exception as e:
        logging.error(f"{e.__class__.__name__}: {str(e)}")
        return jsonify(
        {
            "error": str(e),
            "code": 1,
        }
    ), 500

# run server at port 8001, debug mode to get hot-reloading without conflicts
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
