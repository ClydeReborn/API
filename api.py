import sys
import logging

import httpx
import ujson as json

from flask import Flask, request, jsonify, redirect

# enable logging to stderr, in case you're running clyde via pm2
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(levelname)s - %(message)s"
)

# define the api and variables
app = Flask("ClydeAPI")
system_prompt = (
    "You are named Clyde and are currently chatting in a Discord server. "  # rename Clyde here
    "You are friendly, warm and farcical. "
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
    "Always prefix your messages with the following sequence: 'clyde: ' "
)


# funny redirect instead of a 404 not found
@app.get("/")
async def root():
    return redirect("https://www.urbandictionary.com/ChatGPT")


# route for fetching ai responses
@app.post("/gpt")
async def get_gpt():
    try:
        response = httpx.post("http://127.0.0.1:11434/api/generate", json={
            "model": "llama3",  # use llama3 for the most cutting-edge, llama2-uncensored for freeness, llava for image support
            "template": f"FROM llama3\nPARAMETER num_ctx 400\nSYSTEM system_prompt",
            "prompt": request.json.get("prompt")
        }, timeout=None)
    
        gpt_message = ""
        response_json = response.text.split("\n")
        for doc in response_json:
            json_doc = json.loads(doc)
            if not json_doc["done"]:
                gpt_message += json_doc["response"]
            else:
                break
            
        if not gpt_message:
            raise RuntimeError("No message was returned")
    
        # return the ai response
        logging.info("Message fetched successfully")
        return jsonify(
            {
                # splitting is designed for behavior of llama2's conversation features, may not split correctly in some edge cases
                "message": gpt_message
                .lower()
                .split("user: ", 1)[0]
                .replace("clyde: ", ""),
                "code": 0,
            }
        ), 200
    except Exception as e:
        # log the failure and quit fetching, send error back to the bot
        logging.error("Could not fetch message due to previously encountered issues")
        return jsonify(
            {
                "error": "Unable to fetch local AI response, please start Ollama.",
                "errors": [f"{e.__class__.__name__}: {str(e)}"],
                "code": 1,
            }
        ), 500


# run server at port 8001, debug mode to get hot-reloading without conflicts
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
