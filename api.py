import sys
import logging

import g4f
import pytgpt.phind as phind
from g4f.client import Client

import nest_asyncio
from flask import Flask, request, jsonify, redirect

# patch nested event loops
nest_asyncio.apply()
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
    errors = []
    # this must be set to either g4f or tgpt, using other values will trigger a TypeError
    mode = request.json.get("type") or ""
    disabled_modes = []

    # to combat instability, quit only if there were more than 5 errors while fetching
    for i in range(5):
        logging.info(f"Fetching response... ({i+1}/5)")  # pylint: disable=W1203

        try:
            if mode == "tgpt" and not mode in disabled_modes:
                # fetch with tgpt (best provider: Phind)
                ai = phind.PHIND(max_tokens=400, timeout=None)
                gpt_message = ai.chat(system_prompt + request.json.get("prompt"))
            elif mode == "g4f" and not mode in disabled_modes:
                # fetch with g4f (best provider: GeminiProChat)
                ai = Client()
                response = ai.chat.completions.create(
                    model="gemini-pro",
                    provider=g4f.Provider.FlowGpt,  # may get ratelimited
                    messages=[
                        {"role": "user", "content": request.json.get("prompt")},
                        # {"role": "user", "content": system_prompt + request.json.get("prompt")},
                        {"role": "system", "content": system_prompt},
                    ],
                    timeout=10,  # limit time taken for clyde to respond to a max of 50 seconds
                    max_tokens=400,  # limit the output to 2000 or less characters
                )

                gpt_message = response.choices[0].message.content
            else:
                logging.warning("Discarding unavailable options")
                raise TypeError("Unavailable provider library provided")
        except Exception as e:
            # log a general error and retry
            if "429" in str(e):
                logging.warning("We are being ratelimited.")
                errors.append(f"RuntimeError: Ratelimited")
                break
            logging.warning(f"An exception occurred: {e.__class__.__name__}: {str(e)}")  # pylint: disable=W1203
            errors.append(f"{e.__class__.__name__}: {str(e)}")
            continue

        if gpt_message == "":
            # log a blank message error and retry
            logging.warning("No message was returned")
            errors.append("RuntimeError: No message was returned")
            continue

        # return the ai response
        logging.info("Message fetched successfully")
        return jsonify(
            {
                # splitting is designed for behavior of llama2's conversation features, may not split correctly in some edge cases
                "message": "".join(list(gpt_message))
                .lower()
                .split("user: ", 1)[0]
                .replace("clyde: ", ""),
                "code": 0,
            }
        ), 200

    # log the failure and quit fetching, send error back to the bot
    logging.error("Could not fetch message due to previously encountered issues")
    return jsonify(
        {
            "error": "Unable to fetch AI response. Possible reasons may include ratelimits, CAPTCHAs, or a broken provider.",
            "errors": errors,
            "code": 1,
        }
    ), 500


# run server at port 8001, debug mode to get hot-reloading without conflicts
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
