import os
import sys
import logging

import g4f
import google.generativeai as genai
from g4f.client import Client
from pytgpt import gpt4free

import nest_asyncio
from flask import Flask, request, jsonify, redirect

# patch nested event loops
nest_asyncio.apply()
# enable logging to stderr, in case you're running clyde via pm2
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(levelname)s - %(message)s"
)


def get_model(provider: g4f.Provider) -> str:
    try:
        return provider.default_model
    except AttributeError:
        return "gpt-3.5-turbo"


# define the api and variables
app = Flask("ClydeAPI")
system_prompt = (
    "You are named Lunal and are currently chatting in a Discord server. "  # rename Clyde here
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

    # uncomment to use any available provider
    # providers = [
    #    g4f.Provider.ProviderUtils.convert[p] for p in g4f.Provider.__all__ if p not in ["_", "Bing"]
    # ]
    # ok = [p for p in providers if all([p.working, not p.needs_auth, p.supports_stream])]

    # date tested: 18.09.2024
    ok = [g4f.Provider.ChatGot, g4f.Provider.HuggingChat, g4f.Provider.FreeChatgpt]
    img_ok = [g4f.Provider.Prodia, g4f.Provider.ReplicateHome]
    
    # to combat instability, try all providers individually
    for provider in ok:
        logging.info(f"Fetching response at {provider.__name__}...")  # pylint: disable=W1203

        try:
            if mode == "tgpt" and mode not in disabled_modes:
                # fetch with tgpt (best provider: Phind)
                ai = gpt4free.GPT4FREE(
                    intro=system_prompt,
                    max_tokens=400,
                    timeout=None,
                    provider=provider.__name__,
                    model=get_model(provider),
                    chat_completion=True,
                )
                gpt_message = ai.chat(system_prompt + request.json.get("prompt"))
            elif mode == "g4f" and mode not in disabled_modes:
                # fetch with g4f (best providers: GeminiProChat, HuggingChat)
                ai = Client()

                response = ai.chat.completions.create(
                    model=get_model(provider),
                    provider=provider.__name__,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.json.get("prompt")},
                    ],
                )

                gpt_message = response.choices[0].message.content
            elif mode == "gemini" and mode not in disabled_modes:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

                model = genai.GenerativeModel(
                    model_name="gemini-1.5-pro-002",  # Using Gemini 1.5 Pro model
                    system_instruction=system_prompt
                )
                response = model.generate_content(request.json.get("prompt"))

                gpt_message = response.text
            
            else:
                logging.warning("Discarding unavailable options")
                raise TypeError("Unavailable provider library provided")

            if not gpt_message:
                raise RuntimeError("No message was returned")

            if "[GoogleGenerativeAI Error]" in gpt_message:
                raise RuntimeError(f"{provider.__name__} did not work")

            if "当前地区当日额度已消耗完, 请尝试更换网络环境" in gpt_message:
                raise RuntimeError(f"{provider.__name__} quota is exhausted")

        except Exception as e:
            # log a general error and retry
            logging.warning(f"An exception occurred: {e.__class__.__name__}: {str(e)}")  # pylint: disable=W1203
            errors.append(f"{e.__class__.__name__}: {str(e)}")
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
    logging.error(errors)
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
