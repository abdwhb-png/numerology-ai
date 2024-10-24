import os
import uuid
from flask import Flask, request, jsonify

from graphs.chat_workflow import graph as chat_bot
from functions.chat import generate_response

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Autoriser toutes les origines pour toutes les routes


def parse_responses(generation):
    return {
        "answer": generation["answer"],
        "metadata": {
            "steps": generation["steps"],
            "documents": generation["documents"],
            "context": generation["context"],
        },
    }


@app.route("/")
def main():
    return f"{os.getenv('APP_NAME')} is running ✅ "


@app.route("/test")
def testAI():
    generated = generate_response(chat_bot, "Hello, what can you do for me ?")
    return jsonify(parse_responses(generated["generation"]))


@app.route("/chat", methods=["POST"])
def webhook():
    try:
        if request.is_json:
            data = request.get_json()

            thread_id = data["thread_id"] or str(uuid.uuid4())
            generated = generate_response(
                ai=chat_bot, input=data["user_input"], thread_id=thread_id
            )

            responses = parse_responses(generated["generation"])
            responses["thread_id"] = thread_id

            res = (
                jsonify(
                    {
                        "status": "success",
                        "msg": "Everything went well.",
                        "responses": responses,
                    }
                ),
                200,
            )
        else:
            res = (
                jsonify(
                    {
                        "status": "error",
                        "msg": "Invalid request, expecting JSON",
                        "responses": [],
                    }
                ),
                400,
            )
    except Exception as e:
        res = (
            jsonify(
                {
                    "status": "error",
                    "msg": "API error: " + str(e),
                    "responses": [],
                }
            ),
            500,
        )
    finally:
        return res


if __name__ == "__main__":
    if os.getenv("APP_ENV") == "production":
        app.run()
    else:
        app.run(port=5000)
