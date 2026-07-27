import os

from flask import Flask, jsonify, render_template, request

from chatbot import ChatbotEngine

app = Flask(__name__)
bot = ChatbotEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with a message field."}), 400

    text = data.get("message")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please enter a message."}), 400

    if len(text) > 1000:
        return jsonify({"error": "Please keep messages under 1,000 characters."}), 400

    return jsonify(bot.predict(text))


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true"},
        port=int(os.getenv("PORT", "5000")),
    )
