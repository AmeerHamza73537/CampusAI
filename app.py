from flask import Flask, render_template, request, jsonify
from chatbot import ChatbotEngine

app = Flask(__name__)
bot = ChatbotEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("message", "")
    res = bot.predict(text)
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
