from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Invalid input"}), 400

    text = data["text"]
    result = "Offensive" if "hate" in text.lower() else "Not Offensive"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
