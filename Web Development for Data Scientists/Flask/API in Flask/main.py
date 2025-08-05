from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    # Some machine learning model
    data = {"output": 45, "accuracy": 98.55}
    return jsonify(data), 200

app.run(debug=True)