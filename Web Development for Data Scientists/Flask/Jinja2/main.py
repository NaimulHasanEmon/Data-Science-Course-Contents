from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    # Declaring variables
    name = "Emon"
    language = "Python"
    lucky_numbers = [1, 3, 9, 12, 38, 69, 7, 21]
    footer = "<p>Copyright 2025 | All rights reserved</p>"

    # Passing variables as arguments
    return render_template(
        "index.html",
        name=name,
        lang=language,
        lucky=lucky_numbers,
        footer=footer
        )

app.run(debug=True)