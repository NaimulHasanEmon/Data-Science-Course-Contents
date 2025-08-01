from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        print(request.form)
        email = request.form["email"]
        password = request.form["password"]
        print(f"E-mail: {email} and password: {password}")
        # Save it to the database
        return "<b>Thanks for login</b>"
    return render_template("index.html")

app.run(debug=True)