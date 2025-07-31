from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello_naimul():
    return "<p>Hello, I'm Naimul!</p>"

@app.route("/about")
def about():
    return render_template("index.html")       # connects a html page with the server

@app.route("/contact")
def contact():
    return "<p>This is contact page</p>"

app.run(debug=True)