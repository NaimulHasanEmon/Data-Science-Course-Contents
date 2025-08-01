from flask import Flask, render_template

app = Flask(
    __name__,
    static_folder="assets",
    static_url_path="/files"
    )

@app.route("/")
def hello_naimul():
    return render_template("index.html")
# render template connects a html page with the server

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

app.run(debug=True)