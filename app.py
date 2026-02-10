from flask import Flask, render_template, redirect, url_for
from camera import start_camera
from database import init_db, get_attendance

app = Flask(__name__)

init_db()

@app.route("/")
def dashboard():
    data = get_attendance()
    return render_template("dashboard.html", data=data)

@app.route("/start")
def start():
    start_camera()
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True)
