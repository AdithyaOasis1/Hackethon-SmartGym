from flask import Flask, render_template
import threading
import os
from jinja2 import Environment, FileSystemLoader
app = Flask(__name__)

@app.route('/')
def getAllWorkouts():
    environment = Environment(loader=FileSystemLoader("templates/"))
    # template = environment.get_template("workouts.html")
    return render_template("workouts.html", workouts={"userId": "adithya", "reps": 3})


@app.route('/user')
def catch_all(path):
    return 'You want path: %s' % path

if __name__ == '__main__':
    app.run()

    
    