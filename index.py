from flask import Flask, render_template
import threading
import os
from jinja2 import Environment, FileSystemLoader
import json
app = Flask(__name__)

@app.route('/')
def getAllWorkouts():
    environment = Environment(loader=FileSystemLoader("templates/"))
    with open('DB.json', 'r') as json_file:
        data = json.load(json_file)
    return render_template("workouts.html", workouts=data)


@app.route('/user')
def catch_all(path):
    return 'You want path: %s' % path

if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)

    
    