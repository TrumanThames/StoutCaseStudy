from flask import Flask, flash, redirect, render_template, request
import os
from os import listdir
from random import randint

app = Flask(__name__)


@app.route("/")
def index():
    basename = os.path.basename
    histograms = [os.path.join(os.path.abspath(os.path.dirname(__file__)),'casestudy1_figures','histograms',fil) for fil in listdir(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                      'casestudy1_figures/histograms'))]
    return render_template('user.html', cwd=os.getcwd(), basenm=os.path.basename, **locals())


@app.route("/user/")
def hello():
    users = ["Frank", "Steve", "Alice", "Bruce"]
    return render_template('user.html', **locals())


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)