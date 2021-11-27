from flask import Flask, render_template,request
from src.utils.all_utils import read_yaml
from flask_cors import CORS, cross_origin
import webbrowser
from threading import Timer
import pandas as pd
import pickle
import logging
import os
import subprocess
import yaml
import argparse

import clientApp
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'webapp.log'), level=logging.INFO, format=logging_str,
                    filemode="a")


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            USER_NAME = request.form['USER_NAME']
            USER_PASSWORD = request.form['USER_PASSWORD']
            dict_file = {
                'USER_NAME': USER_NAME,
                'index_name' : 'face_recognition'
            }
            with open('config/config.yaml', 'w') as file:
                yaml.dump(dict_file, file)
            return render_template('login.html')
        except Exception as e:
            logging.error(e)
    else:
        return render_template('login.html')


@app.route('/signup', methods=['POST', 'GET'])
@cross_origin()
def register():
    if request.method == 'POST':
        try:
            USER_NAME = request.form['USER_NAME']
            EMAIL_ID = request.form['EMAIL_ID']
            USER_PASSWORD = request.form['USER_PASSWORD']
            dict_file = {
                'USER_NAME': USER_NAME,
                'index_name' : 'face_recognition'
            }
            with open('config/config.yaml', 'w') as file:
                yaml.dump(dict_file, file)
            generate = clientApp.getFaceEmbeddings()
            return render_template('signup.html', ouput = generate)
        except Exception as e:
            logging.error(e)
    else:
        return render_template('signup.html')


def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')


def start_app():
    Timer(1, open_browser).start()
    app.run(host="127.0.0.1", port=8080,debug=True)


if __name__ == "__main__":
    start_app()

