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

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    # print(request.method)
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
            return render_template('register.html')
        except Exception as e:
            logging.error(e)
    else:
        return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
@cross_origin()
def signup():
    print(request.method)

    if request.method == 'POST':
        
        try:
            print("Inside Try")
            USER_NAME = request.form['USER_NAME']
            print(USER_NAME)
            EMAIL_ID = request.form['EMAIL_ID']
            print(EMAIL_ID)
            USER_PASSWORD = request.form['USER_PASSWORD']
            dict_file = {
                'USER_NAME': USER_NAME,
                'index_name' : 'face_recognition'
            }
            with open('config/config.yaml', 'w') as file:
                yaml.dump(dict_file, file)
            generate = clientApp.getFaceEmbeddings()
            
            return render_template('signup.html', output = generate)
        except Exception as e:
            print("Inside Except")
            logging.error(e)
    else:
        return render_template('signup.html')

@app.route('/predict', methods=['GET'])
@cross_origin()
def predict():
    print(request.method)

    if request.method == 'GET':
        
        try:  
            predict = clientApp.getFacePrediction() 
            return render_template('register.html', output = predict)
        except Exception as e:
            # print("Inside Except")
            logging.error(e)
    else:
        return render_template('register.html')

@app.route('/features', methods=['GET', 'POST'])
@cross_origin()
def features():
    print(request.method)

    if request.method == 'GET':
        
        try:  
            features = clientApp.getFaceFeatures() 
            return render_template('register.html', output = features)
        except Exception as e:
            # print("Inside Except")
            logging.error(e)
    else:
        return render_template('register.html')


def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')


def start_app():
    Timer(1, open_browser).start()
    app.run(host="127.0.0.1", port=8080,debug=True)


if __name__ == "__main__":
    start_app()

