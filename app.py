import flask
from flask import Flask, render_template, request, url_for
import joblib
import feature_extractor
import regex
import pandas as pd
import numpy as np
import sys
import logging
from sklearn.metrics import accuracy_score


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        df.set_index('id', inplace=True)
        return render_template("preview.html",df_view = df)

@app.route('/test')
def test():
    return render_template('predict.html')


@app.route('/predict', methods = ['POST'])
def make_prediction():
    classifier = joblib.load('final_models/elm_final.pkl')
    if request.method=='POST':
        url = request.form['url']
        if not url:
            return render_template('predict.html', label = 'Please input url')
        elif(not(regex.search(r'^(http|ftp)s?://', url))):
            return render_template('predict.html', label = 'Please input full url, for exp- https://facebook.com')
        
        
        checkprediction = feature_extractor.main(url)
        prediction = classifier.predict(checkprediction)

        if prediction[0]==1 :
            label = 1
        elif prediction[0]==-1:
            label = -1
        
        return render_template('result.html', label=label)

        
        
if __name__ == '__main__':
    classifier = joblib.load('final_models/elm_final.pkl')
    app.run()