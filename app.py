from __future__ import division, print_function
# coding=utf-8
import os
import glob
import numpy as np
import librosa as lb

# Keras
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras import models

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

file_model = os.path.join(os.getcwd() ,'weights.h5')

# Define a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

def preprocessing(x):
    shape = np.array(x).shape
    l = (shape[1] - shape[1]%4)/4
    new = x[:,:int(l*4)]
    # print(x.shape)
    new = np.reshape(new, (64, int(l)))
    new = sequence.pad_sequences(new, maxlen = 120, padding="post")
    new = np.reshape(new, (120, 64))
    new = (new+464.4627)/666.9084
    return new

def model_predict(file_path):
    y, sr = lb.load(file_path)
    mfccs_features = lb.feature.mfcc(y=y, sr=sr, n_mfcc=16)
    x = preprocessing(mfccs_features)
    model = models.load_model(file_model)
    result = model.predict(x.reshape(1,120, 64))
    return result
    

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        
        if preds[0][0]>=0.5 :
            return {'message':"You Havecough and symptoms",'value':str(preds[0][0])}
        else:
            return {'message':"You are save you don't have symptops for covid 19",'value':str(preds[0][0])}
        return str(preds[0][0])
    return None


if __name__ == '__main__':
    app.run(debug=True)

