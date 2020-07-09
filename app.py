# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Krish Naik
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='fruit.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path):
    from PIL import Image

    pic = Image.open(img_path)
    img = image.load_img(img_path, target_size=(150, 150))
    array = image.img_to_array(img)
    x = np.expand_dims(array, axis=0)

    vimage = np.vstack([x])

    res = model.predict(vimage)

    result = np.where(res[0] == 1)

    if result[0][0] == 0:
        return "This is an Apple"
    if result[0][0] == 1:
        return "This is a Banana"
    if result[0][0] == 2:
        return "This is an Betroot"
    if result[0][0] == 3:
        return "This is a Guava"
    if result[0][0] == 4:
        return "This is a Lychee"
    if result[0][0] == 5:
        return "This is a Orange"
    if result[0][0] == 6:
        return "This is a Pear"
    if result[0][0] == 7:
        return "This is a PineApple"
    if result[0][0] == 8:
        return "This is a Pomegranate"
    if result[0][0] == 9:
        return "This is a Watermelon"



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
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
