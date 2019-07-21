# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras_retinanet import models
from google.cloud import storage
from keras import backend as K
from datetime import datetime
from detector import gkDetect
import tensorflow as tf
import timeit
import os

app = Flask(__name__)
model = None
sess = None

def load():
    
    # prep
    mod =
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    global sess
    sess = tf.Session(config=config)
    K.set_session(sess)
    global model
    model = models.load_model("{}".format(mod), backbone_name="resnet50")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def dynamic_page():
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 
    bucket =
    folder =
    
    ### Upload ###
    upload_start = timeit.default_timer()
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    
    gcs = storage.Client()
    bucket = gcs.get_bucket('{}'.format(bucket))
    dt = datetime.today().strftime('%Y%m%d')
    
    blob = bucket.blob('{}/{}_input_'.format(folder,dt) + file.filename)
    blob.upload_from_string(file.read(), content_type=file.content_type)
    
    upload_stop = timeit.default_timer()
    upload_time = round((upload_stop - upload_start),1)
    ###
    
    ### Detect ###
    detect_start = timeit.default_timer()
    
    f = blob.public_url
    conf = request.form['conf']
    buff = request.form['buff']
    output = gkDetect(f, filename, conf, buff, model, sess)
    
    detect_stop = timeit.default_timer()
    detect_time = round((detect_stop - detect_start),1)
    ### 

    if request.method == 'POST':
        
        conf = 'Confidence: ' + str(conf)
        buff = 'Buffer: ' + str(buff)
        upload_time = 'Upload Time: ' + str(upload_time)
        detect_time = 'Detect Time: ' + str(detect_time)

        return (render_template('home.html', upload = f, results = output, confidence = conf, buffer = buff,
                                uploadTime = upload_time , detectTime = detect_time))

if __name__ == '__main__':
    print("Loading Keras model and starting Flask server...")
    load()
    app.run(host='127.0.0.1', port=8050, debug=True)