import base64
import json

import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import boto3
import pandas as pd
from flask import Flask
from flask_restful import Api, Resource, abort, reqparse

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('base64Image',required=True, help="Base64 Image string is required")

class_name = ['fifi','webparts']

def convertImageForIngestion(base64Str):
    picture = base64Str.replace("/","_")
    picture = picture.replace("+","-")
    dc = tf.io.decode_base64(picture)
    dc = tf.io.decode_png(dc,3, tf.dtypes.uint8)
    dc = tf.image.resize(dc, (800,800))
    dc = tf.reshape(dc,(1,800,800,3))
    return dc

# Predict app via ML model
class PredictApp(Resource):
    def post(self):
        # verifying the request object
        args = parser.parse_args()
        base64Image = args['base64Image']

        tensorImageArr = convertImageForIngestion(base64Image)
        model = tf.keras.models.load_model('models/trained_model.h5')
        predictions = model.predict(tensorImageArr,steps=1) 
        predicted_app_name = class_name[np.argmax(predictions)]
        
        response = {'prediction': predicted_app_name}
        return response

class PredictAppByImageText(Resource):
    def post(self):
        # parse the body json
        args = parser.parse_args()
        imageBytes = base64.b64decode(args['base64Image']) 

        with open('text-classification-config.json') as f:
            classConfig = json.load(f)

        client = boto3.client('textract', region_name='us-east-1')
        response = client.detect_document_text(
        Document={
            'Bytes': imageBytes
            }
        )

        textArr = []
        for block in response['Blocks']:
            if 'Text' in block:
                textArr.append({"text": block['Text'], "confidence": block['Confidence']})
        
        labels = pd.DataFrame(classConfig)
        data = pd.DataFrame(textArr)

        df = data[(data['text'].str.len()>=3)].head(50)

        predicted_app_name = []
        for app in labels.columns:
            labelTextArr = labels[app].array
            labelTexts = "|".join(labelTextArr)
            if ((df['text'].str.contains(labelTexts)).any()):
                predicted_app_name.append(app)
        predicted_app_name = " or ".join(predicted_app_name)
        
        response = {'prediction': predicted_app_name}
        return response

##
## Actually setup the Api resource routing here
##
api.add_resource(PredictApp, '/predict/appByImage')
api.add_resource(PredictAppByImageText, '/predict/appByImageText')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
