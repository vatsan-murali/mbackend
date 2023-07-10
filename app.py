from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Resize
import tensorflow as tf
from flask_cors import CORS
from io import BytesIO
import torch
import cv2
import sys
import os
import subprocess
import torchvision.transforms as transforms
import yolov5
import base64
import os
from flask import Flask, request
from flask_cors import CORS
from PIL import Image, ImageDraw
from io import BytesIO
import requests 


app = Flask(__name__)
CORS(app, origins="https://mfrontend-nu.vercel.app/")

yolovm = yolov5.load('yolo.pt')
yolovm.conf = 0.4
classes = {0: 'red', 1:'blue' , 2: 'green', 3: 'yellow'}
model = tf.keras.models.load_model('malaria_cnn_model.h5')


def preprocess_image(image):
    image = image.resize((50, 50))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array



@app.route('/predict', methods=['POST'])
def predict():
    print('Entered predict')
    
    file = request.files['image']
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
    image_array = Image.fromarray(img, 'RGB')

    pimage = preprocess_image(image_array)
    
    
    prediction = model.predict(pimage)

    print(prediction)
    print('prediction over')
    
    idx_1 = prediction[0]
    print(idx_1)
    result = idx_1[0]
    print(result)
    
    if result >= 0.5 :
        res = 'Parasitized'
    else :
        res = 'Uninfected'
    
    return jsonify({'prediction': res})

@app.route('/advm', methods=['POST'])
def advm():
    ext = request.files['image'].filename.split('.')[-1]
    image = Image.open(request.files['image'])
    image = image.resize((640, 640))
    p = yolovm(image).pred[0]
    r = 0
    b = 0
    g = 0
    y = 0
    draw = ImageDraw.Draw(image)
    for box in p:
        x1, y1, x2, y2 = box[:4]
        if box[5] == 0:
            r += 1
            label = "Trophozoite"
            color = "red"
        elif box[5] == 1:
            b += 1
            label = "Ring"
            color = "blue"
        elif box[5] == 2:
            g += 1
            label = "Schizont"
            color = "green"
        elif box[5] == 3:
            y += 1
            label = "Gametocyte"
            color = "yellow"
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=1)
        label_width, label_height = draw.textsize(label)
        label_rectangle = [(x1, y1 - label_height - 4), (x1 + label_width, y1 - 4)]
        draw.rectangle(label_rectangle, fill=color)
        draw.text((x1, y1 - label_height - 4), label, fill="white")

    image_io = BytesIO()
    image.save(image_io, format=ext)
    image_io.seek(0)
    response = {
        'image': base64.b64encode(image_io.getvalue()).decode('utf-8'),
        'trophozoite': r,
        'ring': b,
        'schizont': g,
        'gametocyte': y,
        'all': len(p)
    }
    return response


if __name__ == '__main__':
    app.run(debug=True)
