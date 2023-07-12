
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
CORS(app, origins="http://localhost:3000")

yolovm = yolov5.load('yolo.pt')
yolovm.conf = 0.4
classes = {0: 'red', 1:'blue' , 2: 'green', 3: 'yellow'}
model = tf.keras.models.load_model('malaria_cnn_model.h5')



@app.route('/ihc_patch',methods=['POST'])
def ihc_patch():
    file = request.files['image']
    if(file):
        file.save('pix2pix/input/testA/input.png')
        subprocess.run("python pix2pix/test.py --dataroot pix2pix/input/testA/ --name he_pix2pix --gpu_ids -1 --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch")
        original = open('results/he_pix2pix/test_latest/images/input_real.png','rb')
        image_data = original.read()
        orig_base64_data = base64.b64encode(image_data).decode('utf-8')
        fake = open('results/he_pix2pix/test_latest/images/input_fake.png', 'rb')
        image_data = fake.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
    return jsonify({'original': orig_base64_data,'image': base64_data})

@app.route('/he_patch',methods=['POST'])
def he_patch():
    file = request.files['image']
    if(file):
        file.save('pix2pix/input/testB/input.png')
        subprocess.run("python pix2pix/test.py --dataroot pix2pix/input/testB/ --name ihc_pix2pix --gpu_ids -1 --model test --netG unet_256 --direction BtoA --dataset_mode single --norm batch")
        original = open('results/ihc_pix2pix/test_latest/images/input_real.png','rb')
        image_data = original.read()
        orig_base64_data = base64.b64encode(image_data).decode('utf-8')
        fake = open('results/ihc_pix2pix/test_latest/images/input_fake.png', 'rb')
        image_data = fake.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
    return jsonify({'original': orig_base64_data,'image': base64_data})

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
    result = idx_1[0]

    if result >= 0.5:
        res = 'Parasitized'
    else:
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
        # Draw the bounding box rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=1)
        # Draw the label background rectangle
        label_width, label_height = draw.textsize(label)
        label_rectangle = [(x1, y1 - label_height - 4), (x1 + label_width, y1 - 4)]
        draw.rectangle(label_rectangle, fill=color)
        # Draw the label text
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

# from flask import Flask, request, jsonify
# from PIL import Image
# import numpy as np
# from torchvision.transforms import ToTensor, ToPILImage
# import tensorflow as tf
# from flask_cors import CORS
# import torch
# import cv2
# import sys
# import os

# # Get the absolute path to the pytorch-CycleGAN-and-pix2pix-master directory
# pytorch_dir = os.path.join(os.path.dirname(__file__), 'pix2pix')

# # Add the pytorch-CycleGAN-and-pix2pix-master directory to the Python path
# sys.path.append(pytorch_dir)

# # Now you can import the Pix2PixModel
# from pix2pix.models.pix2pix_model import Pix2PixModel
# from pix2pix.options.base_options import BaseOptions

# app = Flask(__name__)
# CORS(app, origins="http://localhost:3000")

# # Load the TensorFlow model
# model = tf.keras.models.load_model('malaria_cnn_model.h5')

# # Preprocess the image for TensorFlow model
# def preprocess_image_tf(image):
#     image = image.resize((50, 50))
#     image_array = np.array(image)
#     image_array = image_array / 255.0
#     image_array = np.expand_dims(image_array, axis=0)

#     return image_array

# # Load the PyTorch model
# opt = BaseOptions().parse()
# model_path = '200_net_D.pth'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# state_dict = torch.load(model_path, map_location=device)
# model = Pix2PixModel(opt)
# model.load_state_dict(state_dict)
# model.eval()
# model.to(device)

# # Preprocess the image for PyTorch model
# def preprocess_image_pt(image):
#     image = image.resize((256, 256))  # Adjust the size according to your model input requirements
#     image_tensor = ToTensor()(image).unsqueeze(0).to(device)
#     return image_tensor

# @app.route('/test', methods=['POST'])
# def test_model():
#     # Load and preprocess the input image
#     file = request.files['image']
#     pil_image = Image.open(file)
#     input_tensor = preprocess_image_pt(pil_image)

#     # Perform the forward pass
#     with torch.no_grad():
#         output_tensor = model(input_tensor)

#     # Perform any necessary post-processing on the output tensor
#     # For example, converting it to an image or extracting relevant information

#     # Return the output as a response
#     return jsonify({'result': output_tensor})


# @app.route('/predict', methods=['POST'])
# def predict():
#     print('Entered predict')

#     file = request.files['image']
#     img_bytes = file.read()
#     img_array = np.frombuffer(img_bytes, dtype=np.uint8)
#     img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
#     image_array = Image.fromarray(img, 'RGB')

#     # Preprocess the image
#     pimage = preprocess_image_tf(image_array)

#     # Make the prediction using the TensorFlow model
#     prediction = model.predict(pimage)

#     print(prediction)
#     print('prediction over')

#     idx_1 = prediction[0]
#     print(idx_1)
#     result = idx_1[0]
#     print(result)

#     if result >= 0.5:
#         res = 'Parasitized'
#     else:
#         res = 'Uninfected'

#     return jsonify({'prediction': res})

# if __name__ == '__main__':
#     app.run(debug=True)
