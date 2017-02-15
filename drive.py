import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

image_width = 200
image_height = 66

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def process_image(image):
    """
        Preprocesses the image before training 
        Converts color to YUV, crops and resizes
        Args:
            data (data_frame): The matrix containing the data from the csv file

        Returns:
            image: the processed image
    """
    #convert to YUV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    #crop image
    image = image[60:140, :]
    
    #resize to smaller image
    image = cv2.resize(image, (image_width,image_height),interpolation=cv2.INTER_AREA)
        
    return image


@sio.on('telemetry')
def telemetry(sid, data):
    print('telemetry')
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    #pre-process images
    image = process_image(image_array)
    transformed_image_array = image[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile(optimizer="adam", loss="mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    print('file loaded')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    print('here')

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)