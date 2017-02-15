from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
#from keras.layers.pooling import MaxPooling2D
import pandas as pd
from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from random import randint

from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot



image_width = 200
image_height = 66
save_process = 1
save_flip = 1

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
    global save_process
    if save_process == 1:
        save_process = 0
        cv2.imwrite('processed_image.png', image)
        
    return image

def preprocess(data):
    """
        Loads and Preprocesses the image before training 
        Randomly selects left, center, or right camera
        Randonly flips 50% of images
        Args:
            data (data_frame): The matrix containing the data from the csv file

        Returns:
            image: the processed image
            steering angle: the steering angle associated with the image (video frame)
    """
    #randonly load left, right, or center image 
    index = np.random.randint(3)
    cameras = ['center', 'left', 'right']
    steering_offset = [0, 0.25, -0.25]
    image_path = data[cameras[index]]
    image = cv2.imread(image_path.strip())
    #set size of image after minimizing
    image = process_image(image)
    
    #offset steering angle for left and right
    steering_angle = data['steering'] + steering_offset[index]
    
    #flip 25% of images to account for left tendency in data
    flip = np.random.random()
    if flip < 0.25:
        steering_angle = -1 * steering_angle 
        image = cv2.flip(image, 1)
        
        global save_flip
        if save_flip == 1:
            save_flip = 0
            cv2.imwrite('flip_image.png', image)
    
    
    return image, steering_angle


def image_generator(data, batch_size):
    """
        Generates (Generator) batches of data for the training model
        Args:
            data (data_frame): The matrix containing the data from the csv file
            batch_size (int): size of the batches

        Returns:
            image: the processed image
            steering angle: the steering angle associated with the image (video frame)
    """

    batches_per_epoch = data.shape[0] // batch_size

    i = 0
    while(True):
        start = i * batch_size
        end = start + batch_size - 1

        X_batch = np.zeros((batch_size, image_height,image_width, 3), dtype=np.float32)
        y_batch= np.zeros((batch_size,), dtype=np.float32)

        j=0 
        for index, row in data.loc[start:end].iterrows():
            #center image
            
            X_batch[j], y_batch[j] = preprocess(row)
            j = j + 1

        i = (i + 1) % batches_per_epoch
        yield X_batch, y_batch
        
        
def get_model():
    """
        Creates the cnn model (based on NVIDIA model)
        Args:
            data (data_frame): The matrix containing the data from the csv file
            batch_size (int): size of the batches

        Returns:
            image: the processed image
            steering angle: the steering angle associated with the image (video frame)
    """
    model = Sequential()

    
    
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(image_height, image_width,3)))
    #NVIDIA model
    #first three convolutional layers with a 2×2 stride and a 5×5 kernel
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add((Dropout(0.5)))   
    model.add(Activation('relu'))   
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid")) 
    model.add((Dropout(0.5)))   
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Activation('relu'))

    #non-strided convolution with a 3×3 kernel size in the last two convolutional layers
    model.add(Convolution2D(64, 3, 3))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    
    model.add(Activation('relu'))

    #3 fully connected layers
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add((Dropout(0.5)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    #SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png')
    
    return model
    

if __name__ == '__main__':
    
    #load data from driving log
    data = pd.read_csv("driving_log.csv", usecols=[0, 1, 2, 3])
    
    print(data.shape[0])
 
    #shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    
    #slip into training and validation data, 80/20
    training_split = 0.8
    split_line = int(data.shape[0] * training_split)
    training_data = data.loc[0:split_line-1]
    validation_data = data.loc[split_line:]

    batch_size = 64

    model = get_model()

    training_generation = image_generator(training_data, batch_size)
    validation_generation = image_generator(validation_data, batch_size)
    
    print('training size')
    print(training_data.shape[0])
    
    print('val size')
    print(validation_data.shape[0])
    
    samples_per_epoch = data.shape[0] // batch_size * batch_size
    nb_epoch = 10
    nb_val_samples = validation_data.shape[0]

    model.compile(optimizer="adam", loss="mse")
    
    print(model.summary())

    hist = model.fit_generator(training_generation, samples_per_epoch=samples_per_epoch,nb_epoch=nb_epoch,
        validation_data=validation_generation, nb_val_samples=nb_val_samples, verbose=2)
    print(hist.history)
        

    #save weights and model
    model.save_weights('model.h5') 
    
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())
