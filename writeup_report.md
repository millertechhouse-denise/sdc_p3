#**Behavioral Cloning** 

##Denise Miller

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./processed_image.png "Processed Image"
[image3]: ./wall1.png "Recovery Image"
[image4]: ./redline1.png "Recovery Image"
[image5]: ./yellowline1.png "Recovery Image"
[image6]: ./processed_image.png "Normal Image"
[image7]: ./flip_image.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a 3 convolutional layers with a 2×2 stride and a 5×5 kernel followed by non-strided convolution with a 3×3 kernel size in the last two convolutional layers followed by 3 fully connected layers 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

(code lines 137-164)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 193-194). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 206).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center driving for 2 passes through the driving path.  Recoverying from left and right lane lines was also added.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

Sources: "End-to-End Deep Learning for Self-Driving Cars" and Slack

The overall strategy for deriving a model architecture was to take an existing, validated model and adapt it to this application.

My first step was to use a convolution neural network model similar to the NVIDIA model, since it was a proven method for this implementation.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80/20). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.  To combat the overfitting, I modified the model to include dropout.

To supplement data, I included images from all 3 cameras.  

Because I started with a proven model and input from slack, my main focus on the project was how to determine what was sufficient training data.  When learning to use the simulator, I would swerve a lot.  Thiw would cause my model to be trained to swerve which would cause it to go off-road.  I needed to collect data with smoother motion.  I also attempted to include data from the second track, but this descreased the success and I therfore focused on the first track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, though it would slow down while driving along the Bridge wall.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 98, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 31, 98, 24)    0           dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       activation_1[0][0]               
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 14, 47, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 14, 47, 36)    0           dropout_2[0][0]                  
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 22, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       activation_3[0][0]               
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 3, 20, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 20, 64)     0           dropout_3[0][0]                  
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       activation_4[0][0]               
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 18, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          1342092     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 1,595,511
Trainable params: 1,595,511
Non-trainable params: 0
____________________________________________________________________________________________________



Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .recover if it drifeted in the incorrect direction These images show what a recovery looks like starting from the left and right :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would address the fact that most of the track turned left.  I did not want the car to drift to the left (this can also effect its ability to recover). For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 400003 number of data points. I then preprocessed this data by converting color to YUV, clipping the data and resizing the data to make it smaller.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the fact that the accuracy did not improve as more epochs were added. I used an adam optimizer so that manually training the learning rate wasn't necessary.
