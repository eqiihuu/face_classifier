# __Face Detection and Facial Expression Classification__

___
## Author
* __Qi Hu :__  qihucn@uw.edu
* __Xinyu Zhao :__ xinyu94@uw.edu
* __Cunzhi Ren :__ renc@uw.edu

___

## Introduction
 * This is a project for face detection and facial expression classification
 * We use Haar Cascade algorithm provided by openCV to detect faces in the video/image
 * We compare different model for facial expression classification: kNN, SVM and CNN.
 * A demo is provided, where you can run the detection and classification in real time, and the program will replace
 the face with a corresponding emoji sign.
 * For more details, please refer to the POSTER 'poster_facedetect.pdf'

___

## Usage
 * __Train (CNN model) :__ run script cnn.py/cnn_keras.py to train a CNN classification model (cnn.py is the tensorflow
  version, cnn_keras.py is the keras version)  
 * __Test (CNN model):__ the model will be ran on test set each epoch. to draw the curve of training and testing process, run eval.py  
 * __kNN, SVM :__ knn.py is for kNN testing, svm.py is for SVM training and testing;
 * __Demo :__ run the script demo.py to start the demo. This demo will use a camera to obtain real time video, and automatically
   replace a detected face with corresponding emoji sign  
   
 * (model parameters can be adjusted in scripts cnn.py/cnn_keras.py, knn.py and svm.py)