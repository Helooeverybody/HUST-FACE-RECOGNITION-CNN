# Face recognition program (By group 23 Intro AI)

Members:

- Nguyen Nhat Minh 20225510
- Doi Sy Thang - 20225528
- Ngo Duy Dat - 20225480
- Ta Ho Thanh Dat - 20225482
- Nguyen Minh Quan - 20225520
- Status: Semi-Complete (pending approval from the group)

## Projects description

A python programme that uses cv2 face detection with a pretrained CNN model for face recognition.
The model is trained using Google Colab T4 GPU and tensorflow libraries on a dataset comprises of
99 different labels, each has about 20 images. The model should be capable to recognise human faces that
it is trained on and be able to put matching label with reasonable accuracy.

## CNN model

Our cnn model is inspired by the VGG-16 model and taylored to be able to be trained on a much smaller
dataset. The model accepts 224x224x1 grayscale images as input and the output will the probability
of each image being assigned to each label. The predicted label is considered to be the label
with the highest probability.

The model consists of 8 Convolution layers with ReLu activation followed by BatchNormalization to
perform feature extraction. The last few layers includes a Flatten layer, a Dense layer with ReLu
activation and a Dense layer with Softmax activation for the output. The model also use Adam
optimiser with custom learning rate and SparseCategoricalCrossentropy loss function.
Detail about the model's architecture can be found in ![summary.txt](./Models/summary.txt).

The dataset is split into two: data_train and data_test and the model is trained exclusively on
data_train. The model is then validate based on its accuracy on data_test and trained with
early escape to prevent increase in validation loss.
Detail about the model's performance can be found in ![performance.png](./Models/performance.png).

## Requirements

`requirements.txt`
cvzone==1.6.1
keras==2.14.0
matplotlib==3.8.0
numpy==1.26.0
opencv-python==4.8.1.78
tensorflow==2.14.0

tensorflow, keras : create and load cnn model
cv2 : video capture, face detection and create app ui
cvzone : create app ui (a true life saver)

## Folder structures

```
.
├── Assets/
├── Faces/
├── Models/
|   ├── summary.txt
|   ├── performance.png
|   ├── cnn_model.keras
|   ├── cnn_weights.keras
|   └── cnn_model.py
├── Modules/
|   └── detect_faces.py
├── create_model.ipynb
└── face_recognition.py
```

## How to use

- run face_recognition.py
- press "f" to flip camera
- press "c" to capture the detected face and give prediction
- press "s" to show captured face
- press "q" to quit
