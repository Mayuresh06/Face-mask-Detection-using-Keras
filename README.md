# Face-mask-Detection-using-Keras
During this pandemic it is very important for everyone to keep wearing a mask in public places and to ensure that everyone is wearing proper mask there should be an automated system which can detect multiple faces through a camera and detect if the person is wearing proper masks or not. So we tried to develop a system which can automatically detect if a person is wearing the mask or not.

![](https://github.com/Mayuresh06/Face-mask-Detection-using-Keras/blob/main/Images/image002.jpg)

As seen in the block diagram the problem statement is divided into two major parts, first part being training a CNN classifier model which can classify any given image into one of the three categories i.e. WITH_MASK, WITHOUT_MASK or WITH_IMPROPER_MASK.
The second part of the system includes deploying the face mask detection on a real time video feed where a MTCNN model is used for face detection, so after face is detected the Region of Interest is extracted then image is fed to the face detection model and the output is printed on to the screen.


## Training The Model:-
* Dataset: - we downloaded the image dataset from kaggle website, the dataset consisted of images of people wearing mask, not wearing mask and wearing improper masks. The total number of images present in the dataset was 6000, equally divided into three classes. 

![](https://github.com/Mayuresh06/Face-mask-Detection-using-Keras/blob/main/Images/image004.jpg)

* Keras Sequential: - The core idea of Keras sequential is simply arranging the Keras layers in a sequential order and so, it is called Sequential API. Most of the ANN also has layers in sequential order and the data flows from one layer to another layer in the given order until the data finally reaches the output layer.
The model consisted of total Trainable parameters are 1,625,315 and total numbers of layers are 4 convolution layers, 3 max pooling layers, 3 dropout layers and one flatten, dense layer each.

![](https://github.com/Mayuresh06/Face-mask-Detection-using-Keras/blob/main/Images/image006.jpg)

* Training the sequential model:- we used the relu activation function for training the model and to calculate the loss we used categorical_crossentropy loss function and as a optimiser we used Adam optimiser. We got a training accuracy of 98.23 percent and testing accuracy of 97.78 percent.

![](https://github.com/Mayuresh06/Face-mask-Detection-using-Keras/blob/main/Images/image007.png)  ![](https://github.com/Mayuresh06/Face-mask-Detection-using-Keras/blob/main/Images/image009.png)

* MTCNN Model:- MTCNN (Multi-task Cascaded Convolution Neural Networks) is an algorithm consisting of 3 stages, which detects the bounding boxes of faces in an image along with their 5 Point Face Landmarks. Each stage gradually improves the detection results by passing its inputs through a CNN, which returns candidate bounding boxes with their scores, followed by non max suppression.
We used MTCNN model to detect faces from the images, the mtcnn model gives an output of a bounding box which acts as our Region of Interest which is given as an input to the face detection model and the output is printed on the screen.
## OUTPUT:-

