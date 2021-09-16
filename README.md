# License Plate Detection using modified SSD

In this project, we intend to implement a system for detecting car license plates by using Deep Learning. License plate detection has a variety of applications, such as detecting the stolen car, and speeding up the process of toll payment. There are several algorithms for Object Detection, each of them has its pros and cons. The SSD architecture is one of the best-known architectures for its high speed and low processing cost. Therefore, it is one of the best options to use in mobiles and Raspberry PI processors. Our objective is to modify the SSD architecture to improve its accuracy in detecting car license plates.
The SSD architecture is robust against some geometric transformations such as translation and scaling, but in case of being projective, its accuracy will decrease, especially in license plate detection. By investigating the SSD architecture and other object detection algorithms, we designed a neural network, in a way that we could determine the correspond projective transformation from its output. We would like to increase the accuracy and preserve the speed and low processing cost as the main features of the SSD architecture at the same time.

We used https://github.com/SeyedHamidreza/car_plate_dataset dataset.
