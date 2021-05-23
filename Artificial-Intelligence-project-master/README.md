# Artificial-Intelligence-project
CSC 434 group project

This group project was assigned by Dr.Yu at the College at Brockport and completed by John Haag, Zach Kovalenko, and Thomas Shear.

The purpose of this project is to create a program that can detect the google captcha and beat it using modern Artificial intelligence Techniques.

We Used both an YOLOv5 object detection model as well as a TensorFlow object detection model.

The TensorFlow model was the model that is linked to a python program that detects the google reCaptcha, returns the screen coordinates back to the program, and passes the coordiantes to the python mouse events to click on the reCaptcha.

This project does not take into consideration the 3x3 box of multiple images or the audio challenge that some google reCaptcha version 2 present as a problem.
This project simply detects the object Captcha, draws a colored bounding box around it. finds coordiantes of the bounding box and returns them to click on the captcha in the real life browser to solve the captcha, and leave a checkmark in the box. 

path to the script that clicks the captcha is as follows:
models/models-master/research/object_detection/clicker.py

path to the object detection model is as follows:
Models/models-master/research/object_detection/ find object detection image something similar to this.

the Two google Colab notebooks are both YOLOv5 models.

We used the github of a fellow brockport college student that did a similar project his github is as follows:
https://github.com/criogenesis/Tensorflow-MachineLearning-Research

you can find the tutorial he used in his repo, he also does a version of a tutorial typed.

it is important to note alot of the code he used to solve a similar problem was changed to use the TensorFlow model with some updated technologies.
not everything in his github is also in ours. our project is the most up to date comparatively.




