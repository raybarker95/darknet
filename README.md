# CITS4402 Project #
Pedestrian Detection Project

Semester 1 2018 

## Team ##
Ray Barker 

Mitchell Poole

## Installation Instructions ##
####Install NVIDIA Driver####

Recommend installing NVIDIA driver 396 - [Guide](
http://tech.amikelive.com/node-731/how-to-properly-install-nvidia-graphics-driver-on-ubuntu-16-04/)

####Install NVIDIA CUDA####

Recommend installing Cuda 9.2 - [Guide](
http://tech.amikelive.com/node-669/guide-installing-cuda-toolkit-9-1-on-ubuntu-16-04/)

####Install NVIDIA CUDNN####
(Developer account required) - [Guide](
https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

####Install OpenCV####
sudo apt-get install libopencv-dev
 
####Clone Git Repo####
git clone https://github.com/raybarker95/darknet-pedestrian-detector

####Compile####
cd darknet-pedestrian-detector

make

####Download Pre-trained Model####

wget https://pjreddie.com/media/files/yolov3.weights

####Test installation####

./darknet detect cfg/yolov3.cfg yolov3.weights data/person.jpg

A box should be drawn around the person, with an estimation certainty shown as a %

####Install tkinter for GUI####

sudo apt-get install python-imaging-tk

####Install Pillow for images in tkinter####

sudo pip install Pillow

####Install opencv for python####

sudo pip install opencv-python

# Forked from Darknet #
![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
