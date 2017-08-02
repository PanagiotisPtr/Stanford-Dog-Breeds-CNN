# Stanford-Dog-Breeds-CNN
A convolutional neural network that is able to recognize 120 different dog breeds from the Stanford dog dataset. The neural network is written 
using the TFLearn API and uses residual layers wich helps make the network deeper and thus understand more complex patterns in images.
Which is crucial since a lot of the dog breeds look similar to each other. Unfortunately I had to sacrifice the image size that was given to the model
in order to be able to make the network itself deeper. The images that are fed into the model are 32x32 and as such doesn't help it reach accuracies anywhere near
the neighborhood of 50% which, over 120 different classes, I would consider acceptable. Now this model achieves 27% accuracy on unseen data with 32x32 images and 
while trained on 0.8GB of VRAM on a GeForce GT740M on my laptop which is decent but not great. 

### The Dataset
The dataset used is the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).I've always wanted to try out a dataset with multiple classes and when I found this one I was delightfully surprised. I think that it is
an excellent dataset but lacks a bit on the amount of data that it provides.

### Useful resources
The residual layers used aren't the vanilla ones that Microsoft describes in their [paper](https://arxiv.org/abs/1512.03385) but instead
uses resnext blocks described by [this paper](https://arxiv.org/pdf/1611.05431.pdf) from the Facebook AI team.

### Deep Learning Topics Applied on this project
  - Dropout
  - Residual layers
  - Image Augmentation
  - Image normalization
  - Cross Entropy Loss Function
  - Gradient Descent Optimization (Done by the tensorflow API)

### What is this project for?
This project was mostly an experiment to see whether or not I can write a neural network that can classify images amongs a lot of different classes 
although due to my hardware's constraints and the relatively limited amound of data the results were a bit less than acceptable.

### Some Samples:
The model may not have exceptional accuracy but works pretty well and I would say that it's top 3 predictions aren't too bad.
Here is an image of my dog that the network was able to classify correctly as a maltez without being trained on this image, which to me seems pretty cool.
My wonderful dog:


<img src="https://raw.githubusercontent.com/PanagiotisPtr/Stanford-Dog-Breeds-CNN/master/samplesToTry/MyDog.jpg" width="250" height="250">

##### Disclaimer
This project isn't meant for production and is mostly an experimentation. I still am a novice Python programmer and as such the code design isn't all that great.
Also I do not own the dataset, please visit http://vision.stanford.edu/aditya86/ImageNetDogs/ for legal information and licesing of the dataset.

##### Foot note
I hope that you find this project informative and educational. Keep on learning!

Panagiotis Petridis
