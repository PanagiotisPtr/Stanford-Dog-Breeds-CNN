'''
MIT License

Copyright (c) 2017 Panagiotis Petridis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
I don't have the rights to the dataset.
The Dataset is the 'Stanford Dogs Dataset'
You can find more information and download it here: http://vision.stanford.edu/aditya86/ImageNetDogs/
'''

print('Loading dependecies...')
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import tflearn

print('Dependecies loaded!')

np.set_printoptions(suppress=True)
sns.set_style('whitegrid')

IMG_WIDTH, IMG_HEIGHT = 32, 32
NB_EPOCHS = 100
BATCH_SIZE = 32
n_classes = 120

# When converting images to float arrays there is a problem with the colors.
# This function fixes that issue.
def im2double(img):
    info = np.iinfo(img.dtype)
    return img.astype(np.float32)/info.max

# Reading images as numpy 3D arrays
def img_to_array(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
    return im2double(img)

# one-hot-encodes array with labels. i.e [[1],[0],[3]] -> [[0,1,0,0], [1,0,0,0], [0,0,0,1]]
def to_onehot(y, n_classes):
    tmp = np.zeros((y.shape[0], n_classes))
    tmp[np.arange(y.shape[0]), y.astype(int)] = 1
    return np.array(tmp, dtype=np.float32)

# Loading the images.
image_paths = glob('Images/*')
breeds = [s[17:] for s in image_paths] # By default I am including an empty folder tree inside the github project 
									   # so that this part of the code can work and you can make predictions on a
									   # dog even if you haven't downloaded the dataset. Make sure you download the
									   # dataset and replace the Images folder with the one that is in the data zip
									   # file.
X = []
y = []
for idx, path in enumerate(image_paths):
    for image in glob(path + "/*"):
        X.append(img_to_array(image))
        y.append(idx)
    print(idx, 'Loaded', breeds[idx])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print('Data loaded!')

y = to_onehot(y, n_classes)

# Spliting the data to train and test sets. Also the fixed random state ensures reproducability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89741) 

# Because I am using residual blocks for layers that parameter n basically controls the number of layers
# for exampels if n=5 then the network consists of 32 layers
# n=9 is 56 layers
# n=18 is I mean 110 layers
n = 5

# I need to first preprocess the data.
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Because the model is quite big, even with n=5, data augmentation helps generalization and, as such, reduces overfitting
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.0)

# Building the tflearn network
net = tflearn.input_data(shape=[None, IMG_WIDTH, IMG_HEIGHT, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 64, 32)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
net = tflearn.fully_connected(net, n_classes, activation='softmax') # Multi class classification basically with softmax

# Training using the Momentum optimizer
opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=opt,
                         loss='categorical_crossentropy')
model = tflearn.DNN(net, checkpoint_path='model_resnext_dog_breeds',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

# If the file of the previus best checkpoint exist, it might as well be loaded.
if os.path.isfile('{}.meta'.format('Best-model')):
    model.load('Best-model')
    print('model loaded!')
else:
    print('Error loading model!')

# Fitting the model to the training data.
# Feel free to uncomment the lines below to train the network on your own. Or leave them commented and make predictions.
'''
model.fit(X_train, y_train, n_epoch=NB_EPOCHS, validation_set=(X_test, y_test),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=BATCH_SIZE, shuffle=True,
          run_id='resnet_breeds')

model.save("Best-model")
'''

# Drop an image onto the terminal and the model shall classify it. ( this was only tested on Ubuntu 16.04 Linux so I don't know if it 
# works on Windows and Mac, although it should work).
test_file = input("Enter image path to classify: ")
while test_file!='exit':
    test_file = test_file.replace("'", "")
    test_file = test_file.replace(" ", "")
    sample = img_to_array(test_file)
    pred = model.predict([sample])[0]
    ans = pred.argsort()[-3:][::-1]
    print("That's most likely a", breeds[ans[0]])
    print("But it might also be a", breeds[ans[1]], "or a", breeds[ans[2]])
    test_file = input("Enter image path to classify: ")
