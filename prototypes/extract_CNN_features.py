import cv2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import preprocess_input
import numpy as np
import os
import pickle


def VGG_all_features(X):
	f = []
	model = VGG19(weights='imagenet')
	for i in range(len(X)):
		img = cv2.imread(X[i])
		img = cv2.resize(img, dsize=(224, 224))
		img = img.astype('float32')
		x = np.expand_dims(img, axis=0)
		x = preprocess_input(x)
		features = model.predict(x)
		if i == 0:
			f = features
		else:
			f = np.vstack((f, features))
	return f

home = os.path.expanduser('~')
data_home = home + '/Documents/research/EP_project/data/matsc_dataset2/'
l1 = os.listdir(data_home)
X = np.empty((1, 500, 270, 3))
y = []
names = []
cnt = 0
for z in range(len(l1)):
	if l1[z][0] == '.':
		continue
	l = os.listdir(data_home + l1[z] + '/')
	y += [z] * len(l)
	cnt += 1
	for i in range(len(l)):
		names.append(data_home+l1[z]+'/'+l[i])

f = VGG_all_features(names)
#pickle.dump([f, y], open(home + '/Documents/research/EP_project/data/inception.pkl', 'wb'))
np.savez(open(home + '/Documents/research/EP_project/data/VGG_matsc_dataset2.npz', 'wb'), f)