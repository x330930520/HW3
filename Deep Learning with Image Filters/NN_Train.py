#This is the main code where we train the neural network

import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,lib.cnmem=1,nvcc.fastmath=False'

import sys


from PIL import Image
import glob
import numpy as np
import scipy.misc as scim

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Convolution2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import model_from_json

num_images = 1280

data_X = np.zeros((1280,240,240,5))

data_Y = np.zeros((1280,240,240,3))



#Fetching the transform set

path = 'small_orig/*.jpg'
ctr = 0


for filename in glob.glob(path)[:num_images]: 
    img = scim.imread(filename)
    data_Y[ctr, :,:,:] = scim.imresize(img, (240,240,3))
    ctr = ctr + 1
	
	
#Fetching the input set

path = 'small/*.jpg'
ctr = 0


for filename in glob.glob(path)[:num_images]: 
    img = scim.imread(filename)
    data_X[ctr, :,:,0:3] = scim.imresize(img, (240,240,3))
    ctr = ctr + 1
	
for i in range(240):
    for j in range(240):
        data_X[:,i,j,3] = i
        data_X[:,i,j,4] = j

		
#Resizing the input and transform set

data_X = data_X.reshape(1280*240*240, 5)

data_Y = data_Y.reshape(1280*240*240, 3)



num_pixel = data_X.shape[0]




#Split the data into test and train

X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2)





#Flattening the Input (Train and Test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train[:,0:3] /= 255
X_test[:,0:3] /= 255
X_train[:,3:5] /= 240
X_test[:,3:5] /= 240



#Flattening the Transformed image (Train and Test)

Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
Y_train /= 255
Y_test /= 255



#Creating model

model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
#model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation = 'relu'))



# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')



batch_size = 1024
epochs = 30

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()



model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[history])



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score)
#print('Test accuracy:', score[1])


#Testing the model on one image

testimg = scim.imread('11.png')

testimg = scim.imresize(testimg, (240,240,3))

test_X = np.zeros((240,240,5))

test_Y = np.zeros((240,240,3))

test_X[:,:,0:3] = testimg

for i in range(240):
    for j in range(240):
        test_X[i,j,3] = i
        test_X[i,j,4] = j

test_X = test_X.reshape(240*240, 5)

test_Y = test_Y.reshape(240*240, 3)

test_X = test_X.astype('float32')

test_X[:,0:3] /= 255

test_X[:,3:5] /= 240

newff = model.predict(test_X)

newff = newff.reshape(240,240,3) * 255

newff = newff.astype(int)

scim.imsave("Predicted.jpg", newff)



#Saving the model	

model.save('model_run_1.h5')


# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")

print("Saved model to disk")


#Plotting and saving the error function

plt.ioff()

fig = plt.figure()

plt.plot(range(1, 3), history.acc)

plt.savefig('Error.png')

plt.close()