import keras
import numpy as np
from keras.models import load_model
import scipy.misc as scim
model = load_model('Amaro.h5')
img1 = scim.imread('test1.jpg')
img2 = scim.imread('test2.jpg')
img3 = scim.imread('test3.jpg')
img4 = scim.imread('test4.jpg')
img5 = scim.imread('test5.jpg')

test_X1 = np.zeros((240,240,5))
test_Y1 = np.zeros((240,240,3))
test_X1[:,:,0:3] = img1
for i in range(240):
    for j in range(240):
        test_X1[i,j,3] = i
        test_X1[i,j,4] = j
test_X1 = test_X1.reshape(240*240, 5)
test_Y1 = test_Y1.reshape(240*240, 3)
test_X1 = test_X1.astype('float32')
test_X1[:, 0:3] /= 255
test_X1[:, 3:5] /= 240
newff1 = model.predict(test_X1)
newff1 = 255 * (newff1 - newff1.min())/(newff1.max() - newff1.min())
newff1 = newff1.reshape(240, 240, 3)
scim.imsave('test1_m_am.jpg',newff1)

test_X2 = np.zeros((240,240,5))
test_Y2 = np.zeros((240,240,3))
test_X2[:,:,0:3] = img2
for i in range(240):
    for j in range(240):
        test_X2[i,j,3] = i
        test_X2[i,j,4] = j
test_X2 = test_X2.reshape(240*240, 5)
test_Y2 = test_Y2.reshape(240*240, 3)
test_X2 = test_X2.astype('float32')
test_X2[:, 0:3] /= 255
test_X2[:, 3:5] /= 240
newff2 = model.predict(test_X2)
newff2 = 255 * (newff2 - newff2.min())/(newff2.max() - newff2.min())
newff2 = newff2.reshape(240, 240, 3)
scim.imsave('test2_m_am.jpg',newff2)

test_X3 = np.zeros((240,240,5))
test_Y3 = np.zeros((240,240,3))
test_X3[:,:,0:3] = img3
for i in range(240):
    for j in range(240):
        test_X3[i,j,3] = i
        test_X3[i,j,4] = j
test_X3 = test_X3.reshape(240*240, 5)
test_Y3 = test_Y3.reshape(240*240, 3)
test_X3 = test_X3.astype('float32')
test_X3[:, 0:3] /= 255
test_X3[:, 3:5] /= 240
newff3 = model.predict(test_X3)
newff3 = 255 * (newff3 - newff3.min())/(newff3.max() - newff3.min())
newff3 = newff3.reshape(240, 240, 3)
scim.imsave('test3_m_am.jpg',newff3)

test_X4 = np.zeros((240,240,5))
test_Y4 = np.zeros((240,240,3))
test_X4[:,:,0:3] = img4
for i in range(240):
    for j in range(240):
        test_X4[i,j,3] = i
        test_X4[i,j,4] = j
test_X4 = test_X4.reshape(240*240, 5)
test_Y4 = test_Y4.reshape(240*240, 3)
test_X4 = test_X4.astype('float32')
test_X4[:, 0:3] /= 255
test_X4[:, 3:5] /= 240
newff4 = model.predict(test_X4)
newff4 = 255 * (newff4 - newff4.min())/(newff4.max() - newff4.min())
newff4 = newff4.reshape(240, 240, 3)
scim.imsave('test4_m_am.jpg',newff4)

test_X5 = np.zeros((240,240,5))
test_Y5 = np.zeros((240,240,3))
test_X5[:,:,0:3] = img5
for i in range(240):
    for j in range(240):
        test_X5[i,j,3] = i
        test_X5[i,j,4] = j
test_X5 = test_X5.reshape(240*240, 5)
test_Y5 = test_Y5.reshape(240*240, 3)
test_X5 = test_X5.astype('float32')
test_X5[:, 0:3] /= 255
test_X5[:, 3:5] /= 240
newff5 = model.predict(test_X5)
newff5 = 255 * (newff5 - newff5.min())/(newff5.max() - newff5.min())
newff5 = newff5.reshape(240, 240, 3)
scim.imsave('test5_m_am.jpg',newff5)