#This is the code to generate the synthetic data set

import os
import numpy as np
import scipy.misc as scim

#os.chdir('/Users/sgkamal/Documents/Documents/Courses/Experimental')

# an_img = np.zeros((240,240,3))
# for channel in range(3):
# 	for i in range(0,60):
# 		for j in range(240):
# 			an_img[i,j,channel] = np.random.uniform(0,0.25)
# 	for i in range(60,120):
# 		for j in range(240):
# 			an_img[i,j,channel] = np.random.uniform(0.25,0.5)
# 	for i in range(120, 180):
# 		for j in range(240):
# 			an_img[i,j,channel] = np.random.uniform(0.5, 0.75)
# 	for i in range(180, 240):
# 		for j in range(240):
# 			an_img[i,j,channel] = np.random.uniform(0.75, 1)



for im_no in range(1280):
    an_img = np.zeros((240* 240* 3))
    for i in range(len(an_img)):
        an_img[i] = np.random.uniform(0,1)
    an_img = an_img.reshape((240, 240, 3))
    an_img = an_img * 255
    an_img = an_img.astype(int)
    im_name = str(im_no+1) + '.jpg'
    scim.imsave(im_name, an_img)


