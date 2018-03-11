import scipy.misc as scim
import matplotlib.pyplot as plt
import glob

im1 = scim.imread('sample_image.jpg')
im2 = scim.imread('sample_image_eb.jpg')
im1.shape

im1_R = im1[:,:,0]
im1_G = im1[:,:,1]
im1_B = im1[:,:,2]
im2_R = im2[:,:,0]
im2_G = im2[:,:,1]
im2_B = im2[:,:,2]

x1 = im1_R.flatten()
x2 = im1_G.flatten()
x3 = im1_B.flatten()
y1 = im2_R.flatten()
y2 = im2_G.flatten()
y3 = im2_B.flatten()
#plt.scatter(x1, y1)
#plt.scatter(x2, y2)
#plt.scatter(x3, y3)

Earlybird = {}
Original = {}
num_images = 1280

path = '/Users/sgkamal/Documents/Documents/Courses/Experimental/Earlybird/*.jpg'
for filename in glob.glob(path)[:num_images]:
    img = scim.imread(filename)
    Earlybird[filename] = img

path = '/Users/sgkamal/Documents/Documents/Courses/Experimental/Original/*.jpg'
for filename in glob.glob(path)[:num_images]:
    img = scim.imread(filename)
    Original[filename] = img

x = []
y = []
for i in Original.keys():
    x.append(Original[i][0,0,0])
for j in Earlybird.keys():
    y.append(Earlybird[j][0,0,0])

plt.scatter(x,y)
plt.show()