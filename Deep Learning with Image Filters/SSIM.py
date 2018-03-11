import matplotlib.pyplot as plt
import numpy as np
import pylab
from skimage import measure



for i in range(1,6):
    f1 = 'true_' + str(i) + '.jpg'
    f2 = 'predict_' + str(i) + '.jpeg'
    a = plt.imread(f1)
    b = plt.imread(f2)
    ssim = measure.compare_ssim(a,b, multichannel = True)
    print (ssim)

