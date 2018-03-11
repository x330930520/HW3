import scipy.misc as scim
import glob
from keras.models import load_model
import numpy as np

model = load_model('earlybird_run_1.h5')

path = 'orig/*.jpeg'


for filename in glob.glob(path): 
    testimg = scim.imread(filename)

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
    
    fname = 'test_' + filename

    scim.imsave(fname, newff)