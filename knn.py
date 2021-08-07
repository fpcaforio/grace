#import
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import pickle
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D
from keras.models import Model
from tensorflow.keras import backend as K
#tf.compat.v1.disable_eager_execution()
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D
from sklearn.metrics import confusion_matrix
import re
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.image as mpimg
import keras.backend as K
from keras.preprocessing import image
from keras.layers import Dense, Input, Layer, InputSpec, Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, normalized_mutual_info_score, adjusted_rand_score#, confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
from keras import callbacks
from sklearn.model_selection import train_test_split
import time


f_myfile = open('/content/drive/MyDrive/ESPERIMENTI/NLS-KDD/gradcam/pickle/testing/XTrain_heatmap_12x12.pickle', 'rb')
XTrain = pickle.load(f_myfile)
f_myfile.close()

f_myfile = open('/content/drive/MyDrive/ESPERIMENTI/NLS-KDD/MAGNETO/12x12/Ytrain.pickle', 'rb')
YTrain = pickle.load(f_myfile)
f_myfile.close()

f_myfile = open('/content/drive/MyDrive/ESPERIMENTI/NLS-KDD/gradcam/pickle/testing/XTest_heatmap_12x12.pickle', 'rb')
XTest = pickle.load(f_myfile)
f_myfile.close()

f_myfile = open('/content/drive/MyDrive/ESPERIMENTI/NLS-KDD/MAGNETO/12x12/Ytest.pickle', 'rb')
YTest = pickle.load(f_myfile)
f_myfile.close()

XTrain = np.array(XTrain)
XTrain = np.uint8(255 * XTrain)
XTrain = XTrain / 255

XTest = np.array(XTest)
XTest = np.uint8(255 * XTest)
XTest = XTest / 255

#plt.imshow(XTrain[0])

XTrain = XTrain.reshape(-1, (12*12))
XTest = XTest.reshape(-1, (12*12))

print(XTrain.shape)
print(XTest.shape)