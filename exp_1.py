import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from util_functions import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

 
    
# # Read in car and non-car images
print("Reading in images...")
car_images = glob.glob('./vehicles/GTI_Far/*.png') + glob.glob('./vehicles/GTI_Left/*.png') + glob.glob('./vehicles/GTI_MiddleClose/*.png')+ glob.glob('./vehicles/GTI_Right/*.png') + glob.glob('./vehicles/KITTI_extracted/*.png')
print("# car images: " + str(len(car_images)))

non_car_images = glob.glob('./non-vehicles/GTI/*.png')+glob.glob('./non-vehicles/Extras/*.png')
non_car_images = non_car_images[:len(car_images)]
print("# non-car images: " + str(len(non_car_images)))
    
cars = []
notcars = []

for car_image in car_images:
    cars.append(car_image)

for non_car_image in non_car_images:
    notcars.append(non_car_image)

# These variables should be the same as those used by the detection algorthm
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 600] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
print(X.shape)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

print("Saving model...")
path_svc = './svc_2_'+ color_space +'_HOG-'+ str(hog_channel) + '_featlen_' + str(len(X_train[0])) +'.pkl'
path_scaler = './scaler_2_'+ color_space +'_HOG-'+ str(hog_channel) + '_featlen_' + str(len(X_train[0])) +'.pkl'
# joblib.dump(svc, path_svc) 
# joblib.dump(X_scaler, path_scaler) 