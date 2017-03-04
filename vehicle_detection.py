import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.ndimage.measurements import label
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from util_functions import *
import time
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

class VehicleDetection:

    heatmaps = []
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

    X_scaler = joblib.load('./scaler_2_YCrCb_HOG-ALL_featlen_10224.pkl') 

    svc = joblib.load('./svc_2_YCrCb_HOG-ALL_featlen_10224.pkl') 
    
    scale = 1.5
    
    n_heatmaps = 1
    heat_threshold = 1
    
    def __init__(self, n_heatmaps=1, heat_threshold=1):
        self.heatmaps = []
        self.n_heatmaps=n_heatmaps
        self.heat_threshold = heat_threshold
        
    def add_heatmap(self, heatmap):
        self.heatmaps.append(heatmap)
        if len(self.heatmaps)>self.n_heatmaps:
            self.heatmaps=self.heatmaps[-self.n_heatmaps:]
            
    def get_av_heatmap(self):
        return np.mean(self.heatmaps,axis=0)
    
    def get_sum_heatmap(self):
        return np.sum(self.heatmaps,axis=0)
    
    def process_image(self, image):

        found_heat = find_cars_heat(image, scale=self.scale, 
                                    ystart=self.y_start_stop[0], ystop=self.y_start_stop[1], 
                                    svc=self.svc, X_scaler=self.X_scaler, 
                                    orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                    cell_per_block=self.cell_per_block, 
                                    spatial_size=self.spatial_size, hist_bins=self.hist_bins)
        self.add_heatmap(found_heat)
        heat = self.get_sum_heatmap() 
    
        heat = apply_threshold(heat,self.heat_threshold)

        # Visualize the heatmap when displaying    
        heat_map = np.clip(heat, 0, 255)
        labels = label(heat_map)

        # Draw bounding boxes on a copy of the image
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        return draw_img, heat_map
    
    def process_image_2(self,orig_image):
            
        image = orig_image.copy().astype(np.float32)/255

        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=self.y_start_stop, 
                                xy_window=(96, 96), xy_overlap=(0.8, 0.8))
        
        box_list = search_windows(image, windows, self.svc, self.X_scaler, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        # Add heat to each box in box list
        heat_sum = add_heat(heat,box_list)
        
        self.add_heatmap(heat_sum)
        heat = self.get_sum_heatmap() 

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,self.heat_threshold)

        # Visualize the heatmap when displaying    
        heat_map = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heat_map)
        draw_img = draw_labeled_bboxes(np.copy(orig_image), labels)
        
        return draw_img, heat_map
    
    def process_1(self, img):
        draw_img,_ = self.process_image(img)
        return draw_img
    
    def process_2(self, img):
        draw_img,_ = self.process_image_2(img)
        return draw_img

