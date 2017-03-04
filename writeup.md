#Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"

[i1a1]: ./vehicle_test/7.png
[i1a2]: ./vehicle_test/13.png
[i1a3]: ./vehicle_test/25.png
[i1a4]: ./vehicle_test/19.png
[i1a5]: ./vehicle_test/31.png
[i1b1]: ./non_vehicle_test/extra40.png
[i1b2]: ./non_vehicle_test/extra67.png
[i1b3]: ./non_vehicle_test/extra39.png
[i1b4]: ./non_vehicle_test/extra38.png
[i1b5]: ./non_vehicle_test/extra35.png
[image2a]: ./output_images/car_img_HOG_bin_hist_4.png
[image2b]: ./output_images/noncar_img_HOG_bin_hist_2.png
[image3]: ./output_images/slide_windows.png
[image4a]: ./output_images/searchwindows_1.png
[image4b]: ./output_images/searchwindows_3.png
[image4c]: ./output_images/searchwindows_6.png
[image5a]: ./output_images/bbox_heatmap_1.png
[image5b]: ./output_images/bbox_heatmap_2.png
[image5c]: ./output_images/bbox_heatmap_3.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]:  https://youtu.be/zUwqcW9DGnY

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Overview of the project files

To this project belong the following files:

- `writeup.md` The writeup (this document).
- `util_functions.py` A Python file containing many of the functions described in the course.
- `train_scaler_svc.py` A Python file with the code to train the Scaler and the linear SVC.
- `vehicle_detection.py` A Python file with the `VehicleDetection` class that brings all the vehicle detection functionality together.
- `P5-notebook.ipynb` The Jupyter Notebook where various steps of this project have been tested and run.
- `P5-notebook.html` HTML version of the Jupyter Notebook.
- `/output_images` The directory containing the resulting images.
- Two pickle files with the saved Scaler and SVC.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file called `train_scaler_svc.py`. 

I started by reading in all the `vehicle` and `non-vehicle` images in lines #16 to #32.  Here is an example of five of each of the `vehicle` and `non-vehicle` classes:

|              Vehicle images              |            non-Vehicle images            |
| :--------------------------------------: | :--------------------------------------: |
| ![][i1a1]![][i1a2]![][i1a3]![][i1a4]![][i1a5] | ![][i1b1]![][i1b2]![][i1b3]![][i1b4]![][i1b5] |


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the output of the 3 different features looks like.

Here are a few examples using HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, a Color histogram with 32 bins in the `YCrCb` color space and  a plot of the spatially binned features with size 32x32 also in  `YCrCb` color space:

![alt text][image2a]

![alt text][image2b]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and when training the Linear SVM I found that the following parameters gave the highest probability of correct classification of vehicles and non-vehicles (> 99.2%):

|          Parameter          |    Value(s)    |
| :-------------------------: | :------------: |
|         Color space         |     YCrCb      |
|     #HOG  orientations      |       12       |
|    #HOG pixels per cell     |      8px       |
|    #HOG cells per block     |       2        |
|        HOG channels         |      all       |
| #Spatial binning dimensions |  32px x 32px   |
|       #Histogram bins       |       32       |
|      X start position       |      0px       |
|       X stop position       | width of image |
|      Y start position       |     400px      |
|       Y stop position       |     600px      |

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier can be found in the file `train_scaler_svm.py`.

First the HOG, histogram features and spatial features were extracted from the vehicle and non-vehicle datasets using the function `extract_features()` which is implemented in the file `util_functions.py`. The function is called at lines #47 and #53 for the two datasets respectively. The extracted features are combined into one dataset.

Then, a scaler is trained in order to normalize the data features. This scaler is the `sklearn.preprocessing.StandardScaler`. It is saved to a pickle file at line #93 for use in the detection functions.

Once the data is scaled, it is split in to a training and test/validation set at line #73. The test/validation set is set to be 20% of the dataset.

Finally a linear SVM is created using `sklearn.svm.LinearSVC` (in line #79) and trained using the data (in line #83). The SVM is run with the default arguments provided by the `sklearn` library. The SVM is also saved to a pickle file for use with the detection functions.

The test/validation accuracy was > 99.2%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented with the functions `slide_window()`  located in the file `util_function.py` at lines #195 to #234 and `search_windows()` at lines #280 to #308.

These functions are called from the `process_image_2()` method in the `VehicleDetection` class located in the file `vehicle_detection.py` at line #82 and #85 respectively.

I tried various window searches of window size 64x64px, 96x96px, 128x128px and 160x160px and each at various overlaps ranging from 0.5x0.5 to 0.8x0.8. In addition I tried starting at various different Y positions from 300 to 400 and ending at 600 to 700. In order to have a very complete search result (including finding cars both close and very far away) I tried to combine various window sizes. Yet in the end I settled on the one window size 96x96 with overlap 0.8x0.8 for speed reasons. The search windows start at 400px and end at 600px. The closest cars are still found this way.

An example of the complete search area can be seen here:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Essentially the performance was optimized by choosing the best values for the parameters as mentioned above.

Here are some example images:

![alt text][image4a]

![alt text][image4b]

![alt text][image4c]

Note that in some images there are some false positives. These will be filtered out using a thresholding function.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The method used to process the videoframes is `process_2()` and is part of the `VehicleDetection` class. This method is actually a wrapper for the method called `process_image_2()` in which the actual implementation resides. It can be found in the lines #76 to #108 in the file `vehicle_detection.py`.

I recorded the positions of positive detections using the `slide_window()` and `search_windows()` functions in each frame of the video.  From the positive detections I created a heat map. This heat map is appended to a list which contains some of previous heat maps. In total this list is to contain the last 25 heat maps - the current one included.

The sum of the last 25 heat maps is then thresholded at value 15. The result is a new heat map on which  I then use `scipy.ndimage.measurements.label()` to identify individual blobs in the heat map.  I then assume each blob corresponds to a vehicle.  I construct bounding boxes to cover the area of each blob detected. 

Here are example results showing the final bounding boxes and the heat maps from three test frames from the video (do note that in this test case no heat maps are summed and the threshold is at value 1):

![alt text][image5a]

![alt text][image5b]

![alt text][image5c]

Note that there is also a method called `process_image()` in the `VehicleDetection` class. This method uses a different function for creating heatmaps. I tried this during experimentation, but I felt that although it is quicker that the above method, it doesn't yield as good results.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It is possible to try an endless combination of window sizes and even using different techniques for finding the cars, but in the end the idea is to have a robust detection system which works fast (enough).

The speed could probably be improved by doing partial searches on the outer edges of the video frames. Partial in the sense that if we have a video of 60 fps then we could easily search 1/6th of each frame, such that the full frame would be searched at 10 fps. In addition, we also know were new cars will appear. The focus for search could be narrowed down to those places.

Once a car is found, we could do a small search around this car every frame. Simply to update the bounding box.

I wonder how the detection will work in different lighting conditions. Possibly the dataset should be augmented with different brightness values and perhaps even changes in hue.

