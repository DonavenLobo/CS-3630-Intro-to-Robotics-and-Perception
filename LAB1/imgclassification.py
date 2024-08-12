#!/usr/bin/env python

##############
#### Your name: Donaven Lobo
##############

import numpy as np
import re, math
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
import ransac_score

class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]
        
        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data
        feature_data = []

        for img in data:
            img = color.rgb2gray(img)
            img = filters.gaussian(exposure.equalize_hist(img))
            features = feature.hog(img, orientations=8, pixels_per_cell=(26, 26), cells_per_block=(6, 6), block_norm='L2-Hys')
            feature_data.append(features)

        # Please do not modify the return type below
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above
        
        # train model and save the trained model to self.classifier

        model = svm.SVC(kernel='linear')
        self.classifier = model.fit(train_data, train_labels)

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels

        predicted_labels = self.classifier.predict(data)
        # Please do not modify the return type below
        return predicted_labels

    def line_fitting(self, data):
        # Please do not modify the header

        # fit a line the to arena wall using RANSAC
        # return two lists containing slopes and y intercepts of the line

        # Create arrays to hold slopes and y intercepts
        slope = []
        intercept = []
        for img in data:
            # Need to take out the lines:
            gray_img = color.rgb2gray(img) # Grayscale the image
            smoothed_img = filters.gaussian(gray_img, sigma=3) # Smoothing the image
            edges = feature.canny(smoothed_img) # Finding the edges using canny edge detection
            y_co, x_co = np.where(edges) #finding the x and y coordinates of the edges
            # x_co contains the x-coordinates (columns)
            # y_cocontains the y-coordinates (rows)

            # Now conduct the RANSAC:
            # Created a helper function
            img_slope, img_intercept = self.RANSAC_helper(x_co, y_co, num_samples = 5, repeats = 1000, acceptable_dist = 1)
            slope.append(img_slope) # Add image slope to slope list
            intercept.append(img_intercept) # Add image y intercept to intercept list


        # Please do not modify the return type below
        return slope, intercept
    

    # Helper Function
    def RANSAC_helper(self, x_co, y_co, num_samples, repeats, acceptable_dist):
        # Initialize best line parameters
        best_score = 0
        best_slope = None
        best_intercept = None

        for _ in range(repeats):
            # Choose a random set of points
            rand_ind = np.random.choice(len(x_co), size=num_samples, replace=False)
            x_subset = x_co[rand_ind]
            y_subset = y_co[rand_ind]

            # Fit a line to the selected points using linear regression
            line = np.polyfit(x_subset, y_subset, 1)
            slope = line[0]
            intercept = line[1]

            # Calculate number of inliers
            distances = np.abs(slope * x_co - y_co + intercept) / np.sqrt(slope**2 + 1) # Calculate the perpendicular distance from all points to the line
            num_inliers = np.sum(distances <= acceptable_dist) # Count inliers (points within acceptable_dist from the line)

            # Update best line if this iteration had more inliers
            if num_inliers > best_score:
                best_score = num_inliers
                best_slope = slope
                best_intercept = intercept

        return best_slope, best_intercept

def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')
    
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))

    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")

if __name__ == "__main__":
    main()