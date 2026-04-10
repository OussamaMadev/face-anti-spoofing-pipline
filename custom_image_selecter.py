"""
.. module:: custom_image_selecter
    :platform: OS X
    :synopsis: This module has functions related to key frame extraction from Katna.
"""

from __future__ import print_function

import os
from glob import glob
import tempfile
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_float
import itertools  
import time
from multiprocessing import Pool
import Katna.image_selector

class ImageSelectionConfig:
    def __init__(self,min_brightness_value=10.0, max_brightness_value=90.0, brightness_step=2.0, min_entropy_value=1.0, max_entropy_value=10.0, entropy_step=0.5):
        """Configuration class for image selection parameters, used to store the parameters for selection of best frames from input list of frames
        
        args:
            - min_brightness_value: minimum brightness value for selection of best frames, float value between 0.0 to 100.0
            - max_brightness_value: maximum brightness value for selection of best frames, float value between min_brightness_value to 100.0
            - brightness_step: step value for iterating over brightness values, float value between 0.0 to max_brightness_value-min_brightness_value
            
            - min_entropy_value: minimum entropy value for selection of best frames, float value between 0.0 to 10.0
            - max_entropy_value: maximum entropy value for selection of best frames, float value between min_entropy_value to 10.0
            - entropy_step: step value for iterating over entropy values, float value between 0.0 to max_entropy_value-min_entropy_value
    
        """

        # Setting for optimum Brightness values
        self.min_brightness_value = min_brightness_value
        self.max_brightness_value = max_brightness_value
        self.brightness_step = brightness_step
        
        # Setting for optimum Contrast/Entropy values
        self.min_entropy_value = min_entropy_value
        self.max_entropy_value = max_entropy_value
        self.entropy_step = entropy_step

    

class CustomImageSelector(object):
    """Class for selection of best top N images from input list of images, Currently following selection method are implemented:
    brightness filtering, contrast/entropy filtering, clustering of frames and variance of laplacian for non blurred images 
    selection

    :param object: base class inheritance
    :type object: class:`Object`
    """

    def __init__(self, n_processes=1, imageSelectionConfig=None):
        """
        Constructor for CustomImageSelector class, initializes the parameters for selection of best frames from input list of frames
        """
        # Setting number of processes for Multiprocessing Pool Object
        self.n_processes = n_processes
        self.image_selection_config = ImageSelectionConfig() if imageSelectionConfig is None else imageSelectionConfig

        # Setting for optimum Brightness values
        self.min_brightness_value = self.image_selection_config.min_brightness_value
        self.max_brightness_value = self.image_selection_config.max_brightness_value
        self.brightness_step = self.image_selection_config.brightness_step

        # Setting for optimum Contrast/Entropy values
        self.min_entropy_value = self.image_selection_config.min_entropy_value
        self.max_entropy_value = self.image_selection_config.max_entropy_value
        self.entropy_step = self.image_selection_config.entropy_step 

    def __get_brightness_score__(self, image):
        """Internal function to compute the brightness of input image , returns brightness score between 0 to 100.0 , 

        :param object: base class inheritance
        :type object: class:`Object`
        :param image: input image
        :type image: Opencv Numpy Image   
        :return: result of Brightness measurment 
        :rtype: float value between 0.0 to 100.0    
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        sum = np.sum(v, dtype=np.float32)
        num_of_pixels = v.shape[0] * v.shape[1]
        brightness_score = (sum * 100.0) / (num_of_pixels * 255.0)
        return brightness_score

    def __get_entropy_score__(self, image):
        """Internal function to compute the entropy/contrast of input image , returns entropy score between 0 to 10 , 
 
        :param object: base class inheritance
        :type object: class:`Object`
        :param image: input image
        :type image: Opencv Numpy Image
        :return: result of Entropy measurment
        :rtype: float value between 0.0 to 10.0
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entr_img = entropy(gray, disk(5))
        all_sum = np.sum(entr_img)
        num_of_pixels = entr_img.shape[0] * entr_img.shape[1]
        entropy_score = (all_sum) / (num_of_pixels)

        return entropy_score

    def __variance_of_laplacian__(self, image):
        """Internal function to compute the laplacian of the image and then return the focus
        measure, which is simply the variance of the laplacian,
 
        :param object: base class inheritance
        :type object: class:`Object`
        :param image: input image
        :type image: Opencv Numpy Image   
        :return: result of cv2.Laplacian
        :rtype: opencv image of type CV_64F    
        """

        return cv2.Laplacian(image, cv2.CV_64F).var()

    def __filter_optimum_brightness_and_contrast_images__(self, input_img_files):
        """ 
        Internal function for selection of given input images with following parameters:
        optimum brightness and contrast range.
        
        MODIFIED: Now returns a list of INDICES instead of the passing images.
 
        :param input_img_files: list of input image data (numpy arrays)
        :return: Returns list of indices for filtered images 
        :rtype: python list of int 
        """

        n_files = len(input_img_files)

        # 1. Calculating the brightness and entropy score using multiprocessing
        pool_obj = Pool(processes=self.n_processes)

        with pool_obj:
            brightness_score = np.array(
                pool_obj.map(self.__get_brightness_score__, input_img_files)
            )

            entropy_score = np.array(
                pool_obj.map(self.__get_entropy_score__, input_img_files)
            )

        # 2. Create boolean masks for brightness and contrast
        brightness_ok = np.logical_and(
            brightness_score > self.min_brightness_value,
            brightness_score < self.max_brightness_value
        )
        
        contrast_ok = np.logical_and(
            entropy_score > self.min_entropy_value,
            entropy_score < self.max_entropy_value
        )

        # 3. Return only the INDICES where both conditions are True
        return [
            i for i in range(n_files)
            if brightness_ok[i] and contrast_ok[i]
        ]
    

    def __prepare_cluster_sets__(self, files):
        """ Internal function for clustering input image files, returns array of indexs of each input file
        (which determines which cluster a given file belongs)
 
        :param object: base class inheritance
        :type object: class:`Object`
        :param files: list of input image files 
        :type files: python list of opencv numpy images
        :return: Returns array containing index for each file for cluster belongingness 
        :rtype: np.array   
        """

        all_hists = []

        # Calculating the histograms for each image and adding them into **all_hists** list
        for img_file in files:
            img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist = hist.reshape((256))
            all_hists.append(hist)

        # Kmeans clustering on the histograms
        kmeans = KMeans(n_clusters=self.nb_clusters, random_state=0).fit(all_hists)
        labels = kmeans.labels_

        # Identifying the label for each image in the cluster and tagging them
        files_clusters_index_array = []
        for i in np.arange(self.nb_clusters):
            index_array = np.where(labels == i)
            files_clusters_index_array.append(index_array)

        files_clusters_index_array = np.array(files_clusters_index_array, dtype=object)
        return files_clusters_index_array

    def __get_laplacian_scores(self, files, n_images):
        """Function to iteratee over each image in the cluster and calculates the laplacian/blurryness 
           score and adds the score to a list

        :param files: list of input filenames 
        :type files: python list of string
        :param n_images: number of images in the given cluster
        :type n_images: int
        :return: Returns list of laplacian scores for each image in the given cluster
        :rtype: python list 
        """

        variance_laplacians = []
        # Iterate over all images in image list
        for image_i in n_images:
            img_file = files[n_images[image_i]]
            img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

            # Calculating the blurryness of image
            variance_laplacian = self.__variance_of_laplacian__(img)
            variance_laplacians.append(variance_laplacian)

        return variance_laplacians

    def __get_best_images_index_from_each_cluster__(
        self, files, files_clusters_index_array
    ):
        """ Internal function returns index of one best image from each cluster

        :param object: base class inheritance
        :type object: class:`Object`
        :param files: list of input filenames 
        :type files: python list of string
        :param files_clusters_index_array: Input is array containing index for each file for cluster belongingness 
        :type: np.array   
        :return: Returns list of filtered files which are best candidate from each cluster
        :rtype: python list 
        """

        filtered_items = []

        # Iterating over every image in each cluster to find the best images from every cluster
        clusters = np.arange(len(files_clusters_index_array))
        for cluster_i in clusters:
            curr_row = files_clusters_index_array[cluster_i][0]
            # kp_lengths = []
            n_images = np.arange(len(curr_row))
            variance_laplacians = self.__get_laplacian_scores(files, n_images)

            # Selecting image with low burr(high laplacian) score
            selected_frame_of_current_cluster = curr_row[np.argmax(variance_laplacians)]
            filtered_items.append(selected_frame_of_current_cluster)

        return filtered_items

    # def __getstate__(self):
    #     """Function to get the state of initialized class object and remove the pool object from it
    #     """
    #     self_dict = self.__dict__.copy()
    #     del self_dict["pool_obj"]
    #     return self_dict

    # def __setstate__(self, state):
    #     """Function to update the state of initialized class object woth the pool object
    #     """
    #     self.__dict__.update(state)

    def select_best_frames(self, input_key_frames, number_of_frames):
        """[summary] Public function for Image selector class: takes list of key-frames images and number of required
        frames as input, returns list of filtered keyframes

        :param object: base class inheritance
        :type object: class:`Object`
        :param input_key_frames: list of input keyframes in list of opencv image format 
        :type input_key_frames: python list opencv images
        :param number_of_frames: Required number of images 
        :type: int   
        :return: Returns list of filtered image files 
        :rtype: python list of images
        """

        self.nb_clusters = number_of_frames

        filtered_key_frames = []
        filtered_images_list = []
        # Repeat until number of frames 
        min_brightness_values = np.arange(self.image_selection_config.min_brightness_value, -0.01, -self.brightness_step)
        max_brightness_values = np.arange(self.image_selection_config.max_brightness_value, 100.01, self.brightness_step)
        min_entropy_values = np.arange(self.image_selection_config.min_entropy_value, -0.01, -self.entropy_step)
        max_entropy_values = np.arange(self.image_selection_config.max_entropy_value, 10.01, self.entropy_step)

        for (min_brightness_value, max_brightness_value, min_entropy_value, max_entropy_value) in itertools.zip_longest(min_brightness_values, max_brightness_values, min_entropy_values, max_entropy_values): 
            if min_brightness_value is None:
                min_brightness_value = 0.0
            if max_brightness_value is None:
                max_brightness_value = 100.0
            if min_entropy_value is None:
                min_entropy_value = 0.0
            if max_entropy_value is None:
                max_entropy_value = 10.0
            self.min_brightness_value = min_brightness_value
            self.max_brightness_value = max_brightness_value
            self.min_entropy_value = min_entropy_value
            self.max_entropy_value = max_entropy_value
            filtered_key_frames_indeces = self.__filter_optimum_brightness_and_contrast_images__(
                input_key_frames, 
            )
            filtered_key_frames = [input_key_frames[i] for i in filtered_key_frames_indeces]

            if len(filtered_key_frames) >= number_of_frames:
                break

        # Selecting the best images from each cluster by first preparing the clusters on basis of histograms 
        # and then selecting the best images from every cluster
        if len(filtered_key_frames) >= self.nb_clusters:
                
                files_clusters_index_array = self.__prepare_cluster_sets__(filtered_key_frames)
                
                selected_images_index = self.__get_best_images_index_from_each_cluster__(
                    filtered_key_frames, files_clusters_index_array
                    )
                

                for index in selected_images_index:
                    img = filtered_key_frames[index]
                    filtered_images_list.append(img)
            
        else:
                # if number of required files are less than requested key-frames return all the files
                for img in filtered_key_frames:
                    filtered_images_list.append(img)
        
        return filtered_images_list
    
    def select_best_frames_paths(self, input_key_frame_paths, number_of_frames, file_name_len = 13):
        """
        Modified Katna selector that takes a list of file paths and returns 
        the filtered list of file paths.
        
        :param input_key_frame_paths: list of paths to images
        :param number_of_frames: Number of diverse frames to select
        :return: List of strings (selected image paths)
        """
        self.nb_clusters = number_of_frames
        
        # 1. LOAD DATA & MAP PATHS TO IMAGES
        # We use a list of dicts so paths and pixels stay perfectly aligned
        data_pool = []
        for path in input_key_frame_paths:
            img = cv2.imread(path)
            if img is not None:
                data_pool.append({"path": path, "img": img})
        

        if not data_pool:
            return []

        filtered_data_pool = []
        
        # 2. ITERATIVE QUALITY FILTERING
        # Generate range of thresholds from         
        min_brightness_values = np.arange(self.min_brightness_value, -0.01, -self.brightness_step)
        max_brightness_values = np.arange(self.max_brightness_value, 100.01, self.brightness_step)
        min_entropy_values = np.arange(self.min_entropy_value, -0.01, -self.entropy_step)
        max_entropy_values = np.arange(self.max_entropy_value, 10.01, self.entropy_step)

        # Zip thresholds and try strict to relaxed constraints
        for (min_b, max_b, min_e, max_e) in itertools.zip_longest(
            min_brightness_values, max_brightness_values, min_entropy_values, max_entropy_values
        ):
            # Fill None values from zip_longest
            self.min_brightness_value = min_b if min_b is not None else 0.0
            self.max_brightness_value = max_b if max_b is not None else 100.0
            self.min_entropy_value = min_e if min_e is not None else 0.0
            self.max_entropy_value = max_e if max_e is not None else 10.0

           # 1. Get the indices of images that pass the quality check
            current_images = [item["img"] for item in data_pool]
            passing_indices = self.__filter_optimum_brightness_and_contrast_images__(current_images)
            
            # 2. Extract the passing data (Paths + Images) using those indices
            filtered_data_pool = [data_pool[i] for i in passing_indices]

            # 3. Check if we have enough to stop the "relaxed filter" loop
            if len(filtered_data_pool) >= number_of_frames:
                break


        # 3. DIVERSITY CLUSTERING
        final_selected_paths = []

        if len(filtered_data_pool) >= self.nb_clusters:
            # Extract just the images that passed the quality filter for clustering
            images_for_clustering = [item["img"] for item in filtered_data_pool]
            
            # Get cluster indices based on histograms
            files_clusters_index_array = self.__prepare_cluster_sets__(images_for_clustering)
            
            # Get the best image indices (sharpest in each cluster)
            selected_indices = self.__get_best_images_index_from_each_cluster__(
                images_for_clustering, files_clusters_index_array
            )

            # Map indices back to the 'path' field
            for index in selected_indices:
                final_selected_paths.append(filtered_data_pool[index]["path"])
        
        else:
            # FALLBACK: If we have fewer frames than clusters, take all that passed quality filtering
            for item in filtered_data_pool:
                final_selected_paths.append(item["path"])

        return [f[-file_name_len:] for f in final_selected_paths]
